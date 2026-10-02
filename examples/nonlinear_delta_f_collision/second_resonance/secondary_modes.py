"""Mode schema, envelope convention, and reduced-model diagnostics.

E(x,t) = sum_a g_a [Ec_a cos(k_a x - omega_a t)
                         + Es_a sin(k_a x - omega_a t)].
Particle acceleration is -E. Both envelope currents have POSITIVE signs.
With A=Ec+i Es, E_a=Re[A exp(-i psi_a)], so omega_inst=omega+d(arg A)/dt.
"""
from dataclasses import asdict, dataclass
import json

import numpy as np

MODE_SCHEME = "traveling_wave_modes_v1"


@dataclass(frozen=True)
class ModeSpec:
    k: float = 1.0
    omega: float = 1.0
    coupling: float = 1.0
    gamma_d: float = 0.0
    E_c: float = 0.0
    E_s: float = 0.0
    evolve: bool = True
    name: str = ""


def parse_modes(params, default_damping):
    raw = params["modes"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("modes must be a nonempty list of ModeSpec objects or mappings")
    specs = []
    for index, item in enumerate(raw):
        if isinstance(item, ModeSpec):
            mode = item
        else:
            values = dict(item)
            values.setdefault("gamma_d", default_damping)
            values.setdefault("E_c", params.get("E_c", 0.0) if index == 0 else 0.0)
            values.setdefault("E_s", params.get("E_s", 0.0) if index == 0 else 0.0)
            mode = ModeSpec(**values)
        if not isinstance(mode.evolve, (bool, np.bool_)):
            raise ValueError("mode evolve must be a boolean")
        if not isinstance(mode.name, str):
            raise ValueError("mode name must be a string")
        values = [mode.k, mode.omega, mode.coupling, mode.gamma_d, mode.E_c, mode.E_s]
        if not np.all(np.isfinite(values)):
            raise ValueError("All mode coefficients must be finite")
        # Existing marker coordinates are periodic on [0, 2*pi).
        if mode.k <= 0 or mode.k != int(mode.k):
            raise ValueError("k must be a positive integer on the 2*pi periodic domain")
        if mode.gamma_d < 0:
            raise ValueError("mode gamma_d must be nonnegative")
        specs.append(mode)
    return specs


def mode_arrays(specs):
    return tuple(np.ascontiguousarray([getattr(m, key) for m in specs], dtype=float)
                 for key in ("k", "omega", "coupling", "gamma_d", "E_c", "E_s"))


def mode_json(specs):
    return json.dumps([asdict(m) for m in specs], sort_keys=True)


def carrier_integral(omega, ds):
    """Integral over a centered frozen-position kick, including omega=0."""
    return ds * np.sinc(np.asarray(omega) * ds / (2.0 * np.pi))


def advance_fields(ec, es, currents, ds, count, density, specs):
    _, omega, coupling, damping, _, _ = mode_arrays(specs)
    active = np.array([m.evolve for m in specs])
    half = np.exp(-damping * ds / 2.0)
    coeff = carrier_integral(omega, ds) * density * coupling / count
    return (np.where(active, (ec * half + coeff * currents[:, 0]) * half, ec),
            np.where(active, (es * half + coeff * currents[:, 1]) * half, es))


def resonance_diagnostics(ec_hist, es_hist, times, specs):
    """Instantaneous isolated-wave estimates; overlap does not prove chaos."""
    k, omega, coupling, _, _, _ = mode_arrays(specs)
    amp = np.hypot(ec_hist, es_hist)
    phi = np.unwrap(np.arctan2(es_hist, ec_hist), axis=0)
    omega_b = np.sqrt(np.abs(k * coupling) * amp)
    width = 2.0 * omega_b / k  # velocity HALF-width of isolated separatrix
    frequency = np.full_like(amp, np.nan)
    if len(times) >= 2:
        frequency = omega + np.gradient(phi, times, axis=0)
        frequency[amp == 0] = np.nan
    pairs = np.array([(a, b) for a in range(len(specs))
                      for b in range(a + 1, len(specs))], dtype=int).reshape(-1, 2)
    overlap = np.full((len(times), len(pairs)), np.nan)
    for index, (a, b) in enumerate(pairs):
        separation = abs(omega[a] / k[a] - omega[b] / k[b])
        if separation > 0:
            present = (width[:, a] > 0) & (width[:, b] > 0)
            overlap[present, index] = (width[present, a] + width[present, b]) / separation
    return dict(E_amp_hist=amp, phi_hist=phi, omega_inst_hist=frequency,
                omega_b_hist=omega_b, separatrix_half_width_hist=width,
                v_res=omega / k, overlap_pairs=pairs, overlap_hist=overlap)


def velocity_distribution(v, w, p, bins, global_count=None):
    """Normalized F0, delta-f, and f from weighted histograms (never density=True).

    For MPI, sum the returned densities across ranks using global_count=N.
    Explicit edges give comparable distributions across cases and restarts.
    """
    edges = np.asarray(bins, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("bins must be strictly increasing velocity edges")
    count = len(v) if global_count is None else global_count
    scale = count * np.diff(edges)
    background = np.histogram(v, bins=edges, weights=p)[0] / scale
    delta = np.histogram(v, bins=edges, weights=w)[0] / scale
    return dict(v_edges=edges, F0=background, delta_f=delta, f=background + delta)
