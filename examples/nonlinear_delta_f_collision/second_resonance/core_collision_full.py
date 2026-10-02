"""Mixed constant-coefficient diffusion, drag, and Krook delta-f backend.

C(delta_f) = D*d_vv(delta_f) + A*d_v(delta_f) - beta*delta_f,
D=nu**3 >= 0, A=alpha**2 >= 0, beta>=0; Gaussian stationary F0.
Collision transition: dv=-A*h+sqrt(2*D*h)*normal, w'=exp(-beta*h)*w,
p'=p*F0(v+dv)/F0(v). Background-weighted moments preserve F0 in expectation.
With diffusion, weights are stochastic, not single-valued ratios to the
marginal marker density. The field kick uses an exact F0 ratio and expm1.
Only nonlinear delta-f is supported: 'full' means all three collisions,
not full-f or a general nonlinear Landau operator.
CPU API: run_timestepping(params); GPU: run_timestepping(params, backend="gpu").
CLI: python core_collision_full.py --backend cpu|gpu --params ... --output ...
GPU/MPI companion: core_collision_full_gpu.py. Existing files stay unchanged.
Optional params['modes'] enables the secondary-resonance traveling-wave extension.
See README.md for its configuration, diagnostics, and restart policies.
"""

from __future__ import annotations

import os
import signal
import tempfile
import warnings
from time import monotonic

import numpy as np
from numba import njit, prange
from wisp.diagnostics import compute_gamma

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None



WEIGHT_SCHEME = "constant_mixed_stochastic_weights_v1"


def _validate_full_mode(method, full_f, nu, beta, dt, vb, alpha):
    if method != "nonlinear" or full_f:
        raise NotImplementedError("Mixed-collision backend requires nonlinear delta-f mode")
    if not all(np.isfinite(value) for value in (nu, beta, dt, vb, alpha)):
        raise ValueError("Collision coefficients, dt, and vb must be finite")
    if nu < 0.0 or beta < 0.0 or alpha < 0.0 or dt <= 0.0 or vb <= 0.0:
        raise ValueError("Require nu, alpha, beta >= 0; dt, vb > 0")


def _require_full_checkpoint(scheme):
    if str(np.asarray(scheme).item()) != WEIGHT_SCHEME:
        raise ValueError(
            "Checkpoint uses incompatible collision weights. "
            "Start a fresh run with the mixed-collision backend."
        )


@njit(inline="always", cache=True)
def _mixed_collision_push(vi, wi, pi, h, ub, inv_vb2, krook_factor, nu_drag,
                          diffusion_sigma, normal):
    delta_v = -nu_drag * h + diffusion_sigma * normal
    exponent = -delta_v * (2.0 * (vi - ub) + delta_v) * inv_vb2
    return vi + delta_v, wi * krook_factor, pi * np.exp(exponent)


@njit(inline="always", cache=True)
def _field_weight_push(vi, wi, pi, delta_v, ub, inv_vb2):
    exponent = -delta_v * (2.0 * (vi - ub) + delta_v) * inv_vb2
    change = pi * np.expm1(exponent)
    return wi - change, pi * np.exp(exponent)


_TWO_PI = 2.0 * np.pi
_UINT32_MASK = np.uint64(0xFFFFFFFF)
_PHILOX_M0 = np.uint32(0xD2511F53)
_PHILOX_M1 = np.uint32(0xCD9E8D57)
_PHILOX_W0 = np.uint32(0x9E3779B9)
_PHILOX_W1 = np.uint32(0xBB67AE85)
_TWO_NEG_53 = 1.0 / 9007199254740992.0


def _progress_iter(iterable, enabled=True, **kwargs):
    if enabled and tqdm is not None:
        return tqdm(iterable, **kwargs)
    return iterable


@njit(inline="always", cache=True)
def _mulhilo32(a, b):
    product = np.uint64(a) * np.uint64(b)
    low = np.uint32(product & _UINT32_MASK)
    high = np.uint32(product >> np.uint64(32))
    return high, low


@njit(inline="always", cache=True)
def _philox4x32_10(particle_index, step_index, seed):
    """Return four Philox words for one global particle and full step."""
    particle = np.uint64(particle_index)
    step = np.uint64(step_index)
    seed64 = np.uint64(seed)

    c0 = np.uint32(particle & _UINT32_MASK)
    c1 = np.uint32(particle >> np.uint64(32))
    c2 = np.uint32(step & _UINT32_MASK)
    c3 = np.uint32(step >> np.uint64(32))
    k0 = np.uint32(seed64 & _UINT32_MASK)
    k1 = np.uint32(seed64 >> np.uint64(32))

    for _ in range(10):
        hi0, lo0 = _mulhilo32(_PHILOX_M0, c0)
        hi1, lo1 = _mulhilo32(_PHILOX_M1, c2)
        n0 = np.uint32(hi1 ^ c1 ^ k0)
        n1 = lo1
        n2 = np.uint32(hi0 ^ c3 ^ k1)
        n3 = lo0
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 = np.uint32(k0 + _PHILOX_W0)
        k1 = np.uint32(k1 + _PHILOX_W1)
    return c0, c1, c2, c3


@njit(inline="always", cache=True)
def _uniform53(high_word, low_word):
    mantissa = (
        (np.uint64(high_word) >> np.uint64(5)) * np.uint64(67108864)
        + (np.uint64(low_word) >> np.uint64(6))
    )
    return (float(mantissa) + 0.5) * _TWO_NEG_53


@njit(inline="always", cache=True)
def _normal_pair(particle_index, step_index, seed):
    """Two independent standard normals from one Philox counter."""
    r0, r1, r2, r3 = _philox4x32_10(particle_index, step_index, seed)
    u1 = _uniform53(r0, r1)
    u2 = _uniform53(r2, r3)
    radius = np.sqrt(-2.0 * np.log(u1))
    angle = _TWO_PI * u2
    return radius * np.cos(angle), radius * np.sin(angle)


@njit(inline="always", cache=True)
def _wrap_two_pi(value):
    return value - _TWO_PI * np.floor(value / _TWO_PI)


@njit(parallel=True, cache=True)
def _parallel_state_copy(x_source, v_source, w_source, p_source):
    """First-touch working arrays in parallel for NUMA placement."""
    count = len(x_source)
    x = np.empty_like(x_source)
    v = np.empty_like(v_source)
    w = np.empty_like(w_source)
    p = np.empty_like(p_source)
    for i in prange(count):
        x[i] = x_source[i]
        v[i] = v_source[i]
        w[i] = w_source[i]
        p[i] = p_source[i]
    return x, v, w, p


@njit(parallel=True, cache=True)
def _strang_first_fused(
    x,
    v,
    w,
    p,
    second_normal,
    e_c,
    e_s,
    t,
    dt,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
    evolve_weights,
    collisions_enabled,
    rng_seed,
    rng_step,
    particle_offset,
):
    count = len(x)
    s_cos = 0.0
    s_sin = 0.0
    phase_shift = t + 0.25 * dt
    impulse_factor = -2.0 * np.sin(0.25 * dt)

    for i in prange(count):
        xi = x[i]
        vi = v[i]
        wi = w[i]
        pi = p[i]

        if collisions_enabled:
            z_first, z_second = _normal_pair(
                particle_offset + i, rng_step, rng_seed
            )
            second_normal[i] = z_second
            vi, wi, pi = _mixed_collision_push(
                vi, wi, pi, 0.5 * dt, ub, inv_vb2, krook_factor, nu_drag,
                diffusion_sigma, z_first
            )

        phase = xi - phase_shift
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)

        if evolve_weights:
            velocity_impulse = impulse_factor * (
                e_c * phase_cos + e_s * phase_sin
            )
            dw = velocity_impulse * 2.0 * (vi - ub) * inv_vb2
            if nonlinear:
                wi, pi = _field_weight_push(
                    vi, wi, pi, velocity_impulse, ub, inv_vb2
                )
            else:
                wi += dw

        if nonlinear:
            vi += impulse_factor * (e_c * phase_cos + e_s * phase_sin)

        v[i] = vi
        w[i] = wi
        p[i] = pi
        s_cos += vi * wi * phase_cos
        s_sin += vi * wi * phase_sin

    return s_cos, s_sin


@njit(parallel=True, cache=True)
def _position_and_field_fused(x, v, w, t, dt):
    count = len(x)
    s_cos = 0.0
    s_sin = 0.0
    phase_shift = t + 0.75 * dt
    for i in prange(count):
        xi = _wrap_two_pi(x[i] + v[i] * dt)
        x[i] = xi
        phase = xi - phase_shift
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)
        current = v[i] * w[i]
        s_cos += current * phase_cos
        s_sin += current * phase_sin
    return s_cos, s_sin


@njit(parallel=True, cache=True)
def _strang_last_fused(
    x,
    v,
    w,
    p,
    second_normal,
    e_c,
    e_s,
    t,
    dt,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
    evolve_weights,
    collisions_enabled,
):
    count = len(x)
    phase_shift = t + 0.75 * dt
    impulse_factor = -2.0 * np.sin(0.25 * dt)

    for i in prange(count):
        xi = x[i]
        vi = v[i]
        wi = w[i]
        pi = p[i]
        phase = xi - phase_shift
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)

        if evolve_weights:
            velocity_impulse = impulse_factor * (
                e_c * phase_cos + e_s * phase_sin
            )
            dw = velocity_impulse * 2.0 * (vi - ub) * inv_vb2
            if nonlinear:
                wi, pi = _field_weight_push(
                    vi, wi, pi, velocity_impulse, ub, inv_vb2
                )
            else:
                wi += dw

        if nonlinear:
            vi += impulse_factor * (e_c * phase_cos + e_s * phase_sin)

        if collisions_enabled:
            vi, wi, pi = _mixed_collision_push(
                vi, wi, pi, 0.5 * dt, ub, inv_vb2, krook_factor, nu_drag,
                diffusion_sigma, second_normal[i]
            )

        v[i] = vi
        w[i] = wi
        p[i] = pi


@njit(parallel=True, cache=True)
def _trotter_fused(
    x,
    v,
    w,
    p,
    e_c,
    e_s,
    t,
    dt,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
    evolve_weights,
    collisions_enabled,
    rng_seed,
    rng_step,
    particle_offset,
):
    count = len(x)
    s_cos = 0.0
    s_sin = 0.0
    phase_shift = t + 0.5 * dt
    impulse_factor = -2.0 * np.sin(0.5 * dt)

    for i in prange(count):
        xi = x[i]
        vi = v[i]
        wi = w[i]
        pi = p[i]

        if collisions_enabled:
            z_first, _ = _normal_pair(particle_offset + i, rng_step, rng_seed)
            vi, wi, pi = _mixed_collision_push(
                vi, wi, pi, dt, ub, inv_vb2, krook_factor, nu_drag,
                diffusion_sigma, z_first
            )

        phase = xi - phase_shift
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)

        if evolve_weights:
            velocity_impulse = impulse_factor * (
                e_c * phase_cos + e_s * phase_sin
            )
            dw = velocity_impulse * 2.0 * (vi - ub) * inv_vb2
            if nonlinear:
                wi, pi = _field_weight_push(
                    vi, wi, pi, velocity_impulse, ub, inv_vb2
                )
            else:
                wi += dw

        if nonlinear:
            vi += impulse_factor * (e_c * phase_cos + e_s * phase_sin)

        current = vi * wi
        s_cos += current * phase_cos
        s_sin += current * phase_sin
        x[i] = _wrap_two_pi(xi + vi * dt)
        v[i] = vi
        w[i] = wi
        p[i] = pi

    return s_cos, s_sin


def _damped_field_from_sums(
    e_c, e_s, s_cos, s_sin, ds, particle_count, nb_over_ne, gamma_d
):
    half_damping = np.exp(-gamma_d * ds / 2.0) if gamma_d != 0.0 else 1.0
    coefficient = 2.0 * np.sin(ds / 2.0) * nb_over_ne / particle_count
    return (
        (e_c * half_damping + coefficient * s_cos) * half_damping,
        (e_s * half_damping + coefficient * s_sin) * half_damping,
    )


def _pack_rng_state(rng):
    return np.array(rng.bit_generator.state, dtype=object)


def _unpack_rng_state(rng_state):
    if isinstance(rng_state, np.ndarray):
        return rng_state.item()
    return rng_state


def _atomic_savez(filename, outputs):
    filename = os.fspath(filename)
    directory = os.path.dirname(os.path.abspath(filename))
    os.makedirs(directory, exist_ok=True)
    fd, temporary_filename = tempfile.mkstemp(
        prefix=f".{os.path.basename(filename)}.", suffix=".npz", dir=directory
    )
    os.close(fd)
    try:
        np.savez(temporary_filename, **outputs)
        with open(temporary_filename, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_filename, filename)
    finally:
        if os.path.exists(temporary_filename):
            os.unlink(temporary_filename)


def _seed_to_uint64(seed, rng):
    if seed is None:
        return np.uint64(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
    return np.uint64(int(seed) & ((1 << 64) - 1))


def _make_outputs(
    e_c,
    e_s,
    x,
    v,
    w,
    p,
    t_start,
    dt,
    e_c_hist,
    e_s_hist,
    e_amp_hist,
    completed_steps,
    rng,
    rng_seed,
    rng_step_start,
    stopped_early,
):
    return {
        "time": t_start + np.arange(completed_steps) * dt,
        "t_start": t_start,
        "t_final": t_start + completed_steps * dt,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "E_c": e_c,
        "E_s": e_s,
        "x": x,
        "v": v,
        "w": w,
        "p": p,
        "E_c_hist": e_c_hist[:completed_steps],
        "E_s_hist": e_s_hist[:completed_steps],
        "E_amp_hist": e_amp_hist[:completed_steps],
        "phi_hist": np.arctan2(
            e_s_hist[:completed_steps], e_c_hist[:completed_steps]
        ),
        "rng_state": _pack_rng_state(rng),
        "rng_seed": np.uint64(rng_seed),
        "rng_step": np.uint64(rng_step_start + completed_steps),
        "backend": np.array("cpu_fused"),
        "collision_weight_scheme": np.array(WEIGHT_SCHEME),
    }


def run_cpu_timestepping(sim_params):
    """Run a fused CPU timestep chunk with restart-safe Philox collisions."""
    if "modes" in sim_params:
        from secondary_runner import run_multimode
        return run_multimode(sim_params, backend="cpu")
    e_c = float(sim_params["E_c"])
    e_s = float(sim_params["E_s"])
    nb_over_ne = float(sim_params["nb_over_ne"])
    n_steps = int(sim_params["n_steps"])
    dt = float(sim_params["dt"])
    n_particles = int(sim_params["n_particles"])
    ub = float(sim_params["ub"])
    vb = float(sim_params["vb"])
    splitting_method = sim_params["splitting_method"]
    method = sim_params["method"]
    for key in ("full_f", "show_progress"):
        if key in sim_params and not isinstance(sim_params[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a JSON boolean, not a string or number")
    full_f = bool(sim_params["full_f"])
    nonlinear = method == "nonlinear"
    evolve_weights = not full_f

    if splitting_method not in {"strang", "trotter"}:
        raise ValueError("splitting_method must be 'strang' or 'trotter'")
    if vb == 0.0:
        raise ValueError("vb must be non-zero")
    if n_steps < 0 or n_particles <= 0:
        raise ValueError("n_steps must be non-negative and n_particles positive")

    gamma_ratio = float(sim_params.get("gamma_ratio", 0.9))
    gamma_l = float(compute_gamma(sim_params))
    gamma_d = float(sim_params.get("gamma_d", gamma_ratio * gamma_l))
    nu_norm = float(sim_params.get("nu_norm", 0.0))
    nu = float(sim_params.get("nu", (1.0 - gamma_ratio) * nu_norm * gamma_l))
    nu_diff = nu**3
    alpha_norm = float(sim_params.get("alpha_norm", 0.0))
    alpha = float(
        sim_params.get("alpha", (1.0 - gamma_ratio) * alpha_norm * gamma_l)
    )
    nu_drag = alpha**2
    beta_norm = float(sim_params.get("beta_norm", 0.0))
    nu_krook = float(
        sim_params.get("beta", (1.0 - gamma_ratio) * beta_norm * gamma_l)
    )
    _validate_full_mode(method, full_f, nu, nu_krook, dt, vb, alpha)
    collisions_enabled = evolve_weights and (
        nu_krook != 0.0 or nu_drag != 0.0 or nu_diff != 0.0
    )

    print("Backend: mixed-collision Numba CPU")
    print("Damping-to-growth ratio: ", gamma_ratio)
    print("Wave damping rate: ", gamma_d)
    print("Normalized diffusion rate: ", nu_norm)
    print("Diffusion rate: ", nu_diff)

    checkpoint_file = sim_params.get("checkpoint_file")
    checkpoint_interval = float(sim_params.get("checkpoint_interval", 0.0) or 0.0)
    if checkpoint_interval < 0.0:
        raise ValueError("checkpoint_interval must be non-negative")

    seed = sim_params.get("seed")
    rng = np.random.default_rng(seed)
    t_start = float(sim_params.get("t_start", 0.0))
    rng_step_start = 0

    restart_file = sim_params.get("restart_file")
    if restart_file is not None:
        with np.load(restart_file, allow_pickle=True) as state:
            _require_full_checkpoint(state.get("collision_weight_scheme", ""))
            e_c = float(state["E_c"])
            e_s = float(state["E_s"])
            x_source = state["x"].copy()
            v_source = state["v"].copy()
            w_source = state["w"].copy()
            if "p" not in state:
                raise ValueError("Mixed-collision checkpoint must contain background weights p")
            p_source = state["p"].copy()
            if "t_final" in state and "t_start" not in sim_params:
                t_start = float(state["t_final"])
            elif "time" in state and len(state["time"]) and "t_start" not in sim_params:
                t_start = float(state["time"][-1] + dt)
            if "rng_state" in state:
                rng.bit_generator.state = _unpack_rng_state(state["rng_state"])
            if "rng_seed" in state:
                rng_seed = np.uint64(state["rng_seed"])
            else:
                if seed is None:
                    raise ValueError(
                        "A legacy checkpoint has no rng_seed; supply sim_params['seed'] "
                        "to begin the optimized Philox stream explicitly."
                    )
                warnings.warn(
                    "Legacy checkpoint detected: collision RNG changes to Philox at restart.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                rng_seed = _seed_to_uint64(seed, rng)
            rng_step_start = int(
                state["rng_step"]
                if "rng_step" in state
                else state.get("completed_steps", 0)
            )
        n_particles = len(x_source)
    else:
        x_source = rng.uniform(0.0, _TWO_PI, n_particles)
        v_source = rng.normal(ub, vb / np.sqrt(2.0), n_particles)
        w_source = np.ones(n_particles) if full_f else np.zeros(n_particles)
        p_source = np.ones(n_particles)
        rng_seed = _seed_to_uint64(seed, rng)

    x, v, w, p = _parallel_state_copy(x_source, v_source, w_source, p_source)
    del x_source, v_source, w_source, p_source
    second_normal = np.empty(n_particles, dtype=np.float64)

    time = t_start + np.arange(n_steps) * dt
    e_c_hist = np.zeros(n_steps)
    e_s_hist = np.zeros(n_steps)
    e_amp_hist = np.zeros(n_steps)
    completed_steps = 0
    stop_requested = False

    inv_vb2 = 1.0 / (vb * vb)
    if splitting_method == "strang":
        collision_ds = 0.5 * dt
    else:
        collision_ds = dt
    krook_factor = np.exp(-nu_krook * collision_ds)
    source_factor = (
        (1.0 - krook_factor) / nu_krook if nu_krook > 0.0 else collision_ds
    )
    diffusion_sigma = (
        np.sqrt(2.0 * nu_diff * collision_ds) if nu_diff > 0.0 else 0.0
    )

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    installed_signal_handlers = {}
    for signal_number in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if signal_number is None:
            continue
        try:
            installed_signal_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_stop)
        except (OSError, TypeError, ValueError):
            continue

    next_checkpoint = (
        monotonic() + checkpoint_interval if checkpoint_interval > 0.0 else None
    )
    show_progress = bool(
        sim_params.get("show_progress", sim_params.get("progress_bar", False))
    )
    step_iter = _progress_iter(
        range(n_steps),
        enabled=show_progress,
        total=n_steps,
        desc="CPU timestepping",
        unit="step",
    )

    try:
        for n in step_iter:
            if stop_requested:
                break
            t = time[n]
            rng_step = rng_step_start + n

            if splitting_method == "strang":
                s_cos, s_sin = _strang_first_fused(
                    x,
                    v,
                    w,
                    p,
                    second_normal,
                    e_c,
                    e_s,
                    t,
                    dt,
                    ub,
                    inv_vb2,
                    krook_factor,
                    source_factor,
                    nu_drag,
                    nu_diff,
                    diffusion_sigma,
                    nonlinear,
                    evolve_weights,
                    collisions_enabled,
                    rng_seed,
                    rng_step,
                    0,
                )
                e_c, e_s = _damped_field_from_sums(
                    e_c,
                    e_s,
                    s_cos,
                    s_sin,
                    0.5 * dt,
                    n_particles,
                    nb_over_ne,
                    gamma_d,
                )
                s_cos, s_sin = _position_and_field_fused(x, v, w, t, dt)
                e_c, e_s = _damped_field_from_sums(
                    e_c,
                    e_s,
                    s_cos,
                    s_sin,
                    0.5 * dt,
                    n_particles,
                    nb_over_ne,
                    gamma_d,
                )
                _strang_last_fused(
                    x,
                    v,
                    w,
                    p,
                    second_normal,
                    e_c,
                    e_s,
                    t,
                    dt,
                    ub,
                    inv_vb2,
                    krook_factor,
                    source_factor,
                    nu_drag,
                    nu_diff,
                    diffusion_sigma,
                    nonlinear,
                    evolve_weights,
                    collisions_enabled,
                )
            else:
                s_cos, s_sin = _trotter_fused(
                    x,
                    v,
                    w,
                    p,
                    e_c,
                    e_s,
                    t,
                    dt,
                    ub,
                    inv_vb2,
                    krook_factor,
                    source_factor,
                    nu_drag,
                    nu_diff,
                    diffusion_sigma,
                    nonlinear,
                    evolve_weights,
                    collisions_enabled,
                    rng_seed,
                    rng_step,
                    0,
                )
                e_c, e_s = _damped_field_from_sums(
                    e_c,
                    e_s,
                    s_cos,
                    s_sin,
                    dt,
                    n_particles,
                    nb_over_ne,
                    gamma_d,
                )

            e_c_hist[n] = e_c
            e_s_hist[n] = e_s
            e_amp_hist[n] = np.hypot(e_c, e_s)
            completed_steps = n + 1

            if stop_requested:
                break
            if (
                checkpoint_file is not None
                and next_checkpoint is not None
                and monotonic() >= next_checkpoint
            ):
                _atomic_savez(
                    checkpoint_file,
                    _make_outputs(
                        e_c,
                        e_s,
                        x,
                        v,
                        w,
                        p,
                        t_start,
                        dt,
                        e_c_hist,
                        e_s_hist,
                        e_amp_hist,
                        completed_steps,
                        rng,
                        rng_seed,
                        rng_step_start,
                        False,
                    ),
                )
                next_checkpoint = monotonic() + checkpoint_interval
    finally:
        for signal_number, previous_handler in installed_signal_handlers.items():
            signal.signal(signal_number, previous_handler)

    outputs = _make_outputs(
        e_c,
        e_s,
        x,
        v,
        w,
        p,
        t_start,
        dt,
        e_c_hist,
        e_s_hist,
        e_amp_hist,
        completed_steps,
        rng,
        rng_seed,
        rng_step_start,
        completed_steps < n_steps,
    )
    if checkpoint_file is not None:
        _atomic_savez(checkpoint_file, outputs)
    return outputs


def run_timestepping(sim_params, backend=None):
    """Dispatch explicitly to CPU or the isolated CUDA/MPI companion."""
    backend = backend or sim_params.get("backend", "cpu")
    if backend == "gpu":
        from core_collision_full_gpu import run_timestepping as run_gpu
        return run_gpu(sim_params)
    if backend != "cpu":
        raise ValueError("backend must be 'cpu' or 'gpu'")
    return run_cpu_timestepping(sim_params)


__all__ = ["run_timestepping", "run_cpu_timestepping"]


def _run_full_cli(run=None, save=None, backend=None):
    """Direct entry point; keeps the original driver and job scripts intact."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default=backend or "cpu")
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--checkpoint-interval", type=float)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    backend = args.backend
    if backend == "gpu":
        from core_collision_full_gpu import run_timestepping as run, save_ranked_outputs as save
    else:
        run, save = run_cpu_timestepping, _atomic_savez
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; choose a new path")
    if args.checkpoint and args.checkpoint.exists() and not args.restart and not args.overwrite:
        raise FileExistsError(f"Checkpoint exists: {args.checkpoint}; choose a new path")
    with args.params.open() as stream:
        params = json.load(stream)
    for key, value in (("checkpoint_file", args.checkpoint), ("restart_file", args.restart)):
        if value is not None:
            params[key] = str(value)
    if args.checkpoint_interval is not None:
        params["checkpoint_interval"] = args.checkpoint_interval
    if backend == "gpu":
        params["expected_gpus"] = args.expected_gpus
    outputs = run(params)
    save(args.output, outputs)
    if int(outputs.get("rank", 0)) == 0:
        print(f"Saved {outputs['completed_steps']} steps to {args.output}")


if __name__ == "__main__":
    _run_full_cli()
