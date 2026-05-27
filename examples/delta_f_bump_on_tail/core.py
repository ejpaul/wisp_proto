"""WISP core simulation engine — Numba-parallelized.

Particle pushers use @njit(parallel=True) with prange for thread-level
parallelism.  The thread count is controlled by the environment variable
NUMBA_NUM_THREADS, which must be set BEFORE importing this module.

If Numba is not installed the module still works (pure-Python fallback),
but without any parallelization or JIT compilation.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Numba import with graceful fallback
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Provide no-op decorator and let prange = range
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper
    prange = range


# ---------------------------------------------------------------------------
# Particle pushers
# ---------------------------------------------------------------------------

@njit(parallel=True)
def push_particles_nonlinear_full_f(x, v, E_c, E_s, t, dt):
    """Advance particles — nonlinear full-f (no weights)."""
    N = len(x)
    for i in prange(N):
        phase = x[i] - t
        E_local = E_c * np.cos(phase) + E_s * np.sin(phase)
        vdot = -E_local
        x[i] = x[i] + v[i] * dt
        v[i] = v[i] + vdot * dt
        x[i] = x[i] % (2.0 * np.pi)
    return x, v


@njit(parallel=True)
def push_particles_nonlinear_delta_f(x, v, w, E_c, E_s, t, dt, ub, vb):
    """Advance particles — nonlinear delta-f (weight evolution with (1-w) correction)."""
    N = len(x)
    for i in prange(N):
        phase = x[i] - t
        E_local = E_c * np.cos(phase) + E_s * np.sin(phase)
        vdot = -E_local
        w[i] = w[i] + (1.0 - w[i]) * vdot * 2.0 * (v[i] - ub) / vb**2 * dt
        x[i] = x[i] + v[i] * dt
        v[i] = v[i] + vdot * dt
        x[i] = x[i] % (2.0 * np.pi)
    return x, v, w


@njit(parallel=True)
def push_particles_linear(x, v, w, E_c, E_s, t, dt, ub, vb):
    """Advance particles — linear delta-f (no velocity update, no (1-w))."""
    N = len(x)
    for i in prange(N):
        phase = x[i] - t
        E_local = E_c * np.cos(phase) + E_s * np.sin(phase)
        vdot = -E_local
        w[i] = w[i] + vdot * 2.0 * (v[i] - ub) / vb**2 * dt
        x[i] = x[i] + v[i] * dt
        x[i] = x[i] % (2.0 * np.pi)
    return x, v, w


# ---------------------------------------------------------------------------
# Field update (current projection)
# ---------------------------------------------------------------------------

@njit(parallel=True)
def compute_current_fourier(x, v, w, t, nb_over_ne):
    """Fourier projection of beam current onto the single mode.

    Returns (dE_c/dt, dE_s/dt).
    """
    N = len(x)
    j_cos = 0.0
    j_sin = 0.0
    for i in prange(N):
        phase = x[i] - t
        j_cos += v[i] * w[i] * np.cos(phase)
        j_sin += v[i] * w[i] * np.sin(phase)
    dE_c_dt = nb_over_ne * j_cos / N
    dE_s_dt = -nb_over_ne * j_sin / N
    return dE_c_dt, dE_s_dt


@njit
def update_field(E_c, E_s, x, v, w, t, dt, nb_over_ne):
    """Update E_c, E_s by one full step."""
    dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t, nb_over_ne)
    return E_c + dE_c_dt * dt, E_s + dE_s_dt * dt


# ---------------------------------------------------------------------------
# Particle initialization helpers
# ---------------------------------------------------------------------------

def init_delta_f_markers_uniform_v(n_particles, vmin, vmax, *, seed=None):
    """Initialize delta-f markers with uniform velocity sampling."""
    if vmax <= vmin:
        raise ValueError(f"vmax must be > vmin, got vmin={vmin}, vmax={vmax}")
    rng = np.random.default_rng(seed)
    v = rng.uniform(vmin, vmax, n_particles)
    w = np.zeros(n_particles)
    return v, w


# ---------------------------------------------------------------------------
# Main time-stepping loop
# ---------------------------------------------------------------------------

def run_timestepping(sim_params, show_progress=False):
    """Run the simulation and return history arrays.

    Parameters
    ----------
    sim_params : dict
        Required keys: E_c, E_s, nb_over_ne, n_steps, dt, n_particles,
                        ub, vb, splitting_method, method, full_f
        Optional keys: seed, v_sampling ("f0"|"uniform"), vmin, vmax
    show_progress : bool
        If True, display a tqdm progress bar over time steps.

    Returns
    -------
    dict with: time, E_c, E_s, x, v, w,
               E_c_hist, E_s_hist, E_amp_hist, phi_hist
    """
    E_c = sim_params["E_c"]
    E_s = sim_params["E_s"]
    nb_over_ne = sim_params["nb_over_ne"]
    n_steps = sim_params["n_steps"]
    dt = sim_params["dt"]
    n_particles = sim_params["n_particles"]
    ub = sim_params["ub"]
    vb = sim_params["vb"]
    splitting_method = sim_params["splitting_method"]
    method = sim_params["method"]
    full_f = sim_params["full_f"]

    seed = sim_params.get("seed", None)
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n_particles)

    if full_f:
        v = rng.normal(ub, vb / np.sqrt(2), n_particles)
        w = np.ones(n_particles)
    else:
        if sim_params.get("v_sampling", "f0") == "uniform":
            vmin = sim_params.get("vmin", 0.0)
            vmax = sim_params.get("vmax", 2.0)
            v, w = init_delta_f_markers_uniform_v(n_particles, vmin, vmax, seed=seed)
        else:
            v = rng.normal(ub, vb / np.sqrt(2), n_particles)
            w = np.zeros(n_particles)

    time = np.arange(n_steps) * dt
    E_c_hist = np.zeros(n_steps)
    E_s_hist = np.zeros(n_steps)
    E_amp_hist = np.zeros(n_steps)

    # Optional progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            step_iter = tqdm(range(n_steps), desc="Stepping", unit="step", ncols=90)
        except ImportError:
            step_iter = range(n_steps)
    else:
        step_iter = range(n_steps)

    for n in step_iter:
        t = time[n]

        if splitting_method == "strang":
            # Half field → full particles → half field
            dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t, nb_over_ne)
            E_c = E_c + dE_c_dt * (dt / 2)
            E_s = E_s + dE_s_dt * (dt / 2)

            if method == "linear":
                x, v, w = push_particles_linear(x, v, w, E_c, E_s, t, dt, ub, vb)
            else:
                if full_f:
                    x, v = push_particles_nonlinear_full_f(x, v, E_c, E_s, t, dt)
                else:
                    x, v, w = push_particles_nonlinear_delta_f(
                        x, v, w, E_c, E_s, t, dt, ub, vb
                    )

            dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t + dt, nb_over_ne)
            E_c = E_c + dE_c_dt * (dt / 2)
            E_s = E_s + dE_s_dt * (dt / 2)

        elif splitting_method == "trotter":
            if method == "linear":
                x, v, w = push_particles_linear(x, v, w, E_c, E_s, t, dt, ub, vb)
            else:
                if full_f:
                    x, v = push_particles_nonlinear_full_f(x, v, E_c, E_s, t, dt)
                else:
                    x, v, w = push_particles_nonlinear_delta_f(
                        x, v, w, E_c, E_s, t, dt, ub, vb
                    )

            dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t, nb_over_ne)
            E_c = E_c + dE_c_dt * dt
            E_s = E_s + dE_s_dt * dt
        else:
            raise ValueError(f"Unknown splitting_method: {splitting_method!r}")

        E_c_hist[n] = E_c
        E_s_hist[n] = E_s
        E_amp_hist[n] = np.sqrt(E_c**2 + E_s**2)

    phi_hist = np.arctan2(E_s_hist, E_c_hist)

    return {
        "time": time,
        "E_c": E_c,
        "E_s": E_s,
        "x": x,
        "v": v,
        "w": w,
        "E_c_hist": E_c_hist,
        "E_s_hist": E_s_hist,
        "E_amp_hist": E_amp_hist,
        "phi_hist": phi_hist,
    }