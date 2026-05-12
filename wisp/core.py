import numpy as np
from numba import njit, prange

@njit(parallel=True)
def push_particles_nonlinear_full_f(x, v, E_c, E_s, t, dt):
    """Advance particles with frozen field (E_c, E_s formulation)."""
    N = len(x)
    for i in prange(N):
        # E(x,t) = E_c * cos(x - t) + E_s * sin(x - t)
        phase = x[i] - t
        E_local = E_c * np.cos(phase) + E_s * np.sin(phase)
        vdot = -E_local
        x[i] = x[i] + v[i] * dt
        v[i] = v[i] + vdot * dt
        # Apply periodic boundary conditions
        x[i] = x[i] % (2 * np.pi)
    return x, v

@njit(parallel=True)
def push_particles_nonlinear_delta_f(x, v, w, E_c, E_s, t, dt, ub, vb):
    """Advance particles with frozen field (E_c, E_s formulation)."""
    N = len(x)
    for i in prange(N):
        # E(x,t) = E_c * cos(x - t) + E_s * sin(x - t)
        phase = x[i] - t
        E_local = E_c * np.cos(phase) + E_s * np.sin(phase)
        vdot = -E_local
        w[i] = w[i] + (1 - w[i]) * vdot * 2 * (v[i] - ub) / vb**2 * dt
        x[i] = x[i] + v[i] * dt
        v[i] = v[i] + vdot * dt
        # Apply periodic boundary conditions
        x[i] = x[i] % (2 * np.pi)
    return x, v, w


@njit(parallel=True)
def push_particles_linear(x, v, w, E_c, E_s, t, dt, ub, vb):
    """Advance particles with frozen field (E_c, E_s formulation, linear)."""
    N = len(x)
    for i in prange(N):
        # E(x,t) = E_c * cos(x - t) + E_s * sin(x - t)
        phase = x[i] - t
        E_local = E_c * np.cos(phase) + E_s * np.sin(phase)
        vdot = -E_local
        w[i] = w[i] + vdot * 2 * (v[i] - ub) / vb**2 * dt
        x[i] = x[i] + v[i] * dt
        # Apply periodic boundary conditions
        x[i] = x[i] % (2 * np.pi)
    return x, v, w


@njit(parallel=True)
def compute_current_fourier(x, v, w, t, nb_over_ne):
    """Compute Fourier components of current for E_c and E_s evolution.

    Returns:
        dE_c_dt: time derivative of E_c = (nb/ne)(1/N) sum_i v_i w_i cos(x_i - t)
        dE_s_dt: time derivative of E_s = -(nb/ne)(1/N) sum_i v_i w_i sin(x_i - t)
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
    """Update field components E_c and E_s."""
    dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t, nb_over_ne)
    E_c_new = E_c + dE_c_dt * dt
    E_s_new = E_s + dE_s_dt * dt
    return E_c_new, E_s_new


def run_timestepping(sim_params):
    """Run the main time-stepping loop and return simulation outputs."""
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

    x = np.random.uniform(0, 2 * np.pi, n_particles)
    v = np.random.normal(ub, vb / np.sqrt(2), n_particles)
    if full_f:
        w = np.ones(n_particles)
    else:
        w = np.zeros(n_particles)

    time = np.arange(n_steps) * dt
    E_c_hist = np.zeros(n_steps)
    E_s_hist = np.zeros(n_steps)
    E_amp_hist = np.zeros(n_steps)

    for n in range(n_steps):
        t = time[n]

        if splitting_method == "strang":
            # Step 1: Half step for field (particles at time t)
            dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t, nb_over_ne)
            E_c = E_c + dE_c_dt * (dt / 2)
            E_s = E_s + dE_s_dt * (dt / 2)

            # Step 2: Full step for particles
            if method == "linear":
                x, v, w = push_particles_linear(x, v, w, E_c, E_s, t, dt, ub, vb)
            else:
                if full_f:
                    x, v = push_particles_nonlinear_full_f(x, v, E_c, E_s, t, dt)
                else:
                    x, v, w = push_particles_nonlinear_delta_f(x, v, w, E_c, E_s, t, dt, ub, vb)

            # Step 3: Half step for field (particles now at time t+dt)
            dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t + dt, nb_over_ne)
            E_c = E_c + dE_c_dt * (dt / 2)
            E_s = E_s + dE_s_dt * (dt / 2)
        elif splitting_method == "trotter":
            # Step 1: full step for particles
            if method == "linear":
                x, v, w = push_particles_linear(x, v, w, E_c, E_s, t, dt, ub, vb)
            else:
                if full_f:
                    x, v = push_particles_nonlinear_full_f(x, v, E_c, E_s, t, dt)
                else:
                    x, v, w = push_particles_nonlinear_delta_f(x, v, w, E_c, E_s, t, dt, ub, vb)

            # Step 2: full step for field
            dE_c_dt, dE_s_dt = compute_current_fourier(x, v, w, t, nb_over_ne)
            E_c = E_c + dE_c_dt * dt
            E_s = E_s + dE_s_dt * dt

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
