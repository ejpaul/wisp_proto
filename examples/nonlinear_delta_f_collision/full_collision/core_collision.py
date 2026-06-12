import numpy as np
from numba import njit, prange


# ---------------------------------------------------------------------------
# Exact sub-operators for three-way operator splitting (Strang and Trotter).
#
# The idea here is different from the pushers above. We split the dynamics into three 
# simple pieces, move the particles (X), kick the weights (W), and kick the field (F), 
# and solve each little piece exactly over its sub-interval.
# ---------------------------------------------------------------------------

@njit(parallel=True)
def position_push(x, v, ds):
    """Exact position sub-operator F^(1): x_i -> (x_i + v_i * ds) mod 2pi.

    Pure drift: every particle just coasts at its own velocity for a time ds.
    Nothing else changes, so this one is genuinely exact with no tricks.
    """
    N = len(x)
    for i in prange(N):
        x[i] = (x[i] + v[i] * ds) % (2 * np.pi)
    return x


@njit(parallel=True)
def collision_push_particles(v, w, ds, ub, nu_krook, nu_drag, nu_diff, random_normals):
    """Particle-native drag/diffusion/Krook collision sub-operator.

    Velocity is advanced by the SDE

        dv = -nu_drag * (v - ub) dt + sqrt(2 * nu_diff) dW

    using the exact Ornstein-Uhlenbeck update when nu_drag > 0.
    Krook relaxation damps delta-f weights.
    """
    N = len(v)

    krook_factor = np.exp(-nu_krook * ds)

    if nu_drag > 0.0:
        drag_factor = np.exp(-nu_drag * ds)

        if nu_diff > 0.0:
            sigma = np.sqrt((nu_diff / nu_drag) * (1.0 - drag_factor * drag_factor))
        else:
            sigma = 0.0

        for i in prange(N):
            v[i] = ub + (v[i] - ub) * drag_factor + sigma * random_normals[i]
            w[i] = w[i] * krook_factor

    else:
        if nu_diff > 0.0:
            sigma = np.sqrt(2.0 * nu_diff * ds)
        else:
            sigma = 0.0

        for i in prange(N):
            v[i] = v[i] + sigma * random_normals[i]
            w[i] = w[i] * krook_factor

    return v, w


@njit(parallel=True)
def weight_push_exact(x, v, w, E_c, E_s, t, ds, ub, vb, nonlinear=False):
    """Exact weight sub-operator (delta-f analogue of F^(2)), frozen x, E_c, E_s.

    With the positions held fixed, we work out the exact velocity kick a
    particle picks up from the field over the time slice ds, then feed
    that kick into the delta-f weight equation. Linear mode uses dw = vdot * ...
    ; nonlinear mode uses dw = (1 - w) * vdot * ..., matching push_particles_nonlinear_delta_f.
    """
    N = len(x)
    sin_half = np.sin(ds / 2.0)
    inv_vb2 = 1.0 / vb**2
    for i in prange(N):
        phase_mid = x[i] - t - ds / 2.0
        vdot_impulse = -2.0 * sin_half * (E_c * np.cos(phase_mid) + E_s * np.sin(phase_mid))
        dw = vdot_impulse * 2.0 * (v[i] - ub) * inv_vb2
        if nonlinear:
            w[i] = w[i] + (1.0 - w[i]) * dw
        else:
            w[i] = w[i] + dw
    return w


@njit(parallel=True)
def velocity_push_exact(x, v, E_c, E_s, t, ds):
    """Exact velocity sub-operator for nonlinear full-f / delta-f, frozen x, E_c, E_s."""
    N = len(x)
    sin_half = np.sin(ds / 2.0)
    for i in prange(N):
        phase_mid = x[i] - t - ds / 2.0
        vdot_impulse = -2.0 * sin_half * (E_c * np.cos(phase_mid) + E_s * np.sin(phase_mid))
        v[i] = v[i] + vdot_impulse
    return v


@njit(parallel=True)
def field_update_exact(E_c, E_s, x, v, w, t, ds, nb_over_ne):
    """Exact field sub-operator F^(3), frozen x, v, w.

    Mirror image of the weight push: with the particles held fixed, we advance
    the field by the exact contribution the current makes over the slice ds.
    Same `2*sin(ds/2)` exact-integral factor, same midpoint phase. The sum over
    particles is the (cos/sin) Fourier projection of the current j = sum(v*w).
    """
    N = len(x)
    coeff = 2.0 * np.sin(ds / 2.0) * nb_over_ne / N
    s_cos = 0.0
    s_sin = 0.0
    for i in prange(N):
        phase_mid = x[i] - t - ds / 2.0
        s_cos += v[i] * w[i] * np.cos(phase_mid)
        s_sin += v[i] * w[i] * np.sin(phase_mid)
    E_c_new = E_c + coeff * s_cos
    E_s_new = E_s + coeff * s_sin
    return E_c_new, E_s_new


# ---------------------------------------------------------------------------
# Current / field helpers
# ---------------------------------------------------------------------------

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
    nonlinear = method == "nonlinear"
    nu_krook = sim_params.get("nu_krook", sim_params.get("beta", 0.0))
    nu_drag = sim_params.get("nu_drag", 0.0)
    nu_diff = sim_params.get("nu_diff", 0.0)
    collisions_enabled = (nu_krook != 0.0) or (nu_drag != 0.0) or (nu_diff != 0.0)

    # Pass a "seed" in sim_params to get
    # the exact same particle initialisation every run (handy for debugging and
    # for reproducible results in the paper/git history). Leave it out (None)
    # and you get a fresh random draw each time.
    seed = sim_params.get("seed", None)
    rng = np.random.default_rng(seed)

    x = rng.uniform(0, 2 * np.pi, n_particles)
    v = rng.normal(ub, vb / np.sqrt(2), n_particles)

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
            # Proper three-operator Strang split built from the exact
            # sub-operators above. The composition is a palindrome,
            #     W_{dt/2} o F_{dt/2} o X_{dt} o F_{dt/2} o W_{dt/2}
            # (read right-to-left, so the first thing that actually runs is the
            # right-most W). Because the sequence is symmetric, the whole step
            # is second-order accurate in dt. Compared to the old "half field /
            # full push / half field" version, this also kicks the weights
            # exactly and uses the exact sub-operators throughout, so it's a lot
            # cleaner and more accurate.

            # Step 0: optional Krook/drag/diffusion collision kick over [t, t + dt]
            if collisions_enabled and not full_f:
                random_normals = rng.normal(0.0, 1.0, n_particles)
                v, w = collision_push_particles(v, w, dt / 2.0, ub, nu_krook, nu_drag, nu_diff, random_normals)

            # Step 1: half weight / velocity push over [t, t + dt/2]
            if not full_f:
                w = weight_push_exact(
                    x, v, w, E_c, E_s, t, dt / 2.0, ub, vb, nonlinear
                )
            if nonlinear:
                v = velocity_push_exact(x, v, E_c, E_s, t, dt / 2.0)

            # Step 2: half field update over [t, t + dt/2]
            E_c, E_s = field_update_exact(E_c, E_s, x, v, w, t, dt / 2.0, nb_over_ne)

            # Step 3: full position push over [t, t + dt]
            x = position_push(x, v, dt)

            # Step 4: half field update over [t + dt/2, t + dt]
            E_c, E_s = field_update_exact(
                E_c, E_s, x, v, w, t + dt / 2.0, dt / 2.0, nb_over_ne
            )

            # Step 5: half weight / velocity push over [t + dt/2, t + dt]
            if not full_f:
                w = weight_push_exact(
                    x, v, w, E_c, E_s, t + dt / 2.0, dt / 2.0, ub, vb, nonlinear
                )
            if nonlinear:
                v = velocity_push_exact(x, v, E_c, E_s, t + dt / 2.0, dt / 2.0)

            # Step 6: optional Krook/drag/diffusion collision kick over [t, t + dt]
            if collisions_enabled and not full_f:
                random_normals = rng.normal(0.0, 1.0, n_particles)
                v, w = collision_push_particles(v, w, dt / 2.0, ub, nu_krook, nu_drag, nu_diff, random_normals)


        elif splitting_method == "trotter":
            # First-order Lie-Trotter split using the same three exact
            # sub-operators as Strang (weight W, field F, position X), each
            # applied once at full dt in the forward Strang order:
            #     W_{dt} o F_{dt} o X_{dt}
            
            if collisions_enabled and not full_f:
                random_normals = rng.normal(0.0, 1.0, n_particles)
                v, w = collision_push_particles(v, w, dt, ub, nu_krook, nu_drag, nu_diff, random_normals)
            if not full_f:
                w = weight_push_exact(x, v, w, E_c, E_s, t, dt, ub, vb, nonlinear)
            if nonlinear:
                v = velocity_push_exact(x, v, E_c, E_s, t, dt)
            E_c, E_s = field_update_exact(E_c, E_s, x, v, w, t, dt, nb_over_ne)
            x = position_push(x, v, dt)

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