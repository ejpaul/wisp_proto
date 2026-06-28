import numpy as np
import matplotlib.pyplot as plt


def plot_delta_f_velocity_distribution(v, w, ub, vb, bins=20, ax=None):
    """Plot velocity histogram with Maxwellian overlay.

    Returns:
        (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    v_grid = np.linspace(ub - 3 * vb, ub + 3 * vb, 100)
    ax.hist(v, weights=w, density=True, bins=bins)
    ax.plot(v_grid, np.exp(-(v_grid - ub) ** 2 / (vb**2)) / np.sqrt(np.pi * vb**2))
    ax.axvline(ub, color="red")
    ax.axvline(1, color="black")
    ax.axvline(ub + vb / np.sqrt(2), color="green")
    ax.axvline(ub - vb / np.sqrt(2), color="green")
    ax.set_xlabel(r"$\frac{v k}{\omega}$")
    return fig, ax


def plot_initial_velocity_distribution(sim_params, bins=20, ax=None):
    """Plot initial velocity distribution based on simulation inputs."""
    ub = sim_params["ub"]
    vb = sim_params["vb"]
    n_particles = sim_params["n_particles"]
    v = np.random.normal(ub, vb / np.sqrt(2), n_particles)
    w = np.ones(n_particles)
    return plot_delta_f_velocity_distribution(v, w, ub, vb, bins=bins, ax=ax)

def plot_full_f_velocity_distribution(v, ub, vb, bins=20, ax=None):
    """Plot full distribution function f."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    plt.hist(v, bins=bins, density=True)
    v_grid = np.linspace(ub - 3 * vb, ub + 3 * vb, 100)
    ax.plot(v_grid, np.exp(-(v_grid - ub) ** 2 / (vb**2)) / np.sqrt(np.pi * vb**2))
    ax.axvline(ub, color="red")
    ax.axvline(1, color="black")
    ax.axvline(ub + vb / np.sqrt(2), color="green")
    ax.axvline(ub - vb / np.sqrt(2), color="green")
    ax.set_xlabel(r"$\frac{v k}{\omega}$")
    return fig, ax

def compute_gamma(sim_params):
    """Compute linear growth/damping rate gamma from simulation inputs."""
    nb_over_ne = sim_params["nb_over_ne"]
    ub = sim_params["ub"]
    vb = sim_params["vb"]
    return -np.sqrt(np.pi) * nb_over_ne * np.exp(-(1 - ub) ** 2 / (vb**2)) * (1 - ub) / vb**3

def fit_gamma(time, E_amp_hist, t_min, t_max, eps=1e-300):
    mask = (time >= t_min) & (time <= t_max)
    y = np.log(np.maximum(E_amp_hist[mask], eps))
    t = time[mask]
    coeffs = np.polyfit(t, y, 1)
    return coeffs[0]

def compute_gamma_krook(sim_params):
    """Compute linear growth/damping rate gamma from simulation inputs
    in the presence of a Krook collision operator. 
    The imaginary part reflects a finite-width resonant beam response.

        pi * delta(omega - k v_b) -> 1 / ((omega - k v_b)^2 + nu^2)

    The real Langmuir-wave frequency is based on the bulk plasma.
    In the limit nu -> 0, the Lorentzian becomes the Landau delta function,
    and this reduces to collisionless_eq341_growth_rate.
    """
    nb_over_ne = sim_params["nb_over_ne"]
    ub = sim_params["ub"]
    vb = sim_params["vb"]
    beta = sim_params["beta"]

    if beta == 0:
        return 0.5 * np.pi* (nb_over_ne / (np.sqrt(np.pi)*vb) * 2.0*(ub - 1) / vb**2 * np.exp(-(1 - ub) ** 2 / (vb**2)))
    else:
        return 0.5 * _lorentzian_beam_slope_integral(nb_over_ne, ub, vb, beta)
    
def _lorentzian_beam_slope_integral(nb, ub, vb, beta):
    """Return int F_b'(v) nu/[nu^2 + (omega-kv)^2] dv.

    This is the scalar Lorentzian-broadened beam response used by
    lorentzian_broadened_beam_response_growth_rate. 
    """

    n_res_quad = 512

    x, w = np.polynomial.legendre.leggauss(n_res_quad)
    theta = 0.5 * np.pi * x
    theta_weights = 0.5 * np.pi * w

    v_res_grid = 1.0 - beta * np.tan(theta)
    beam = _maxwellian(v_res_grid, nb, ub, vb)
    beam_slope = beam * (-2.0 * (v_res_grid - ub) / vb**2)

    return float((theta_weights * beam_slope).sum())

def _maxwellian(v, density, drift, thermal_speed):
    return density / (np.sqrt(np.pi) * thermal_speed) * np.exp(-np.power((v - drift) / thermal_speed, 2))