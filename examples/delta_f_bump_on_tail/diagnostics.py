"""Diagnostics for WISP simulations.

Provides analytic growth rate, velocity distribution plotting,
and growth-rate fitting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def compute_gamma(sim_params):
    """Analytic linear growth rate (Berk & Breizman beam-only formula).

    Uses the wisp convention where f_b ∝ exp(-(v-ub)²/vb²),
    i.e. the thermal width parameter vb corresponds to σ = vb/√2.
    """
    nb_over_ne = sim_params["nb_over_ne"]
    ub = sim_params["ub"]
    vb = sim_params["vb"]
    return (
        -np.sqrt(np.pi)
        * nb_over_ne
        * np.exp(-((1 - ub) ** 2) / (vb ** 2))
        * (1 - ub)
        / vb ** 3
    )


def plot_initial_velocity_distribution(sim_params, bins=20, ax=None):
    """Plot the initial beam velocity distribution."""
    ub = sim_params["ub"]
    vb = sim_params["vb"]
    n_particles = sim_params["n_particles"]
    v = np.random.default_rng(0).normal(ub, vb / np.sqrt(2), n_particles)
    w = np.ones(n_particles)
    return plot_delta_f_velocity_distribution(v, w, ub, vb, bins=bins, ax=ax)


def plot_delta_f_velocity_distribution(v, w, ub, vb, bins=20, ax=None):
    """Plot weighted velocity histogram with Maxwellian overlay."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    v_grid = np.linspace(ub - 3 * vb, ub + 3 * vb, 100)
    ax.hist(v, weights=w, density=True, bins=bins, alpha=0.7,
            label=r"$\delta f$ histogram")
    ax.plot(
        v_grid,
        np.exp(-((v_grid - ub) ** 2) / (vb ** 2)) / np.sqrt(np.pi * vb ** 2),
        "k--", lw=1.5, label=r"$f_0$ (Maxwellian)",
    )
    ax.axvline(ub, color="red", ls=":", label=rf"$u_b = {ub:.2f}$")
    ax.axvline(1, color="black", ls=":", label=r"$v_\phi = \omega/k$")
    ax.set_xlabel(r"$v k / \omega$")
    ax.legend(fontsize=8)
    return fig, ax