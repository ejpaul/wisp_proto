"""
Goal 3: Distribution function diagnostics.

Runs nonlinear delta-f and nonlinear full-f simulations with the same physics,
then plots:
  (a) |E|(t) for both methods
  (b) Reconstructed f₀ + δf from delta-f weights
  (c) Full-f velocity histogram
  (d) Overlay comparison of both

Produces 1 four-panel figure ready for the PI meeting.

Usage:
    python distribution_diagnostics.py           # full run
    python distribution_diagnostics.py --quick   # fast test
"""

# === SET THREAD COUNT BEFORE ANY OTHER IMPORTS ===
import os
os.environ["NUMBA_NUM_THREADS"] = "8"

import argparse
import time as _time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core import run_timestepping


# =========================================================================
# Analytic predictions
# =========================================================================

def compute_gamma(nb_over_ne, ub, vb):
    """Berk-Breizman beam-only linear growth rate."""
    return (
        -np.sqrt(np.pi)
        * nb_over_ne
        * np.exp(-((1 - ub) ** 2) / (vb ** 2))
        * (1 - ub)
        / vb ** 3
    )


def f0_maxwellian(v, ub, vb):
    """Equilibrium beam Maxwellian f₀(v) (wisp convention: σ = vb/√2)."""
    return np.exp(-((v - ub) ** 2) / (vb ** 2)) / np.sqrt(np.pi * vb ** 2)


# =========================================================================
# Reconstruction: f₀ + δf from delta-f weights
# =========================================================================

def reconstruct_f0_plus_delta_f(v, w, ub, vb, bins=80):
    """Reconstruct the full distribution f₀ + δf from delta-f particle data.

    In the delta-f method, the weight w_i represents δf/f₀ evaluated at
    the particle's position in velocity space. So the full distribution is:

        f(v) = f₀(v) * (1 + w)

    We bin particles by velocity and compute the mean (1 + w) in each bin,
    then multiply by f₀ at the bin center.

    Returns
    -------
    bin_centers, f_reconstructed, f0_at_bins, delta_f
    """
    v_lo = ub - 4 * vb
    v_hi = ub + 4 * vb
    bin_edges = np.linspace(v_lo, v_hi, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Bin particles and compute mean weight per bin
    count, _ = np.histogram(v, bins=bin_edges)
    wsum, _ = np.histogram(v, bins=bin_edges, weights=w)

    mean_w = np.zeros_like(bin_centers)
    valid = count > 0
    mean_w[valid] = wsum[valid] / count[valid]

    f0_at_bins = f0_maxwellian(bin_centers, ub, vb)

    # f = f₀ * (1 + <w>)
    f_reconstructed = f0_at_bins * (1.0 + mean_w)
    delta_f = f0_at_bins * mean_w

    return bin_centers, f_reconstructed, f0_at_bins, delta_f


# =========================================================================
# Full-f histogram
# =========================================================================

def histogram_full_f(v, ub, vb, bins=80):
    """Normalized velocity histogram for full-f particles."""
    v_lo = ub - 4 * vb
    v_hi = ub + 4 * vb
    bin_edges = np.linspace(v_lo, v_hi, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    counts, _ = np.histogram(v, bins=bin_edges, density=True)
    return bin_centers, counts


# =========================================================================
# Parameters
# =========================================================================

NONLINEAR_BASE = dict(
    E_c=1e-12,
    E_s=-1e-12,
    nb_over_ne=5e-5,
    n_steps=80_000,       # t_total = 8000, enough for saturation at ~t=5650
    dt=0.1,
    n_particles=500_000,
    vb=0.1,
    ub=1.1,
    splitting_method="strang",
    method="nonlinear",
    full_f=False,
    seed=42,
)


# =========================================================================
# Main figure
# =========================================================================

def make_figure(params_df, params_ff, save_path="fig4_distribution_diagnostics.png"):
    """Run both methods and produce a 4-panel comparison figure."""

    gamma_th = compute_gamma(params_df["nb_over_ne"], params_df["ub"], params_df["vb"])
    ub = params_df["ub"]
    vb = params_df["vb"]

    print("=" * 60)
    print("  Figure 4: Distribution diagnostics (delta-f vs full-f)")
    print("=" * 60)
    print(f"  gamma_theory = {gamma_th:.4e}")
    print(f"  delta-f:  N = {params_df['n_particles']:,},  n_steps = {params_df['n_steps']}")
    print(f"  full-f:   N = {params_ff['n_particles']:,},  n_steps = {params_ff['n_steps']}")
    print()

    print("  Running nonlinear delta-f ...")
    t0 = _time.perf_counter()
    out_df = run_timestepping(params_df, show_progress=True)
    print(f"  delta-f done in {_time.perf_counter() - t0:.1f}s")

    print("  Running nonlinear full-f ...")
    t0 = _time.perf_counter()
    out_ff = run_timestepping(params_ff, show_progress=True)
    print(f"  full-f done in {_time.perf_counter() - t0:.1f}s")

    # --- Reconstruct distributions ---
    vc_df, f_total, f0_bins, delta_f = reconstruct_f0_plus_delta_f(
        out_df["v"], out_df["w"], ub, vb
    )
    vc_ff, hist_ff = histogram_full_f(out_ff["v"], ub, vb)
    v_grid = np.linspace(ub - 4 * vb, ub + 4 * vb, 300)
    f0_smooth = f0_maxwellian(v_grid, ub, vb)

    # --- 4-panel figure ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (a) Top-left: |E|(t) comparison
    ax = axes[0, 0]
    ax.semilogy(out_df["time"], out_df["E_amp_hist"], lw=0.8, label="delta-f")
    ax.semilogy(out_ff["time"], out_ff["E_amp_hist"], lw=0.8, alpha=0.7, label="full-f")
    # Theory line
    E0 = out_df["E_amp_hist"][0]
    t_ref = out_df["time"]
    ax.semilogy(t_ref, E0 * np.exp(gamma_th * t_ref), "k--", lw=1, alpha=0.5,
                label=rf"theory $\gamma$={gamma_th:.2e}")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|\hat{E}|$")
    ax.set_title("|E|(t) — both methods")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls=":", alpha=0.5)

    # (b) Top-right: f₀ + δf reconstruction (delta-f)
    ax = axes[0, 1]
    ax.plot(v_grid, f0_smooth, "k--", lw=1.5, label=r"$f_0$ (initial)")
    ax.plot(vc_df, f_total, "b-", lw=1.5, label=r"$f_0 + \delta f$ (delta-f)")
    ax.fill_between(vc_df, f0_bins, f_total, alpha=0.3, color="blue")
    ax.axvline(1.0, color="gray", ls=":", alpha=0.6, label=r"$v_\phi$")
    ax.set_xlabel(r"$v k / \omega$")
    ax.set_ylabel("f(v)")
    ax.set_title(r"Reconstructed $f_0 + \delta f$ (nonlinear $\delta f$)")
    ax.legend(fontsize=8)

    # (c) Bottom-left: full-f histogram
    ax = axes[1, 0]
    ax.plot(v_grid, f0_smooth, "k--", lw=1.5, label=r"$f_0$ (initial)")
    ax.plot(vc_ff, hist_ff, "g-", lw=1.5, label="full-f histogram (final)")
    ax.axvline(1.0, color="gray", ls=":", alpha=0.6, label=r"$v_\phi$")
    ax.set_xlabel(r"$v k / \omega$")
    ax.set_ylabel("f(v)")
    ax.set_title("Full-f velocity distribution (final)")
    ax.legend(fontsize=8)

    # (d) Bottom-right: overlay comparison
    ax = axes[1, 1]
    ax.plot(v_grid, f0_smooth, "k--", lw=1.5, label=r"$f_0$ (initial)")
    ax.plot(vc_df, f_total, "b-", lw=1.5, label=r"delta-f: $f_0 + \delta f$")
    ax.plot(vc_ff, hist_ff, "g-", lw=1.5, alpha=0.7, label="full-f: histogram")
    ax.axvline(1.0, color="gray", ls=":", alpha=0.6, label=r"$v_\phi$")
    ax.set_xlabel(r"$v k / \omega$")
    ax.set_ylabel("f(v)")
    ax.set_title(r"Comparison: $\delta f$ reconstruction vs full-f")
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Distribution diagnostics  |  delta-f N={params_df['n_particles']:,}, "
        f"full-f N={params_ff['n_particles']:,}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Saved → {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    params_df = dict(NONLINEAR_BASE, full_f=False)
    params_ff = dict(NONLINEAR_BASE, full_f=True)

    if args.quick:
        params_df["n_particles"] = 100_000
        params_df["n_steps"] = 70_000     # t=7000, need to reach saturation
        params_ff["n_particles"] = 500_000
        params_ff["n_steps"] = 70_000
    else:
        # Full-f needs more particles for comparable noise
        params_ff["n_particles"] = 5_000_000

    make_figure(params_df, params_ff)

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()