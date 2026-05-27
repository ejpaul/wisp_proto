"""
Goal 1 & 2: Nonlinear saturated amplitude and resolution scans.

Runs nonlinear delta-f simulations, detects saturation, and:
  (a) Compares measured E_sat with theory prediction: E_sat = (3.2)^2 * gamma^2
  (b) Scans gamma and E_sat vs N_particles
  (c) Scans gamma and E_sat vs dt

Produces 3 figures ready for the PI meeting.

Usage:
    python saturation_and_scans.py           # full run
    python saturation_and_scans.py --quick   # fast test
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
    """Berk-Breizman beam-only linear growth rate (wisp convention)."""
    return (
        -np.sqrt(np.pi)
        * nb_over_ne
        * np.exp(-((1 - ub) ** 2) / (vb ** 2))
        * (1 - ub)
        / vb ** 3
    )


def predict_E_sat(gamma):
    """Theory: E_sat = (3.2)^2 * gamma^2."""
    return (3.2 ** 2) * (gamma ** 2)


# =========================================================================
# Saturation detection
# =========================================================================

def detect_saturation(time, E_amp, slope_eps=1e-3, streak_time=40.0, window_time=60.0):
    """Find saturation time from |E|(t) using local-slope criterion.
    
    Key: only looks for saturation AFTER the field has reached at least
    10% of its peak value. This prevents detecting the slow-growth region
    as "flat". Also uses a tighter slope threshold (1e-3 instead of 5e-3)
    since gamma itself can be ~3e-3.
    """
    dt_eff = time[1] - time[0] if len(time) > 1 else 1.0
    Wn = max(5, int(round(window_time / dt_eff)))
    streak_len = max(5, int(round(streak_time / dt_eff)))

    logE = np.log(np.clip(E_amp, 1e-30, None))
    slopes = np.full_like(logE, np.nan)
    Acol = np.vstack([np.arange(Wn) * dt_eff, np.ones(Wn)]).T

    for i in range(Wn, len(logE)):
        y = logE[i - Wn : i]
        c = np.linalg.lstsq(Acol, y, rcond=None)[0]
        slopes[i] = c[0]

    # Only search after E reaches 10% of its peak value
    E_peak = np.max(E_amp)
    search_threshold = E_peak * 0.1
    
    growth_indices = np.where(E_amp >= search_threshold)[0]
    if len(growth_indices) == 0:
        peak_idx = int(np.argmax(E_amp))
        return peak_idx, time[peak_idx], E_amp[peak_idx]
    
    search_start = growth_indices[0]

    sat_idx = None
    for i in range(max(search_start, Wn), len(slopes) - streak_len):
        seg = slopes[i : i + streak_len]
        if np.all(np.isfinite(seg)) and np.all(np.abs(seg) < slope_eps):
            sat_idx = i
            break

    if sat_idx is None:
        # Fallback: use peak E as saturation point
        sat_idx = int(np.argmax(E_amp))

    # Measure E_sat as average over a window around saturation
    avg_start = sat_idx
    avg_end = min(len(E_amp), sat_idx + max(10, int(100.0 / dt_eff)))
    E_sat_val = np.mean(E_amp[avg_start:avg_end])

    return sat_idx, time[sat_idx], E_sat_val


# =========================================================================
# Growth rate fitting
# =========================================================================

def auto_fit_gamma(time, E_amp, r2_thresh=0.95, min_efolds=3.0):
    """Robustly extract growth rate from |E|(t).

    Strategy:
      1. Scan ALL possible (t0, t1) windows on a grid
      2. Require the fit spans at least `min_efolds` e-foldings of growth
         (i.e. log|E| changes by at least `min_efolds`)
      3. Among all windows with R² >= r2_thresh, pick the one with the
         LONGEST time span — this avoids overfitting to short noisy segments
      4. If no window meets the e-folding criterion, relax it and warn

    Returns
    -------
    gamma, r2, t0, t1  — or (nan, nan, None, None) if no fit found
    """
    mask = E_amp > 0
    tt = time[mask]
    yy = np.log(E_amp[mask])

    if len(tt) < 20:
        return np.nan, np.nan, None, None

    dt_eff = tt[1] - tt[0] if len(tt) > 1 else 1.0

    # Build candidate windows: step through start points, vary end points
    # Use coarser grid for speed (~100 start points × ~50 end points)
    n_pts = len(tt)
    start_step = max(1, n_pts // 100)
    end_step = max(1, n_pts // 50)

    best = None  # (gamma, r2, t0, t1, span)

    for i0 in range(0, n_pts - 20, start_step):
        for i1 in range(i0 + 20, n_pts, end_step):
            ts = tt[i0:i1]
            ys = yy[i0:i1]

            # Check e-folding range
            delta_logE = ys[-1] - ys[0]
            if delta_logE < min_efolds:
                continue  # not enough growth in this window

            # Must be monotonically-ish increasing (growth, not oscillation)
            # Quick check: slope should be positive
            if ys[-1] <= ys[0]:
                continue

            A = np.vstack([ts, np.ones_like(ts)]).T
            c = np.linalg.lstsq(A, ys, rcond=None)[0]
            gamma_try = c[0]

            if gamma_try <= 0:
                continue

            yhat = A @ c
            ssr = np.sum((ys - yhat) ** 2)
            sst = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1.0 - ssr / max(1e-12, sst)

            if r2 < r2_thresh:
                continue

            span = ts[-1] - ts[0]

            # Pick the LONGEST window that meets all criteria
            if best is None or span > best[4]:
                best = (gamma_try, r2, ts[0], ts[-1], span)

    # If nothing found with strict criteria, relax e-folding requirement
    if best is None:
        return auto_fit_gamma_relaxed(tt, yy, r2_thresh=0.85)

    return best[0], best[1], best[2], best[3]


def auto_fit_gamma_relaxed(tt, yy, r2_thresh=0.85):
    """Fallback: find best-R² window without e-folding requirement."""
    n_pts = len(tt)
    start_step = max(1, n_pts // 80)
    end_step = max(1, n_pts // 40)

    best = None

    for i0 in range(0, n_pts - 15, start_step):
        for i1 in range(i0 + 15, n_pts, end_step):
            ts = tt[i0:i1]
            ys = yy[i0:i1]

            A = np.vstack([ts, np.ones_like(ts)]).T
            c = np.linalg.lstsq(A, ys, rcond=None)[0]
            gamma_try = c[0]

            if gamma_try <= 0:
                continue

            yhat = A @ c
            ssr = np.sum((ys - yhat) ** 2)
            sst = np.sum((ys - np.mean(ys)) ** 2)
            r2 = 1.0 - ssr / max(1e-12, sst)

            if r2 < r2_thresh:
                continue

            span = ts[-1] - ts[0]
            if best is None or span > best[4]:
                best = (gamma_try, r2, ts[0], ts[-1], span)

    if best is None:
        print("  WARNING: no valid growth window found!")
        return np.nan, np.nan, None, None

    print(f"  NOTE: relaxed fit (e-folding criterion not met). "
          f"window=[{best[2]:.0f},{best[3]:.0f}], R²={best[1]:.3f}")
    return best[0], best[1], best[2], best[3]


# =========================================================================
# Nonlinear baseline parameters
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
# Figure 1: Single nonlinear run — saturation amplitude check
# =========================================================================

def figure1_saturation_check(params, save_path="fig1_saturation_check.png"):
    gamma_th = compute_gamma(params["nb_over_ne"], params["ub"], params["vb"])
    E_sat_th = predict_E_sat(gamma_th)

    print("=" * 60)
    print("  Figure 1: Saturation amplitude check")
    print("=" * 60)
    print(f"  gamma_theory  = {gamma_th:.4e}")
    print(f"  E_sat_theory  = (3.2)^2 * gamma^2 = {E_sat_th:.4e}")
    print(f"  N_particles   = {params['n_particles']:,}")
    print(f"  n_steps       = {params['n_steps']:,}  (t_max = {params['n_steps']*params['dt']:.0f})")
    print()

    out = run_timestepping(params, show_progress=True)
    time = out["time"]
    E_amp = out["E_amp_hist"]

    sat_idx, t_sat, E_sat_meas = detect_saturation(time, E_amp)
    sat_window = (time >= t_sat) & (time <= t_sat + 50)
    E_sat_avg = np.mean(E_amp[sat_window]) if np.count_nonzero(sat_window) > 10 else E_sat_meas

    result = auto_fit_gamma(time, E_amp)
    gamma_fit, r2_fit, t0_fit, t1_fit = result

    ratio = E_sat_avg / E_sat_th if E_sat_th > 0 else np.nan
    print(f"  gamma_fit     = {gamma_fit:.4e}")
    if t0_fit is not None:
        print(f"  fit window    = [{t0_fit:.0f}, {t1_fit:.0f}]  (R²={r2_fit:.4f})")
    print(f"  t_sat         = {t_sat:.1f}")
    print(f"  E_sat (meas)  = {E_sat_avg:.4e}")
    print(f"  E_sat (theory)= {E_sat_th:.4e}")
    print(f"  ratio meas/th = {ratio:.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(time, E_amp, lw=0.8, label="|E(t)|")
    E0 = E_amp[0]
    t_line = np.linspace(0, t_sat, 300)
    ax.semilogy(t_line, E0 * np.exp(gamma_th * t_line), "g--", lw=1.5,
                label=rf"theory $\gamma$={gamma_th:.3e}")
    ax.axhline(E_sat_th, color="red", ls="--", lw=1.5,
               label=rf"$E_{{sat}}$ theory = $(3.2)^2 \gamma^2$ = {E_sat_th:.2e}")
    ax.axhline(E_sat_avg, color="orange", ls=":", lw=1.5,
               label=rf"$E_{{sat}}$ measured = {E_sat_avg:.2e}")
    ax.axvline(t_sat, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel(r"$|\hat{E}|$", fontsize=12)
    ax.set_title(
        f"Nonlinear saturation check  |  "
        f"N={params['n_particles']:,}, dt={params['dt']}  |  "
        f"ratio = {ratio:.2f}", fontsize=13)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}\n")
    plt.close(fig)
    return {"gamma_fit": gamma_fit, "E_sat_meas": E_sat_avg, "E_sat_th": E_sat_th}


# =========================================================================
# Figure 2: Gamma and E_sat vs N_particles
# =========================================================================

def figure2_scan_N(params, N_list, save_path="fig2_scan_N.png"):
    gamma_th = compute_gamma(params["nb_over_ne"], params["ub"], params["vb"])
    E_sat_th = predict_E_sat(gamma_th)

    print("=" * 60)
    print("  Figure 2: Nonlinear convergence — gamma & E_sat vs N")
    print("=" * 60)
    print(f"  gamma_theory = {gamma_th:.4e},  E_sat_theory = {E_sat_th:.4e}")
    print(f"  N_list = {N_list}\n")

    rows = []
    for i, N in enumerate(N_list):
        print(f"  [{i+1}/{len(N_list)}] N = {N:,}")
        t0 = _time.perf_counter()
        p = dict(params, n_particles=N)
        out = run_timestepping(p)

        result = auto_fit_gamma(out["time"], out["E_amp_hist"])
        gamma_fit = result[0]
        _, t_sat, E_sat = detect_saturation(out["time"], out["E_amp_hist"])
        sat_win = (out["time"] >= t_sat) & (out["time"] <= t_sat + 50)
        E_sat_avg = np.mean(out["E_amp_hist"][sat_win]) if np.count_nonzero(sat_win) > 5 else E_sat

        rows.append({"N": N, "gamma": gamma_fit, "E_sat": E_sat_avg, "t_sat": t_sat})
        elapsed = _time.perf_counter() - t0
        print(f"  → gamma={gamma_fit:.4e}  E_sat={E_sat_avg:.4e}  t_sat={t_sat:.0f}  ({elapsed:.1f}s)\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    Ns = [r["N"] for r in rows]
    gammas = [r["gamma"] for r in rows]
    Esats = [r["E_sat"] for r in rows]

    ax1.semilogx(Ns, gammas, "o-", lw=1.5)
    ax1.axhline(gamma_th, color="k", ls="--", label=rf"theory $\gamma$={gamma_th:.3e}")
    ax1.set_xlabel("N_particles"); ax1.set_ylabel(r"$\gamma$ (fitted)")
    ax1.set_title(r"Growth rate vs $N$"); ax1.legend(); ax1.grid(True, which="both", ls=":")

    ax2.loglog(Ns, Esats, "o-", lw=1.5)
    ax2.axhline(E_sat_th, color="red", ls="--", label=rf"$(3.2)^2 \gamma^2$ = {E_sat_th:.2e}")
    ax2.set_xlabel("N_particles"); ax2.set_ylabel(r"$E_{{sat}}$")
    ax2.set_title(r"Saturated amplitude vs $N$"); ax2.legend(); ax2.grid(True, which="both", ls=":")

    fig.suptitle("Nonlinear convergence: N_particles scan", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}\n")
    plt.close(fig)
    return rows


# =========================================================================
# Figure 3: Gamma and E_sat vs dt
# =========================================================================

def figure3_scan_dt(params, dt_list, save_path="fig3_scan_dt.png"):
    gamma_th = compute_gamma(params["nb_over_ne"], params["ub"], params["vb"])
    E_sat_th = predict_E_sat(gamma_th)
    t_total = params["n_steps"] * params["dt"]

    print("=" * 60)
    print("  Figure 3: Nonlinear convergence — gamma & E_sat vs dt")
    print("=" * 60)
    print(f"  gamma_theory = {gamma_th:.4e},  E_sat_theory = {E_sat_th:.4e}")
    print(f"  t_total = {t_total:.0f},  dt_list = {dt_list}\n")

    rows = []
    for i, dt in enumerate(dt_list):
        n_steps = int(round(t_total / dt))
        print(f"  [{i+1}/{len(dt_list)}] dt = {dt}, n_steps = {n_steps:,}")
        t0 = _time.perf_counter()
        p = dict(params, dt=dt, n_steps=n_steps)
        out = run_timestepping(p)

        result = auto_fit_gamma(out["time"], out["E_amp_hist"])
        gamma_fit = result[0]
        _, t_sat, E_sat = detect_saturation(out["time"], out["E_amp_hist"])
        sat_win = (out["time"] >= t_sat) & (out["time"] <= t_sat + 50)
        E_sat_avg = np.mean(out["E_amp_hist"][sat_win]) if np.count_nonzero(sat_win) > 5 else E_sat

        rows.append({"dt": dt, "gamma": gamma_fit, "E_sat": E_sat_avg, "t_sat": t_sat})
        elapsed = _time.perf_counter() - t0
        print(f"  → gamma={gamma_fit:.4e}  E_sat={E_sat_avg:.4e}  t_sat={t_sat:.0f}  ({elapsed:.1f}s)\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    dts = [r["dt"] for r in rows]
    gammas = [r["gamma"] for r in rows]
    Esats = [r["E_sat"] for r in rows]

    ax1.semilogx(dts, gammas, "o-", lw=1.5)
    ax1.axhline(gamma_th, color="k", ls="--", label=rf"theory $\gamma$={gamma_th:.3e}")
    ax1.set_xlabel(r"$\Delta t$"); ax1.set_ylabel(r"$\gamma$ (fitted)")
    ax1.set_title(r"Growth rate vs $\Delta t$"); ax1.legend(); ax1.grid(True, which="both", ls=":")

    ax2.semilogx(dts, Esats, "o-", lw=1.5)
    ax2.axhline(E_sat_th, color="red", ls="--", label=rf"$(3.2)^2 \gamma^2$ = {E_sat_th:.2e}")
    ax2.set_xlabel(r"$\Delta t$"); ax2.set_ylabel(r"$E_{{sat}}$")
    ax2.set_title(r"Saturated amplitude vs $\Delta t$"); ax2.legend(); ax2.grid(True, which="both", ls=":")

    fig.suptitle(r"Nonlinear convergence: $\Delta t$ scan", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {save_path}\n")
    plt.close(fig)
    return rows


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Fewer particles and coarser scans for quick testing")
    args = parser.parse_args()

    params = dict(NONLINEAR_BASE)

    if args.quick:
        params["n_particles"] = 100_000
        params["n_steps"] = 70_000    # t=7000, just enough for saturation
        N_list = [50_000, 100_000, 200_000]
        dt_list = [0.20, 0.10, 0.05]
    else:
        N_list = [50_000, 100_000, 200_000, 500_000, 1_000_000]
        dt_list = [0.40, 0.20, 0.10, 0.05, 0.02]

    t_start = _time.perf_counter()
    figure1_saturation_check(params)
    t_fig1 = _time.perf_counter() - t_start

    # Estimate remaining time based on figure1
    n_scan_runs = len(N_list) + len(dt_list)
    est_remaining = t_fig1 * n_scan_runs
    print(f"  Figure 1 took {t_fig1:.0f}s. Scans have {n_scan_runs} runs.")
    print(f"  Estimated remaining time: ~{est_remaining/60:.0f} minutes\n")

    figure2_scan_N(params, N_list)
    figure3_scan_dt(params, dt_list)

    total = _time.perf_counter() - t_start
    print("\n" + "=" * 60)
    print(f"  All figures saved. Total time: {total/60:.1f} minutes.")
    print("=" * 60)


if __name__ == "__main__":
    main()