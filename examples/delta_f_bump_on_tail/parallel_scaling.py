"""Parallel scaling study for WISP.

Measures wall-clock time as a function of NUMBA_NUM_THREADS to find
the thread count where performance converges (i.e. stops improving).

IMPORTANT: NUMBA_NUM_THREADS must be set BEFORE importing numba.
This script spawns a fresh subprocess for each thread count to ensure
the environment variable takes effect.

Usage
-----
    python parallel_scaling.py                 # auto-detect max threads
    python parallel_scaling.py --max-threads 8 # cap at 8
    python parallel_scaling.py --quick         # fewer repeats, smaller problem
"""

import subprocess
import sys
import os
import json
import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# The benchmark script that runs inside each subprocess.
# It times a single simulation after a JIT warmup pass.
# ---------------------------------------------------------------------------

BENCH_TEMPLATE = r'''
import os
os.environ["NUMBA_NUM_THREADS"] = "{n_threads}"

import time
import json
import numpy as np

# Import AFTER setting the env var
from core import run_timestepping

# --- Simulation parameters ---
params = dict(
    E_c      = 1e-12,
    E_s      = 0.0,
    nb_over_ne = 5e-7,
    n_steps  = {n_steps},
    dt       = 0.01,
    n_particles = {n_particles},
    vb       = 0.5,
    ub       = 1.05,
    splitting_method = "strang",
    method   = "linear",
    full_f   = False,
    seed     = 42,
)

# --- JIT warmup (small problem, compiles all code paths) ---
warmup = dict(params, n_steps=10, n_particles=200, seed=0)
run_timestepping(warmup)

# --- Timed run ---
t0 = time.perf_counter()
run_timestepping(params)
t1 = time.perf_counter()

print(json.dumps({{"threads": {n_threads}, "wall_s": t1 - t0}}))
'''


def run_benchmark(n_threads, n_particles, n_steps, cwd):
    """Spawn a subprocess with the given thread count and return wall time."""
    script = BENCH_TEMPLATE.format(
        n_threads=n_threads,
        n_particles=n_particles,
        n_steps=n_steps,
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        timeout=600,
        cwd=cwd,
    )
    if proc.returncode != 0:
        print(f"  [threads={n_threads}] FAILED:\n{proc.stderr[:300]}")
        return None

    # Parse JSON from last line of stdout
    for line in reversed(proc.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    print(f"  [threads={n_threads}] No JSON output")
    return None


def main():
    parser = argparse.ArgumentParser(description="WISP parallel scaling study")
    parser.add_argument("--max-threads", type=int, default=None,
                        help="Maximum thread count to test (default: all cores)")
    parser.add_argument("--n-particles", type=int, default=200_000,
                        help="Number of particles (default: 200000)")
    parser.add_argument("--n-steps", type=int, default=2000,
                        help="Number of time steps (default: 2000)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Timing repeats per thread count (default: 3)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 50k particles, 500 steps, 1 repeat")
    parser.add_argument("--output", type=str, default="parallel_scaling.png",
                        help="Output plot filename")
    args = parser.parse_args()

    if args.quick:
        args.n_particles = 50_000
        args.n_steps = 500
        args.repeats = 1

    max_threads = args.max_threads or os.cpu_count() or 4
    # Build list: 1, 2, 4, 8, ... up to max_threads, always including 1 and max
    thread_counts = sorted(set(
        [1] +
        [2**p for p in range(1, 10) if 2**p <= max_threads] +
        [max_threads]
    ))

    cwd = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  WISP Parallel Scaling Study")
    print("=" * 60)
    print(f"  Particles : {args.n_particles:,}")
    print(f"  Steps     : {args.n_steps:,}")
    print(f"  Repeats   : {args.repeats}")
    print(f"  Threads   : {thread_counts}")
    print(f"  Working dir: {cwd}")
    print()

    results = []

    for nt in thread_counts:
        times = []
        for rep in range(args.repeats):
            data = run_benchmark(nt, args.n_particles, args.n_steps, cwd)
            if data is not None:
                times.append(data["wall_s"])
                print(f"  threads={nt:3d}  rep={rep}  time={data['wall_s']:.3f}s")

        if times:
            results.append({
                "threads": nt,
                "time_mean": np.mean(times),
                "time_std": np.std(times),
                "time_min": np.min(times),
                "times": times,
            })
        else:
            print(f"  threads={nt}: all runs failed")

    if len(results) < 2:
        print("\nNot enough data points. Check that Numba is installed.")
        return

    # --- Print summary table ---
    print("\n" + "=" * 60)
    print(f"  {'Threads':>8s}  {'Mean (s)':>10s}  {'Std (s)':>10s}  {'Speedup':>8s}")
    print("-" * 60)
    t1_time = results[0]["time_mean"]
    for r in results:
        speedup = t1_time / r["time_mean"]
        print(f"  {r['threads']:>8d}  {r['time_mean']:>10.3f}  {r['time_std']:>10.3f}  {speedup:>8.2f}x")
    print("=" * 60)

    # --- Save data as JSON ---
    json_path = args.output.replace(".png", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nData saved to {json_path}")

    # --- Plot ---
    threads = np.array([r["threads"] for r in results])
    means   = np.array([r["time_mean"] for r in results])
    stds    = np.array([r["time_std"] for r in results])
    speedup = t1_time / means

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: wall time vs threads
    ax1.errorbar(threads, means, yerr=stds, fmt="o-", capsize=4, lw=1.5)
    ax1.set_xlabel("NUMBA_NUM_THREADS")
    ax1.set_ylabel("Wall time (s)")
    ax1.set_title(f"Execution time\n({args.n_particles:,} particles, {args.n_steps} steps)")
    ax1.set_xticks(threads)
    ax1.grid(True, ls=":", alpha=0.6)

    # Right: speedup vs threads
    ax2.plot(threads, speedup, "o-", lw=1.5, label="Measured")
    ax2.plot(threads, threads.astype(float), "k--", alpha=0.4, label="Ideal (linear)")
    ax2.set_xlabel("NUMBA_NUM_THREADS")
    ax2.set_ylabel("Speedup  (T₁ / Tₙ)")
    ax2.set_title("Parallel speedup")
    ax2.set_xticks(threads)
    ax2.legend()
    ax2.grid(True, ls=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()