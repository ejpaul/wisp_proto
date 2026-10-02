"""Matched baseline/prescribed/self-consistent experiments from one checkpoint."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

import core_collision_full as cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--secondary-omega", type=float, action="append", default=[])
    args = parser.parse_args()
    if args.warmup_steps < 0:
        parser.error("warmup-steps must be nonnegative")
    with args.params.open() as stream:
        params = json.load(stream)
    if len(params.get("modes", [])) != 2:
        parser.error("The matched scan requires exactly two modes in params")
    if params.get("restart_file"):
        parser.error("Use the solver directly to branch from an external restart; scan creates its own parent")
    params.pop("checkpoint_file", None)
    rank, comm = 0, None
    if args.backend == "gpu":
        import core_collision_full_gpu as gpu
        comm = gpu._get_comm()
        rank = comm.Get_rank()
        run = lambda p: gpu.run_timestepping(p, comm)
        save = lambda path, out: gpu.save_ranked_outputs(path, out, comm)
    else:
        run, save = cpu.run_cpu_timestepping, cpu._atomic_savez
    error = None
    if rank == 0:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            error = str(exc)
    if comm is not None:
        error = comm.bcast(error, root=0)
    if error:
        raise ValueError(f"Choose a new output directory: {error}")

    parent = args.output_dir / "common_checkpoint.npz"
    warmup = dict(params, modes=params["modes"][:1], n_steps=args.warmup_steps,
                  checkpoint_file=str(args.output_dir / "warmup_checkpoint.npz"))
    initial = run(warmup)
    save(parent, initial)
    if initial["stopped_early"]:
        if rank == 0:
            print(f"Warmup interrupted; checkpoint saved to {parent}")
        return
    del initial
    cases = []
    for name, coupling, evolve in (
        ("baseline", 0., False),
        ("prescribed", params["modes"][1].get("coupling", 1.), False),
        ("self_consistent", params["modes"][1].get("coupling", 1.), True),
    ):
        modes = deepcopy(params["modes"])
        modes[1].update(coupling=coupling, evolve=evolve)
        cases.append((name, modes))
    for index, omega in enumerate(args.secondary_omega):
        modes = deepcopy(params["modes"])
        modes[1].update(omega=omega, evolve=False)
        cases.append((f"detuned_{index:02d}", modes))

    traces, metrics = {}, {}
    for name, modes in cases:
        p = dict(params, modes=modes, restart_file=str(parent), restart_mode_policy="branch",
                 checkpoint_file=str(args.output_dir / f"{name}_checkpoint.npz"))
        if rank == 0:
            (args.output_dir / f"{name}.json").write_text(json.dumps(p, indent=2) + "\n")
        result = run(p)
        save(args.output_dir / f"{name}.npz", result)
        if rank == 0:
            traces[name] = {key: result[key] for key in
                            ("time", "E_amp_hist", "v_edges", "delta_f", "overlap_hist")}
            metrics[name] = dict(final_primary_amplitude=float(np.hypot(result["E_c"][0], result["E_s"][0])),
                                 final_secondary_amplitude=float(np.hypot(result["E_c"][1], result["E_s"][1])),
                                 particle_energy=float(result["particle_energy"]),
                                 completed_steps=int(result["completed_steps"]))
        stopped = bool(result["stopped_early"])
        del result
        if stopped:
            break
    if rank == 0:
        from matplotlib import pyplot as plt
        (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for name, result in traces.items():
            axes[0].plot(result["time"], result["E_amp_hist"][:, 0], label=name)
            edges = result["v_edges"]
            axes[1].plot((edges[:-1]+edges[1:])/2, result["delta_f"], label=name)
            axes[2].plot(result["time"], result["overlap_hist"][:, 0], label=name)
        axes[0].set(xlabel="t", ylabel="Primary |E|")
        axes[1].set(xlabel="v", ylabel="Final delta f (weighted density)")
        axes[2].set(xlabel="t", ylabel="Isolated-width overlap estimate")
        axes[0].legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(args.output_dir / "comparison.png", dpi=160)
        plt.close(fig)
        print(f"Saved matched experiments to {args.output_dir}")


if __name__ == "__main__":
    main()
