"""Shared CPU/CUDA orchestration for the traveling-wave extension.

The scalar entry points dispatch here only when params['modes'] is present.
Collision coefficients and stochastic streams follow core_collision_full.py.
"""
import json
import os
import signal
from time import monotonic

import numpy as np
from wisp.diagnostics import compute_gamma

import core_collision_full as base
from secondary_modes import (
    MODE_SCHEME, advance_fields, carrier_integral, mode_arrays, mode_json,
    parse_modes, resonance_diagnostics, velocity_distribution,
)


def _configuration(params):
    p = dict(params)
    for key in ("full_f", "show_progress"):
        if key in p and not isinstance(p[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a boolean")
    for key, minimum in (("n_steps", 0), ("n_particles", 1)):
        if isinstance(p[key], bool) or int(p[key]) != p[key] or p[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    if p["splitting_method"] not in ("strang", "trotter"):
        raise ValueError("splitting_method must be strang or trotter")
    gamma_l = float(compute_gamma(p))
    ratio = float(p.get("gamma_ratio", .9))
    default_damping = float(p.get("gamma_d", ratio * gamma_l))
    for name in ("nu", "alpha", "beta"):
        p[name] = float(p.get(name, (1. - ratio) * p.get(name + "_norm", 0.) * gamma_l))
    base._validate_full_mode(p["method"], p["full_f"], p["nu"], p["beta"],
                             p["dt"], p["vb"], p["alpha"])
    if not np.isfinite(p["ub"]) or not np.isfinite(p["nb_over_ne"]) or p["nb_over_ne"] < 0:
        raise ValueError("ub must be finite and nb_over_ne must be finite and nonnegative")
    specs = parse_modes(p, default_damping)
    signature = json.dumps({key: p[key] for key in (
        "dt", "ub", "vb", "nb_over_ne", "nu", "alpha", "beta", "splitting_method"
    )}, sort_keys=True)
    return p, specs, signature


def _load_state(path, comm, gpu):
    """Read metadata on rank zero; broadcast errors before any collectives."""
    rank = comm.Get_rank() if gpu else 0
    metadata = arrays = error = None
    if rank == 0:
        try:
            with np.load(path, allow_pickle=True) as state:
                required = ("E_c", "E_s", "t_final", "time_origin", "rng_seed", "rng_step",
                            "rng_state", "collision_weight_scheme", "mode_scheme", "modes_json",
                            "physics_signature", "n_ranks", "generation", "global_n_particles",
                            "shard_pattern")
                metadata = {key: state[key].copy() for key in required if key in state}
                arrays = {key: state[key].copy() for key in ("x", "v", "w", "p")
                          if key in state}
                metadata["has_particles"] = len(arrays) == 4
                metadata["particle_count"] = (len(arrays["x"]) if len(arrays) == 4
                                              else int(state["global_n_particles"]))
        except Exception as exc:
            error = str(exc)
    if gpu:
        error, metadata = comm.bcast((error, metadata), root=0)
    if error:
        raise ValueError(f"Cannot read checkpoint: {error}")
    base._require_full_checkpoint(metadata.get("collision_weight_scheme", ""))
    if not gpu:
        if not metadata["has_particles"]:
            raise ValueError("CPU restart requires a full CPU checkpoint, not a GPU manifest")
        return metadata, arrays, 0
    from core_collision_full_gpu import _partition, _scatter_array
    if metadata["has_particles"]:
        counts, offsets = _partition(metadata["particle_count"], comm.Get_size())
        local = {key: _scatter_array(arrays[key] if rank == 0 else None,
                                    counts, offsets, comm) for key in ("x", "v", "w", "p")}
        return metadata, local, int(offsets[rank])
    local = None
    try:
        if int(metadata["n_ranks"]) != comm.Get_size():
            raise ValueError("GPU restart requires the same number of ranks as the manifest")
        pattern = str(np.asarray(metadata["shard_pattern"]).item())
        shard_path = os.path.join(os.path.dirname(os.path.abspath(path)), pattern.format(rank=rank))
        with np.load(shard_path, allow_pickle=True) as shard:
            for key in ("generation", "rng_step", "rng_seed", "n_ranks", "global_n_particles"):
                if int(shard[key]) != int(metadata[key]):
                    raise ValueError(f"Shard/manifest mismatch: {key}")
            if int(shard["rank"]) != rank:
                raise ValueError("Shard rank mismatch")
            base._require_full_checkpoint(shard["collision_weight_scheme"])
            local = {key: shard[key].copy() for key in ("x", "v", "w", "p")}
            offset = int(shard["particle_offset"])
            counts, offsets = _partition(metadata["particle_count"], comm.Get_size())
            if offset != offsets[rank] or len(local["x"]) != counts[rank]:
                raise ValueError("Shard partition mismatch")
    except Exception as exc:
        error = str(exc)
    errors = comm.allgather(error) if comm.Get_size() > 1 else [error]
    if any(errors):
        raise ValueError(f"Cannot load checkpoint shards: {errors}")
    return metadata, local, offset


def _restart_fields(metadata, specs, signature, policy):
    ec = np.atleast_1d(metadata["E_c"]).astype(float)
    es = np.atleast_1d(metadata["E_s"]).astype(float)
    old_modes = metadata.get("modes_json")
    old_signature = metadata.get("physics_signature")
    if policy not in ("strict", "branch"):
        raise ValueError("restart_mode_policy must be strict or branch")
    if old_signature is not None and str(old_signature) != signature:
        raise ValueError("Restart physics mismatch (dt, collisions, distribution, or splitting)")
    if old_modes is None:
        if policy != "branch":
            raise ValueError("Scalar checkpoints require explicit restart_mode_policy='branch'")
        old = [dict(k=1., omega=1., coupling=1.)]
    else:
        if str(metadata.get("mode_scheme", "")) != MODE_SCHEME:
            raise ValueError("Incompatible mode scheme")
        old = json.loads(str(old_modes))
    new = json.loads(mode_json(specs))
    if ec.shape != (len(old),) or es.shape != ec.shape:
        raise ValueError("Checkpoint field shape does not match modes")
    if not np.all(np.isfinite(ec)) or not np.all(np.isfinite(es)):
        raise ValueError("Checkpoint fields must be finite")
    if policy == "strict":
        if old_signature is None:
            raise ValueError("Checkpoint lacks restart physics metadata")
        # Seeds are irrelevant on continuation; dynamic field values come from the checkpoint.
        strip = lambda modes: [{k: v for k, v in m.items() if k not in ("E_c", "E_s")}
                               for m in modes]
        if strip(old) != strip(new):
            raise ValueError("Restart mode mismatch; use branch explicitly to change modes")
        return ec, es
    if any(old[0][key] != new[0][key] for key in ("k", "omega", "coupling")):
        raise ValueError("Branch must preserve the primary k, omega, and coupling")
    new_ec, new_es = mode_arrays(specs)[-2:]
    new_ec[0], new_es[0] = ec[0], es[0]
    return new_ec, new_es


def run_multimode(params, backend="cpu", comm=None):
    p, specs, signature = _configuration(params)
    gpu = backend == "gpu"
    rank, size, offset = 0, 1, 0
    if gpu:
        import core_collision_full_gpu as gb
        from numba import config, cuda
        import secondary_gpu as kernels
        comm = gb._get_comm(comm)
        rank, size = comm.Get_rank(), comm.Get_size()
        if p.get("expected_gpus") is not None and int(p["expected_gpus"]) != size:
            raise ValueError(f"Expected {p['expected_gpus']} MPI ranks/GPUs, received {size}")
        if not config.ENABLE_CUDASIM:
            visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d]
            cuda.select_device(0 if len(visible) == 1 else int(os.environ.get("SLURM_LOCALID", rank)))
    else:
        import secondary_cpu as kernels
    dt, n_steps, count = float(p["dt"]), int(p["n_steps"]), int(p["n_particles"])
    k, omega, coupling, damping, ec, es = mode_arrays(specs)
    rng = np.random.default_rng(p.get("seed") if rank == 0 else 0)
    rng_step = 0
    t_start = float(p.get("t_start", 0.))
    origin = t_start
    if not np.isfinite(t_start):
        raise ValueError("t_start must be finite")
    if p.get("restart_file"):
        metadata, state, offset = _load_state(p["restart_file"], comm, gpu)
        ec, es = _restart_fields(metadata, specs, signature, p.get("restart_mode_policy", "strict"))
        t_start = float(metadata["t_final"])
        if "t_start" in p and p["t_start"] != t_start:
            raise ValueError("Restart t_start must equal checkpoint t_final")
        count = int(metadata["particle_count"])
        if count != p["n_particles"]:
            raise ValueError("Restart n_particles differs from checkpoint")
        rng_seed, rng_step = np.uint64(metadata["rng_seed"]), int(metadata["rng_step"])
        origin = float(metadata.get("time_origin", t_start - rng_step * dt))
        if rank == 0 and "rng_state" in metadata:
            rng.bit_generator.state = base._unpack_rng_state(metadata["rng_state"])
    else:
        if rank == 0:
            state = dict(x=rng.uniform(0., 2 * np.pi, count),
                         v=rng.normal(p["ub"], p["vb"] / np.sqrt(2), count),
                         w=np.zeros(count), p=np.ones(count))
            rng_seed = base._seed_to_uint64(p.get("seed"), rng)
        if gpu:
            rng_seed = np.uint64(comm.bcast(int(rng_seed) if rank == 0 else None, root=0))
            counts, offsets = gb._partition(count, size)
            state = {key: gb._scatter_array(state[key] if rank == 0 else None,
                                            counts, offsets, comm) for key in ("x", "v", "w", "p")}
            offset = int(offsets[rank])
    state_error = None
    for key in ("x", "v", "w", "p"):
        state[key] = np.ascontiguousarray(state[key], dtype=float)
        if (state[key].ndim != 1 or state[key].shape != state["x"].shape
                or not np.all(np.isfinite(state[key]))):
            state_error = "Particle arrays must be finite matching vectors"
    if gpu and size > 1:
        errors = comm.allgather(state_error)
        state_error = next((error for error in errors if error), None)
    if state_error:
        raise ValueError(state_error)
    h = dt / 2 if p["splitting_method"] == "strang" else dt
    integral = carrier_integral(omega, h)
    krook, drag = np.exp(-p["beta"] * h), p["alpha"]**2
    sigma = np.sqrt(2 * p["nu"]**3 * h)
    collisions = bool(p["beta"] or drag or sigma)
    if gpu:
        state = {key: cuda.to_device(value) for key, value in state.items()}
        dk, do, dg, di = [cuda.to_device(a) for a in (k, omega, coupling, integral)]
        dec, des = cuda.to_device(ec), cuda.to_device(es)
        sums_device = cuda.device_array((len(k), 2), dtype=float)
        sums_zero = np.zeros((len(k), 2))
        blocks = max(1, (len(state["x"]) + gb.THREADS_PER_BLOCK - 1) // gb.THREADS_PER_BLOCK)

    def push(mid, step, after):
        args = (mid, h, p["ub"], 1. / p["vb"]**2, krook, drag, sigma,
                collisions, after, rng_seed, step, offset)
        particles = tuple(state[key] for key in ("x", "v", "w", "p"))
        if gpu:
            dec.copy_to_device(ec)
            des.copy_to_device(es)
            kernels.kick[blocks, gb.THREADS_PER_BLOCK](
                *particles, dk, do, dg, dec, des, di, *args)
        else:
            kernels.kick(*particles, k, omega, coupling, ec, es, integral, *args)

    def drift():
        if gpu:
            kernels.drift[blocks, gb.THREADS_PER_BLOCK](state["x"], state["v"], dt)
        else:
            kernels.drift(state["x"], state["v"], dt)

    def currents(mid):
        if gpu:
            sums_device.copy_to_device(sums_zero)
            kernels.current_sums[(blocks, len(k)), gb.THREADS_PER_BLOCK](
                state["x"], state["v"], state["w"], dk, do, mid, sums_device)
            return gb._allreduce_sums(sums_device.copy_to_host(), comm)
        return kernels.current_sums(state["x"], state["v"], state["w"], k, omega, mid)

    ec_hist, es_hist = np.empty((n_steps, len(k))), np.empty((n_steps, len(k)))
    completed = 0
    stopped = False
    interval = float(p.get("checkpoint_interval", 0.) or 0.)
    if not np.isfinite(interval) or interval < 0:
        raise ValueError("checkpoint_interval must be finite and nonnegative")
    deadline = monotonic() + interval
    checkpoint = p.get("checkpoint_file")

    def snapshot():
        host = {key: value.copy_to_host() if gpu else value.copy() for key, value in state.items()}
        # Use an absolute step clock: repeated chunking produces identical carrier phases.
        times = origin + (rng_step + np.arange(1, completed + 1)) * dt
        result = dict(host, E_c=ec.copy(), E_s=es.copy(),
                      E_c_hist=ec_hist[:completed].copy(), E_s_hist=es_hist[:completed].copy(),
                      time=times, time_origin=origin, t_start=t_start,
                      t_final=origin + (rng_step + completed) * dt, dt=dt,
                      completed_steps=completed, stopped_early=completed < n_steps,
                      rng_seed=rng_seed, rng_step=np.uint64(rng_step + completed),
                      collision_weight_scheme=np.array(base.WEIGHT_SCHEME),
                      mode_scheme=np.array(MODE_SCHEME), modes_json=np.array(mode_json(specs)),
                      physics_signature=np.array(signature), mode_k=k, mode_omega=omega,
                      mode_coupling=coupling, mode_gamma_d=damping,
                      mode_evolve=np.array([m.evolve for m in specs]),
                      backend=np.array("cuda_multimode" if gpu else "cpu_multimode"),
                      global_n_particles=count, local_n_particles=len(host["x"]),
                      particle_offset=offset, rank=rank, n_ranks=size,
                      restart_parent=np.array(str(p.get("restart_file", ""))),
                      restart_mode_policy=np.array(p.get("restart_mode_policy", "strict")))
        if rank == 0:
            result["rng_state"] = base._pack_rng_state(rng)
        result.update(resonance_diagnostics(result["E_c_hist"], result["E_s_hist"], times, specs))
        result["wave_energy_hist"] = np.sum(result["E_amp_hist"]**2, axis=1) / 2
        result["particle_energy_local"] = p["nb_over_ne"] * np.sum((host["p"] + host["w"]) * host["v"]**2) / (2 * count)
        bins = p.get("velocity_bins", np.linspace(p["ub"] - 6 * p["vb"], p["ub"] + 6 * p["vb"], 129))
        distribution = velocity_distribution(host["v"], host["w"], host["p"], bins, count)
        if gpu:
            for key in ("F0", "delta_f", "f"):
                distribution[key] = gb._allreduce_sums(distribution[key], comm)
        result.update(distribution)
        energy = np.array([result["particle_energy_local"]])
        result["particle_energy"] = (gb._allreduce_sums(energy, comm) if gpu else energy)[0]
        return result

    def save(output):
        if gpu:
            gb.save_ranked_outputs(checkpoint, output, comm)
        else:
            base._atomic_savez(checkpoint, output)

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    handlers = {}
    for signum in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if signum is not None:
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, request_stop)
                handlers[signum] = previous
            except (OSError, ValueError):
                pass
    try:
        for n in base._progress_iter(range(n_steps), enabled=rank == 0 and p.get("show_progress", False)):
            t = origin + (rng_step + n) * dt
            if p["splitting_method"] == "strang":
                push(t + .25 * dt, rng_step + n, False)
                ec, es = advance_fields(ec, es, currents(t + .25 * dt), h, count, p["nb_over_ne"], specs)
                drift()
                ec, es = advance_fields(ec, es, currents(t + .75 * dt), h, count, p["nb_over_ne"], specs)
                push(t + .75 * dt, rng_step + n, True)
            else:
                push(t + .5 * dt, rng_step + n, False)
                sums = currents(t + .5 * dt)
                drift()
                ec, es = advance_fields(ec, es, sums, h, count, p["nb_over_ne"], specs)
            ec_hist[n], es_hist[n] = ec, es
            completed = n + 1
            if gpu:
                stopped = gb._allreduce_sums(np.array([float(stopped)]), comm)[0] > 0
            if stopped:
                break
            due = bool(checkpoint and interval > 0 and monotonic() >= deadline)
            if gpu:
                due = comm.bcast(due if rank == 0 else None, root=0)
            if due:
                save(snapshot())
                deadline = monotonic() + interval
    finally:
        for signum, previous in handlers.items():
            signal.signal(signum, previous)
    output = snapshot()
    if checkpoint:
        save(output)
    return output
