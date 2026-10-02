"""Mixed constant-coefficient diffusion, drag, and Krook delta-f backend.

C(delta_f) = D*d_vv(delta_f) + A*d_v(delta_f) - beta*delta_f,
D=nu**3 >= 0, A=alpha**2 >= 0, beta>=0; Gaussian stationary F0.
Collision transition: dv=-A*h+sqrt(2*D*h)*normal, w'=exp(-beta*h)*w,
p'=p*F0(v+dv)/F0(v). Background-weighted moments preserve F0 in expectation.
With diffusion, weights are stochastic, not single-valued ratios to the
marginal marker density. The field kick uses an exact F0 ratio and expm1.
Only nonlinear delta-f is supported: 'full' means all three collisions,
not full-f or a general nonlinear Landau operator.
CPU API: run_timestepping(params); GPU: run_timestepping(params, backend="gpu").
CLI: python core_collision_full.py --backend cpu|gpu --params ... --output ...
GPU/MPI companion: core_collision_full_gpu.py. Existing files stay unchanged.
Optional params['modes'] enables the secondary-resonance traveling-wave extension.
See README.md for its configuration, diagnostics, and restart policies.
"""

from __future__ import annotations

import math
import os
import signal
from time import monotonic

import numpy as np


def _install_numba_cuda_numpy_compatibility():
    """Restore aliases required by Numba-CUDA releases predating NumPy 2.5."""
    if hasattr(np, "row_stack"):
        return

    # NumPy removed ``row_stack`` in 2.5 after deprecating it as an alias for
    # ``vstack``.  Some Numba-CUDA releases still look up the old name when
    # their CUDA registries are loaded lazily at the first kernel launch.
    def row_stack(arrays, *, dtype=None, casting="same_kind"):
        return np.vstack(arrays, dtype=dtype, casting=casting)

    setattr(np, "row_stack", row_stack)


_install_numba_cuda_numpy_compatibility()

from numba import config, cuda, float64, uint32, uint64
from wisp.diagnostics import compute_gamma

from core_collision_full import (
    WEIGHT_SCHEME,
    _validate_full_mode,
    _require_full_checkpoint,
    _run_full_cli,
    _atomic_savez,
    _damped_field_from_sums,
    _pack_rng_state,
    _progress_iter,
    _seed_to_uint64,
    _unpack_rng_state,
)

if config.ENABLE_CUDASIM or os.environ.get("WISP_GPU_SERIAL") == "1":
    # Some laptop MPI builds initialize an unavailable network provider during
    # import.  CUDASIM and explicit serial development do not need MPI.
    MPI = None
else:
    try:
        from mpi4py import MPI
    except ImportError:  # serial CUDA development remains available
        MPI = None


THREADS_PER_BLOCK = 256
_TWO_PI = 2.0 * math.pi
_UINT32_MASK_INT = 0xFFFFFFFF
_PHILOX_M0_INT = 0xD2511F53
_PHILOX_M1_INT = 0xCD9E8D57
_PHILOX_W0_INT = 0x9E3779B9
_PHILOX_W1_INT = 0xBB67AE85
_TWO_NEG_53 = 1.0 / 9007199254740992.0


class _SerialComm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, value, root=0):
        del root
        return value

    def Barrier(self):
        return None


_SERIAL_COMM = _SerialComm()


def _get_comm(comm=None):
    if comm is not None:
        return comm
    if MPI is not None:
        return MPI.COMM_WORLD
    return _SERIAL_COMM


@cuda.jit(device=True, inline=True)
def _mulhilo32_cuda(a, b):
    product = uint64(a) * uint64(b)
    low = uint32(product & uint64(_UINT32_MASK_INT))
    high = uint32(product >> uint64(32))
    return high, low


@cuda.jit(device=True, inline=True)
def _philox4x32_10_cuda(particle_index, step_index, seed):
    particle = uint64(particle_index)
    step = uint64(step_index)
    seed64 = uint64(seed)
    c0 = uint32(particle & uint64(_UINT32_MASK_INT))
    c1 = uint32(particle >> uint64(32))
    c2 = uint32(step & uint64(_UINT32_MASK_INT))
    c3 = uint32(step >> uint64(32))
    k0 = uint32(seed64 & uint64(_UINT32_MASK_INT))
    k1 = uint32(seed64 >> uint64(32))

    for _ in range(10):
        hi0, lo0 = _mulhilo32_cuda(uint32(_PHILOX_M0_INT), c0)
        hi1, lo1 = _mulhilo32_cuda(uint32(_PHILOX_M1_INT), c2)
        n0 = uint32(hi1 ^ c1 ^ k0)
        n1 = lo1
        n2 = uint32(hi0 ^ c3 ^ k1)
        n3 = lo0
        c0, c1, c2, c3 = n0, n1, n2, n3
        k0 = uint32(
            (uint64(k0) + uint64(_PHILOX_W0_INT))
            & uint64(_UINT32_MASK_INT)
        )
        k1 = uint32(
            (uint64(k1) + uint64(_PHILOX_W1_INT))
            & uint64(_UINT32_MASK_INT)
        )
    return c0, c1, c2, c3


@cuda.jit(device=True, inline=True)
def _uniform53_cuda(high_word, low_word):
    mantissa = (
        (uint64(high_word) >> uint64(5)) * uint64(67108864)
        + (uint64(low_word) >> uint64(6))
    )
    return (float64(mantissa) + 0.5) * _TWO_NEG_53


@cuda.jit(device=True, inline=True)
def _normal_pair_cuda(particle_index, step_index, seed):
    r0, r1, r2, r3 = _philox4x32_10_cuda(particle_index, step_index, seed)
    u1 = _uniform53_cuda(r0, r1)
    u2 = _uniform53_cuda(r2, r3)
    radius = math.sqrt(-2.0 * math.log(u1))
    angle = _TWO_PI * u2
    return radius * math.cos(angle), radius * math.sin(angle)


@cuda.jit(device=True, inline=True)
def _wrap_two_pi_cuda(value):
    return value - _TWO_PI * math.floor(value / _TWO_PI)


@cuda.jit(device=True, inline=True)
def _collision_cuda(
    vi,
    wi,
    pi,
    normal,
    collision_ds,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
):
    # Translation-invariant diffusion/drag kernel plus background importance weight.
    delta_v = -nu_drag * collision_ds + diffusion_sigma * normal
    exponent = -delta_v * (2.0 * (vi - ub) + delta_v) * inv_vb2
    return vi + delta_v, wi * krook_factor, pi * math.exp(exponent)


@cuda.jit(device=True, inline=True)
def _weight_velocity_cuda(
    vi,
    wi,
    pi,
    phase_cos,
    phase_sin,
    e_c,
    e_s,
    impulse_factor,
    ub,
    inv_vb2,
    nonlinear,
    evolve_weights,
):
    velocity_impulse = impulse_factor * (e_c * phase_cos + e_s * phase_sin)
    if evolve_weights:
        dw = velocity_impulse * 2.0 * (vi - ub) * inv_vb2
        if nonlinear:
            exponent = -velocity_impulse * (2.0 * (vi - ub) + velocity_impulse) * inv_vb2
            wi -= pi * math.expm1(exponent)
            pi *= math.exp(exponent)
        else:
            wi += dw
    if nonlinear:
        vi += velocity_impulse
    return vi, wi, pi


@cuda.jit
def _reset_sums_cuda(sums):
    i = cuda.grid(1)
    if i < len(sums):
        sums[i] = 0.0


@cuda.jit
def _strang_first_cuda(
    x,
    v,
    w,
    p,
    second_normal,
    sums,
    e_c,
    e_s,
    t,
    dt,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
    evolve_weights,
    collisions_enabled,
    rng_seed,
    rng_step,
    particle_offset,
):
    shared_cos = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    shared_sin = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    i = cuda.grid(1)
    lane = cuda.threadIdx.x
    local_cos = 0.0
    local_sin = 0.0

    if i < len(x):
        xi = x[i]
        vi = v[i]
        wi = w[i]
        pi = p[i]
        if collisions_enabled:
            z_first, z_second = _normal_pair_cuda(
                particle_offset + i, rng_step, rng_seed
            )
            second_normal[i] = z_second
            vi, wi, pi = _collision_cuda(
                vi,
                wi,
                pi,
                z_first,
                0.5 * dt,
                ub,
                inv_vb2,
                krook_factor,
                source_factor,
                nu_drag,
                nu_diff,
                diffusion_sigma,
                nonlinear,
            )
        phase = xi - (t + 0.25 * dt)
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        vi, wi, pi = _weight_velocity_cuda(
            vi,
            wi,
            pi,
            phase_cos,
            phase_sin,
            e_c,
            e_s,
            -2.0 * math.sin(0.25 * dt),
            ub,
            inv_vb2,
            nonlinear,
            evolve_weights,
        )
        v[i] = vi
        w[i] = wi
        p[i] = pi
        current = vi * wi
        local_cos = current * phase_cos
        local_sin = current * phase_sin

    shared_cos[lane] = local_cos
    shared_sin[lane] = local_sin
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if lane < stride:
            shared_cos[lane] += shared_cos[lane + stride]
            shared_sin[lane] += shared_sin[lane + stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        cuda.atomic.add(sums, 0, shared_cos[0])
        cuda.atomic.add(sums, 1, shared_sin[0])


@cuda.jit
def _position_and_field_cuda(x, v, w, sums, t, dt):
    shared_cos = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    shared_sin = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    i = cuda.grid(1)
    lane = cuda.threadIdx.x
    local_cos = 0.0
    local_sin = 0.0
    if i < len(x):
        xi = _wrap_two_pi_cuda(x[i] + v[i] * dt)
        x[i] = xi
        phase = xi - (t + 0.75 * dt)
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        current = v[i] * w[i]
        local_cos = current * phase_cos
        local_sin = current * phase_sin
    shared_cos[lane] = local_cos
    shared_sin[lane] = local_sin
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if lane < stride:
            shared_cos[lane] += shared_cos[lane + stride]
            shared_sin[lane] += shared_sin[lane + stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        cuda.atomic.add(sums, 0, shared_cos[0])
        cuda.atomic.add(sums, 1, shared_sin[0])


@cuda.jit
def _strang_last_cuda(
    x,
    v,
    w,
    p,
    second_normal,
    e_c,
    e_s,
    t,
    dt,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
    evolve_weights,
    collisions_enabled,
):
    i = cuda.grid(1)
    if i < len(x):
        vi = v[i]
        wi = w[i]
        pi = p[i]
        phase = x[i] - (t + 0.75 * dt)
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        vi, wi, pi = _weight_velocity_cuda(
            vi,
            wi,
            pi,
            phase_cos,
            phase_sin,
            e_c,
            e_s,
            -2.0 * math.sin(0.25 * dt),
            ub,
            inv_vb2,
            nonlinear,
            evolve_weights,
        )
        if collisions_enabled:
            vi, wi, pi = _collision_cuda(
                vi,
                wi,
                pi,
                second_normal[i],
                0.5 * dt,
                ub,
                inv_vb2,
                krook_factor,
                source_factor,
                nu_drag,
                nu_diff,
                diffusion_sigma,
                nonlinear,
            )
        v[i] = vi
        w[i] = wi
        p[i] = pi


@cuda.jit
def _trotter_cuda(
    x,
    v,
    w,
    p,
    sums,
    e_c,
    e_s,
    t,
    dt,
    ub,
    inv_vb2,
    krook_factor,
    source_factor,
    nu_drag,
    nu_diff,
    diffusion_sigma,
    nonlinear,
    evolve_weights,
    collisions_enabled,
    rng_seed,
    rng_step,
    particle_offset,
):
    shared_cos = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    shared_sin = cuda.shared.array(THREADS_PER_BLOCK, dtype=float64)
    i = cuda.grid(1)
    lane = cuda.threadIdx.x
    local_cos = 0.0
    local_sin = 0.0
    if i < len(x):
        xi = x[i]
        vi = v[i]
        wi = w[i]
        pi = p[i]
        if collisions_enabled:
            z_first, _ = _normal_pair_cuda(particle_offset + i, rng_step, rng_seed)
            vi, wi, pi = _collision_cuda(
                vi,
                wi,
                pi,
                z_first,
                dt,
                ub,
                inv_vb2,
                krook_factor,
                source_factor,
                nu_drag,
                nu_diff,
                diffusion_sigma,
                nonlinear,
            )
        phase = xi - (t + 0.5 * dt)
        phase_cos = math.cos(phase)
        phase_sin = math.sin(phase)
        vi, wi, pi = _weight_velocity_cuda(
            vi,
            wi,
            pi,
            phase_cos,
            phase_sin,
            e_c,
            e_s,
            -2.0 * math.sin(0.5 * dt),
            ub,
            inv_vb2,
            nonlinear,
            evolve_weights,
        )
        current = vi * wi
        local_cos = current * phase_cos
        local_sin = current * phase_sin
        x[i] = _wrap_two_pi_cuda(xi + vi * dt)
        v[i] = vi
        w[i] = wi
        p[i] = pi
    shared_cos[lane] = local_cos
    shared_sin[lane] = local_sin
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if lane < stride:
            shared_cos[lane] += shared_cos[lane + stride]
            shared_sin[lane] += shared_sin[lane + stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        cuda.atomic.add(sums, 0, shared_cos[0])
        cuda.atomic.add(sums, 1, shared_sin[0])


def _partition(global_count, size):
    counts = np.full(size, global_count // size, dtype=np.int64)
    counts[: global_count % size] += 1
    offsets = np.zeros(size, dtype=np.int64)
    offsets[1:] = np.cumsum(counts[:-1])
    return counts, offsets


def _scatter_array(global_array, counts, offsets, comm):
    rank = comm.Get_rank()
    local = np.empty(int(counts[rank]), dtype=np.float64)
    if comm.Get_size() == 1:
        local[:] = global_array
        return local
    if MPI is None:  # pragma: no cover - guarded by communicator size
        raise RuntimeError("mpi4py is required for multi-GPU execution")
    send_spec = [global_array, counts, offsets, MPI.DOUBLE] if rank == 0 else None
    comm.Scatterv(send_spec, local, root=0)
    return local


def _allreduce_sums(local_sums, comm):
    if comm.Get_size() == 1:
        return local_sums
    if MPI is None:  # pragma: no cover - guarded by communicator size
        raise RuntimeError("mpi4py is required for multi-GPU execution")
    comm.Allreduce(MPI.IN_PLACE, local_sums, op=MPI.SUM)
    return local_sums


def _rank_filename(filename, rank, generation=None):
    filename = os.fspath(filename)
    stem, extension = os.path.splitext(filename)
    if extension.lower() != ".npz":
        stem, extension = filename, ".npz"
    generation_tag = "" if generation is None else f".step{int(generation):012d}"
    return f"{stem}{generation_tag}.rank{rank:04d}{extension}"


def save_ranked_outputs(filename, outputs, comm=None):
    """Atomically save one particle shard per rank plus a rank-zero manifest."""
    comm = _get_comm(comm)
    rank = comm.Get_rank()
    generation = int(outputs["rng_step"])
    shard_filename = _rank_filename(filename, rank, generation)
    shard = {
        "x": outputs["x"],
        "v": outputs["v"],
        "w": outputs["w"],
        "p": outputs["p"],
        "particle_offset": outputs["particle_offset"],
        "local_n_particles": outputs["local_n_particles"],
        "global_n_particles": outputs["global_n_particles"],
        "rng_seed": outputs["rng_seed"],
        "rng_step": outputs["rng_step"],
        "rank": rank,
        "n_ranks": comm.Get_size(),
        "generation": generation,
        "backend": np.array("cuda_mpi_fused"),
        "collision_weight_scheme": np.array(WEIGHT_SCHEME),
    }
    # A new immutable shard generation is committed before the manifest.  If
    # the process is killed while any shard is being written, the old manifest
    # still points to the previous complete generation.
    _atomic_savez(shard_filename, shard)
    comm.Barrier()
    if rank == 0:
        manifest = {
            key: value
            for key, value in outputs.items()
            if key not in {"x", "v", "w", "p"}
        }
        manifest["n_ranks"] = comm.Get_size()
        manifest["generation"] = generation
        rank_zero_name = os.path.basename(_rank_filename(filename, 0, generation))
        manifest["shard_pattern"] = np.array(
            rank_zero_name.replace(".rank0000", ".rank{rank:04d}")
        )
        _atomic_savez(filename, manifest)
    comm.Barrier()


def _load_restart(restart_file, sim_params, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == 0:
        with np.load(restart_file, allow_pickle=True) as state:
            metadata = {
                "collision_weight_scheme": str(np.asarray(state.get("collision_weight_scheme", "")).item()),
                "E_c": float(state["E_c"]),
                "E_s": float(state["E_s"]),
                "t_final": float(state.get("t_final", 0.0)),
                "rng_seed": int(state["rng_seed"]) if "rng_seed" in state else None,
                "rng_step": int(state.get("rng_step", state.get("completed_steps", 0))),
                "n_ranks": int(state.get("n_ranks", 0)),
                "shard_pattern": (
                    str(np.asarray(state["shard_pattern"]).item())
                    if "shard_pattern" in state
                    else None
                ),
                "global_n_particles": int(
                    state.get("global_n_particles", len(state["x"]) if "x" in state else 0)
                ),
                "has_full_state": all(key in state for key in ("x", "v", "w")),
                "has_background_weights": "p" in state,
                "rng_state": state["rng_state"] if "rng_state" in state else None,
            }
            if metadata["has_full_state"]:
                full_state = {
                    "x": state["x"].copy(),
                    "v": state["v"].copy(),
                    "w": state["w"].copy(),
                    "p": state["p"].copy() if "p" in state else None,
                }
            else:
                full_state = None
    else:
        metadata = None
        full_state = None
    metadata = comm.bcast(metadata, root=0)
    # All ranks validate after broadcast so rejection is collective.
    _require_full_checkpoint(metadata["collision_weight_scheme"])
    if metadata["has_full_state"] and not metadata["has_background_weights"]:
        raise ValueError("Mixed-collision checkpoint must contain background weights p")

    if metadata["rng_seed"] is None:
        seed = sim_params.get("seed")
        if seed is None:
            raise ValueError(
                "A legacy checkpoint has no rng_seed; supply sim_params['seed'] "
                "to begin the optimized Philox stream explicitly."
            )
        metadata["rng_seed"] = int(seed) & ((1 << 64) - 1)

    if metadata["has_full_state"]:
        counts, offsets = _partition(metadata["global_n_particles"], size)
        local_state = {
            key: _scatter_array(
                full_state[key] if rank == 0 else None, counts, offsets, comm
            )
            for key in ("x", "v", "w", "p")
        }
        particle_offset = int(offsets[rank])
    else:
        if metadata["n_ranks"] != size:
            raise ValueError(
                f"Checkpoint has {metadata['n_ranks']} shards but this run has {size} ranks"
            )
        if metadata["shard_pattern"] is None:
            shard_file = _rank_filename(restart_file, rank)
        else:
            shard_file = os.path.join(
                os.path.dirname(os.path.abspath(os.fspath(restart_file))),
                metadata["shard_pattern"].format(rank=rank),
            )
        with np.load(shard_file, allow_pickle=True) as shard:
            local_state = {key: shard[key].copy() for key in ("x", "v", "w", "p")}
            particle_offset = int(shard["particle_offset"])
    return metadata, local_state, particle_offset


def _make_local_outputs(
    e_c,
    e_s,
    x,
    v,
    w,
    p,
    particle_offset,
    global_n_particles,
    t_start,
    dt,
    e_c_hist,
    e_s_hist,
    e_amp_hist,
    completed_steps,
    rng_state,
    rng_seed,
    rng_step_start,
    stopped_early,
    rank,
    size,
):
    outputs = {
        "time": t_start + np.arange(completed_steps) * dt,
        "t_start": t_start,
        "t_final": t_start + completed_steps * dt,
        "completed_steps": completed_steps,
        "stopped_early": stopped_early,
        "E_c": e_c,
        "E_s": e_s,
        "x": x,
        "v": v,
        "w": w,
        "p": p,
        "E_c_hist": e_c_hist[:completed_steps],
        "E_s_hist": e_s_hist[:completed_steps],
        "E_amp_hist": e_amp_hist[:completed_steps],
        "phi_hist": np.arctan2(
            e_s_hist[:completed_steps], e_c_hist[:completed_steps]
        ),
        "rng_seed": np.uint64(rng_seed),
        "rng_step": np.uint64(rng_step_start + completed_steps),
        "particle_offset": particle_offset,
        "local_n_particles": len(x),
        "global_n_particles": global_n_particles,
        "rank": rank,
        "n_ranks": size,
        "backend": np.array("cuda_mpi_fused"),
        "collision_weight_scheme": np.array(WEIGHT_SCHEME),
    }
    if rank == 0 and rng_state is not None:
        outputs["rng_state"] = rng_state
    return outputs


def _copy_device_state(d_x, d_v, d_w, d_p):
    return (
        d_x.copy_to_host(),
        d_v.copy_to_host(),
        d_w.copy_to_host(),
        d_p.copy_to_host(),
    )


def run_timestepping(sim_params, comm=None):
    """Run a distributed GPU timestep chunk and return this rank's state."""
    if "modes" in sim_params:
        from secondary_runner import run_multimode
        return run_multimode(sim_params, backend="gpu", comm=comm)
    comm = _get_comm(comm)
    rank = comm.Get_rank()
    size = comm.Get_size()
    expected_gpus = sim_params.get("expected_gpus")
    if expected_gpus is not None and size != int(expected_gpus):
        raise ValueError(f"Expected {expected_gpus} MPI ranks/GPUs, received {size}")
    if size > 1 and MPI is None:
        raise RuntimeError("mpi4py is required for multi-GPU execution")

    if not config.ENABLE_CUDASIM:
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
        visible_devices = [
            device
            for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            if device
        ]
        # ``--gpus-per-task=1`` exposes one device to each rank and renumbers
        # it to ordinal zero.  The fallback also supports an unrestricted
        # development allocation where all devices remain visible.
        device_ordinal = 0 if len(visible_devices) == 1 else local_rank
        cuda.select_device(device_ordinal)

    e_c = float(sim_params["E_c"])
    e_s = float(sim_params["E_s"])
    nb_over_ne = float(sim_params["nb_over_ne"])
    n_steps = int(sim_params["n_steps"])
    dt = float(sim_params["dt"])
    global_n_particles = int(sim_params["n_particles"])
    ub = float(sim_params["ub"])
    vb = float(sim_params["vb"])
    splitting_method = sim_params["splitting_method"]
    method = sim_params["method"]
    for key in ("full_f", "show_progress"):
        if key in sim_params and not isinstance(sim_params[key], (bool, np.bool_)):
            raise ValueError(f"{key} must be a JSON boolean, not a string or number")
    full_f = bool(sim_params["full_f"])
    nonlinear = method == "nonlinear"
    evolve_weights = not full_f

    if splitting_method not in {"strang", "trotter"}:
        raise ValueError("splitting_method must be 'strang' or 'trotter'")
    if vb == 0.0:
        raise ValueError("vb must be non-zero")
    if n_steps < 0 or global_n_particles <= 0:
        raise ValueError("n_steps must be non-negative and n_particles positive")

    gamma_ratio = float(sim_params.get("gamma_ratio", 0.9))
    gamma_l = float(compute_gamma(sim_params))
    gamma_d = float(sim_params.get("gamma_d", gamma_ratio * gamma_l))
    nu_norm = float(sim_params.get("nu_norm", 0.0))
    nu = float(sim_params.get("nu", (1.0 - gamma_ratio) * nu_norm * gamma_l))
    nu_diff = nu**3
    alpha_norm = float(sim_params.get("alpha_norm", 0.0))
    alpha = float(
        sim_params.get("alpha", (1.0 - gamma_ratio) * alpha_norm * gamma_l)
    )
    nu_drag = alpha**2
    beta_norm = float(sim_params.get("beta_norm", 0.0))
    nu_krook = float(
        sim_params.get("beta", (1.0 - gamma_ratio) * beta_norm * gamma_l)
    )
    _validate_full_mode(method, full_f, nu, nu_krook, dt, vb, alpha)
    collisions_enabled = evolve_weights and (
        nu_krook != 0.0 or nu_drag != 0.0 or nu_diff != 0.0
    )

    if rank == 0:
        print(f"Backend: mixed-collision Numba CUDA ({size} MPI rank(s))")
        print("Damping-to-growth ratio: ", gamma_ratio)
        print("Wave damping rate: ", gamma_d)
        print("Normalized diffusion rate: ", nu_norm)
        print("Diffusion rate: ", nu_diff)

    seed = sim_params.get("seed")
    rng = np.random.default_rng(seed if rank == 0 else 0)
    rng_state = None
    t_start = float(sim_params.get("t_start", 0.0))
    rng_step_start = 0
    restart_file = sim_params.get("restart_file")

    if restart_file is not None:
        metadata, local_state, particle_offset = _load_restart(
            restart_file, sim_params, comm
        )
        e_c = metadata["E_c"]
        e_s = metadata["E_s"]
        global_n_particles = metadata["global_n_particles"]
        if "t_start" not in sim_params:
            t_start = metadata["t_final"]
        rng_seed = np.uint64(metadata["rng_seed"])
        rng_step_start = metadata["rng_step"]
        if rank == 0 and metadata["rng_state"] is not None:
            rng.bit_generator.state = _unpack_rng_state(metadata["rng_state"])
            rng_state = _pack_rng_state(rng)
        x_local = local_state["x"]
        v_local = local_state["v"]
        w_local = local_state["w"]
        p_local = local_state["p"]
    else:
        counts, offsets = _partition(global_n_particles, size)
        if rank == 0:
            x_global = rng.uniform(0.0, _TWO_PI, global_n_particles)
            v_global = rng.normal(ub, vb / np.sqrt(2.0), global_n_particles)
            w_global = (
                np.ones(global_n_particles)
                if full_f
                else np.zeros(global_n_particles)
            )
            p_global = np.ones(global_n_particles)
            rng_seed_value = int(_seed_to_uint64(seed, rng))
            rng_state = _pack_rng_state(rng)
        else:
            x_global = v_global = w_global = p_global = None
            rng_seed_value = None
        rng_seed = np.uint64(comm.bcast(rng_seed_value, root=0))
        x_local = _scatter_array(x_global, counts, offsets, comm)
        v_local = _scatter_array(v_global, counts, offsets, comm)
        w_local = _scatter_array(w_global, counts, offsets, comm)
        p_local = _scatter_array(p_global, counts, offsets, comm)
        particle_offset = int(offsets[rank])

    local_count = len(x_local)
    d_x = cuda.to_device(x_local)
    d_v = cuda.to_device(v_local)
    d_w = cuda.to_device(w_local)
    d_p = cuda.to_device(p_local)
    d_second_normal = cuda.device_array(local_count, dtype=np.float64)
    d_sums = cuda.device_array(3, dtype=np.float64)
    blocks = (local_count + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    e_c_hist = np.zeros(n_steps)
    e_s_hist = np.zeros(n_steps)
    e_amp_hist = np.zeros(n_steps)
    time = t_start + np.arange(n_steps) * dt
    completed_steps = 0
    stop_requested = False
    global_stop = False

    inv_vb2 = 1.0 / (vb * vb)
    collision_ds = 0.5 * dt if splitting_method == "strang" else dt
    krook_factor = np.exp(-nu_krook * collision_ds)
    source_factor = (
        (1.0 - krook_factor) / nu_krook if nu_krook > 0.0 else collision_ds
    )
    diffusion_sigma = (
        np.sqrt(2.0 * nu_diff * collision_ds) if nu_diff > 0.0 else 0.0
    )

    checkpoint_file = sim_params.get("checkpoint_file")
    checkpoint_interval = float(sim_params.get("checkpoint_interval", 0.0) or 0.0)
    checkpoint_check_steps = max(1, int(sim_params.get("checkpoint_check_steps", 100)))
    if checkpoint_interval < 0.0:
        raise ValueError("checkpoint_interval must be non-negative")
    next_checkpoint = (
        monotonic() + checkpoint_interval if checkpoint_interval > 0.0 else None
    )

    def request_stop(signum, frame):
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    installed_signal_handlers = {}
    for signal_number in (signal.SIGTERM, getattr(signal, "SIGUSR1", None)):
        if signal_number is None:
            continue
        try:
            installed_signal_handlers[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, request_stop)
        except (OSError, TypeError, ValueError):
            continue

    show_progress = rank == 0 and bool(
        sim_params.get("show_progress", sim_params.get("progress_bar", False))
    )
    step_iter = _progress_iter(
        range(n_steps),
        enabled=show_progress,
        total=n_steps,
        desc="GPU timestepping",
        unit="step",
    )

    def reduced_device_sums():
        local_sums = d_sums.copy_to_host()
        local_sums[2] = 1.0 if stop_requested else 0.0
        return _allreduce_sums(local_sums, comm)

    def snapshot(stopped_early):
        x_host, v_host, w_host, p_host = _copy_device_state(d_x, d_v, d_w, d_p)
        return _make_local_outputs(
            e_c,
            e_s,
            x_host,
            v_host,
            w_host,
            p_host,
            particle_offset,
            global_n_particles,
            t_start,
            dt,
            e_c_hist,
            e_s_hist,
            e_amp_hist,
            completed_steps,
            rng_state,
            rng_seed,
            rng_step_start,
            stopped_early,
            rank,
            size,
        )

    try:
        for n in step_iter:
            if global_stop:
                break
            t = time[n]
            rng_step = rng_step_start + n

            _reset_sums_cuda[1, 32](d_sums)
            if splitting_method == "strang":
                _strang_first_cuda[blocks, THREADS_PER_BLOCK](
                    d_x,
                    d_v,
                    d_w,
                    d_p,
                    d_second_normal,
                    d_sums,
                    e_c,
                    e_s,
                    t,
                    dt,
                    ub,
                    inv_vb2,
                    krook_factor,
                    source_factor,
                    nu_drag,
                    nu_diff,
                    diffusion_sigma,
                    nonlinear,
                    evolve_weights,
                    collisions_enabled,
                    rng_seed,
                    rng_step,
                    particle_offset,
                )
                sums = reduced_device_sums()
                global_stop = sums[2] > 0.0
                e_c, e_s = _damped_field_from_sums(
                    e_c,
                    e_s,
                    sums[0],
                    sums[1],
                    0.5 * dt,
                    global_n_particles,
                    nb_over_ne,
                    gamma_d,
                )

                _reset_sums_cuda[1, 32](d_sums)
                _position_and_field_cuda[blocks, THREADS_PER_BLOCK](
                    d_x, d_v, d_w, d_sums, t, dt
                )
                sums = reduced_device_sums()
                global_stop = global_stop or sums[2] > 0.0
                e_c, e_s = _damped_field_from_sums(
                    e_c,
                    e_s,
                    sums[0],
                    sums[1],
                    0.5 * dt,
                    global_n_particles,
                    nb_over_ne,
                    gamma_d,
                )
                _strang_last_cuda[blocks, THREADS_PER_BLOCK](
                    d_x,
                    d_v,
                    d_w,
                    d_p,
                    d_second_normal,
                    e_c,
                    e_s,
                    t,
                    dt,
                    ub,
                    inv_vb2,
                    krook_factor,
                    source_factor,
                    nu_drag,
                    nu_diff,
                    diffusion_sigma,
                    nonlinear,
                    evolve_weights,
                    collisions_enabled,
                )
            else:
                _trotter_cuda[blocks, THREADS_PER_BLOCK](
                    d_x,
                    d_v,
                    d_w,
                    d_p,
                    d_sums,
                    e_c,
                    e_s,
                    t,
                    dt,
                    ub,
                    inv_vb2,
                    krook_factor,
                    source_factor,
                    nu_drag,
                    nu_diff,
                    diffusion_sigma,
                    nonlinear,
                    evolve_weights,
                    collisions_enabled,
                    rng_seed,
                    rng_step,
                    particle_offset,
                )
                sums = reduced_device_sums()
                global_stop = sums[2] > 0.0
                e_c, e_s = _damped_field_from_sums(
                    e_c,
                    e_s,
                    sums[0],
                    sums[1],
                    dt,
                    global_n_particles,
                    nb_over_ne,
                    gamma_d,
                )

            e_c_hist[n] = e_c
            e_s_hist[n] = e_s
            e_amp_hist[n] = np.hypot(e_c, e_s)
            completed_steps = n + 1

            checkpoint_due = False
            if completed_steps % checkpoint_check_steps == 0:
                if rank == 0:
                    checkpoint_due = (
                        checkpoint_file is not None
                        and next_checkpoint is not None
                        and monotonic() >= next_checkpoint
                    )
                checkpoint_due = comm.bcast(checkpoint_due, root=0)
            if checkpoint_due:
                save_ranked_outputs(checkpoint_file, snapshot(False), comm)
                next_checkpoint = monotonic() + checkpoint_interval
            if global_stop:
                break
    finally:
        for signal_number, previous_handler in installed_signal_handlers.items():
            signal.signal(signal_number, previous_handler)

    outputs = snapshot(completed_steps < n_steps)
    if checkpoint_file is not None:
        save_ranked_outputs(checkpoint_file, outputs, comm)
    return outputs


__all__ = ["run_timestepping", "save_ranked_outputs"]


if __name__ == "__main__":
    _run_full_cli(run_timestepping, save_ranked_outputs, "gpu")
