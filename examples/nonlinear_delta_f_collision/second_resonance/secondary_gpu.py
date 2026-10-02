"""CUDA multimode kernels; particle storage O(N), mode storage O(M)."""
import math
from numba import cuda, float64
from core_collision_full_gpu import (
    THREADS_PER_BLOCK, _normal_pair_cuda, _collision_cuda,
    _weight_velocity_cuda, _wrap_two_pi_cuda,
)


@cuda.jit
def kick(x, v, w, p, k, omega, coupling, ec, es, integral, t_mid,
         h, ub, inv_vb2, krook, drag, sigma, collisions, after, seed, step, offset):
    i = cuda.grid(1)
    if i < len(x):
        vi, wi, pi = v[i], w[i], p[i]
        z = 0.0
        if collisions:
            z0, z1 = _normal_pair_cuda(offset + i, step, seed)
            z = z1 if after else z0
            if not after:
                vi, wi, pi = _collision_cuda(
                    vi, wi, pi, z, h, ub, inv_vb2, krook, 0., drag, 0., sigma, True)
        dv = 0.0
        for a in range(len(k)):
            phase = k[a] * x[i] - omega[a] * t_mid
            dv -= coupling[a] * integral[a] * (
                ec[a] * math.cos(phase) + es[a] * math.sin(phase))
        vi, wi, pi = _weight_velocity_cuda(
            vi, wi, pi, 1., 0., dv, 0., 1., ub, inv_vb2, True, True)
        if collisions and after:
            vi, wi, pi = _collision_cuda(
                vi, wi, pi, z, h, ub, inv_vb2, krook, 0., drag, 0., sigma, True)
        v[i], w[i], p[i] = vi, wi, pi


@cuda.jit
def drift(x, v, dt):
    i = cuda.grid(1)
    if i < len(x):
        x[i] = _wrap_two_pi_cuda(x[i] + v[i] * dt)


@cuda.jit
def current_sums(x, v, w, k, omega, t_mid, sums):
    sc = cuda.shared.array(THREADS_PER_BLOCK, float64)
    ss = cuda.shared.array(THREADS_PER_BLOCK, float64)
    lane = cuda.threadIdx.x
    i = cuda.blockIdx.x * cuda.blockDim.x + lane
    a = cuda.blockIdx.y
    c, s = 0.0, 0.0
    if i < len(x):
        phase = k[a] * x[i] - omega[a] * t_mid
        c = v[i] * w[i] * math.cos(phase)
        s = v[i] * w[i] * math.sin(phase)
    sc[lane], ss[lane] = c, s
    cuda.syncthreads()
    stride = cuda.blockDim.x // 2
    while stride > 0:
        if lane < stride:
            sc[lane] += sc[lane + stride]
            ss[lane] += ss[lane + stride]
        cuda.syncthreads()
        stride //= 2
    if lane == 0:
        cuda.atomic.add(sums, (a, 0), sc[0])
        cuda.atomic.add(sums, (a, 1), ss[0])
