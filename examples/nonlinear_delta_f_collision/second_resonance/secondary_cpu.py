"""Bounded-memory multimode CPU kernels, reusing the full-collision weights/RNG."""
import numpy as np
from numba import njit, prange
from core_collision_full import (
    _mixed_collision_push, _field_weight_push, _normal_pair, _wrap_two_pi,
)


@njit(parallel=True, cache=True)
def kick(x, v, w, p, k, omega, coupling, ec, es, integral, t_mid,
         h, ub, inv_vb2, krook, drag, sigma, collisions, after, seed, step, offset):
    for i in prange(len(x)):
        vi, wi, pi = v[i], w[i], p[i]
        z = 0.0
        if collisions:
            z0, z1 = _normal_pair(offset + i, step, seed)
            z = z1 if after else z0
            if not after:
                vi, wi, pi = _mixed_collision_push(
                    vi, wi, pi, h, ub, inv_vb2, krook, drag, sigma, z)
        dv = 0.0
        for a in range(len(k)):
            phase = k[a] * x[i] - omega[a] * t_mid
            dv -= coupling[a] * integral[a] * (
                ec[a] * np.cos(phase) + es[a] * np.sin(phase))
        wi, pi = _field_weight_push(vi, wi, pi, dv, ub, inv_vb2)
        vi += dv
        if collisions and after:
            vi, wi, pi = _mixed_collision_push(
                vi, wi, pi, h, ub, inv_vb2, krook, drag, sigma, z)
        v[i], w[i], p[i] = vi, wi, pi


@njit(parallel=True, cache=True)
def drift(x, v, dt):
    for i in prange(len(x)):
        x[i] = _wrap_two_pi(x[i] + v[i] * dt)


@njit(parallel=True, cache=True)
def current_sums(x, v, w, k, omega, t_mid):
    # Fixed particle chunks make the reduction order independent of thread
    # scheduling, allocation alignment, and checkpoint boundaries.
    chunk_size = 1024
    chunks = (len(x) + chunk_size - 1) // chunk_size
    partial = np.zeros((chunks, len(k), 2))
    for chunk in prange(chunks):
        for a in range(len(k)):
            sc, ss = 0.0, 0.0
            for i in range(chunk * chunk_size, min((chunk + 1) * chunk_size, len(x))):
                phase = k[a] * x[i] - omega[a] * t_mid
                sc += v[i] * w[i] * np.cos(phase)
                ss += v[i] * w[i] * np.sin(phase)
            partial[chunk, a, 0], partial[chunk, a, 1] = sc, ss
    sums = np.zeros((len(k), 2))
    for a in range(len(k)):
        for chunk in range(chunks):
            sums[a, 0] += partial[chunk, a, 0]
            sums[a, 1] += partial[chunk, a, 1]
    return sums
