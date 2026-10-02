# Secondary resonances in the reduced WISP model

The local `core_collision_full.py` and `core_collision_full_gpu.py` now accept
`modes`. With no `modes` key they retain their original scalar paths, output
shapes, collision schemes, and checkpoint behavior. The files in `full_collision`
are not modified. Run from this directory so Python imports these local solvers.

This is the proposed **1D traveling-wave model** for studying additional resonant
velocities, perturbations of trapped orbits, and nonlinear mode competition. It
does not add gyrophase or perpendicular action and cannot represent a second
cyclotron harmonic. Multiple carriers alone do not demonstrate secondary islands
or chaotic transport; those require converged orbit/distribution measurements.

## Equations and conventions

For each mode, `psi_a = k_a*x - omega_a*t`, with x periodic on [0, 2*pi):

```text
dx/dt = v
dv/dt = -sum_a g_a (Ec_a cos(psi_a) + Es_a sin(psi_a))
dEc_a/dt = (nb/ne) g_a <v*w*cos(psi_a)> - gamma_d,a Ec_a
dEs_a/dt = (nb/ne) g_a <v*w*sin(psi_a)> - gamma_d,a Es_a
C(delta f) = nu^3 d_vv(delta f) + alpha^2 d_v(delta f) - beta delta f
F0(v) = exp(-(v-ub)^2/vb^2)/(sqrt(pi)*vb)
```

Angle brackets are marker averages (global N under MPI). Both current signs
match the supplied full-collision solvers. For A=Ec+i Es, the field is
Re[A exp(-i psi)] and the instantaneous frequency is omega + d(arg A)/dt.
The envelope is evolved about a prescribed carrier; no dispersion solver is
added. The existing k=omega=1 `compute_gamma` is used only as the reference for
legacy normalized collision rates and default damping. Supply explicit
per-mode `gamma_d` when the modes have different damping; this default is not a
generalized dispersion prediction.

Every field kick sums all mode impulses first and then applies the original
exact Gaussian background-weight ratio, preserving w+p during the field kick.
Collision kicks retain the original Philox (particle ID, absolute step, seed)
stream and exact F0 ratio. Strang/Trotter ordering matches the supplied solvers.
The centered carrier integral is `ds*sinc(omega*ds/(2*pi))`, including omega=0.
The same coupling appears in force and current, permitting reciprocal energy
exchange for evolving, undamped modes in the continuous reduced equations.
Prescribed modes provide external forcing. Finite marker noise, collisions,
damping, and timestep error must be considered in energy tests.

## Mode configuration

Each dictionary in `modes` accepts:

| Key | Meaning | Default |
| --- | --- | --- |
| k | Positive integer spatial harmonic on this domain | 1 |
| omega | Finite signed carrier frequency, including zero | 1 |
| coupling | Signed force/current coupling g | 1 |
| gamma_d | Nonnegative envelope damping | Top-level legacy damping |
| E_c, E_s | Initial envelope quadratures | Top-level values for primary, zero for others |
| evolve | true: evolve current and damping; false: constant prescribed envelope | true |
| name | Diagnostic/restart label | empty string |

`secondary_modes.ModeSpec` is also accepted by the Python API; its dataclass
defaults are explicit (including gamma_d=0), rather than inherited from the
top-level input. Unknown mode keys fail. Full-f/linear mode remains unsupported
because the parent full-collision scheme requires nonlinear delta-f.

```bash
NUMBA_NUM_THREADS=4 python core_collision_full.py --backend cpu \
  --params secondary_resonance_example.json --output two_modes.npz

# One CUDA device per MPI rank, within a suitable allocation:
srun -n 4 --gpus-per-task=1 python core_collision_full_gpu.py --backend gpu \
  --expected-gpus 4 --params secondary_resonance_example.json --output two_modes_gpu.npz

NUMBA_NUM_THREADS=4 python secondary_resonance_scan.py \
  --params secondary_resonance_example.json --output-dir comparison \
  --warmup-steps 100 --secondary-omega 2.2 --secondary-omega 2.004
```

The example is a short functionality experiment, not a converged chirping or
secondary-island benchmark. Its carrier resonances are v=1 and v=1.01.
The scan runs baseline (secondary coupling=0), prescribed secondary, evolving
secondary, and optional prescribed detunings. All branches share the same
particle checkpoint, primary field, collision seed, and absolute RNG step.
It saves exact per-case inputs, NPZ outputs, metrics, and `comparison.png` in a
new directory. `n_steps` applies to each branch after `warmup-steps`.
Use `--backend gpu` under MPI for the same sequential experiments across GPUs.

## Outputs, diagnostics, and restart

Multimode E_c/E_s have shape (M,), and histories have shape (steps,M).
`time` is the **post-step** time for each field-history sample (the scalar
legacy path retains its original pre-step labels). An absolute step clock makes
CPU continuation independent of chunk boundaries. Output includes:

- Carrier resonances `v_res=omega/k`, unwrapped phase, instantaneous frequency,
  isolated bounce frequency `sqrt(abs(k*g)*|E|)`, and separatrix velocity
  **half-width** `2*omega_b/k`.
- Pair indices and overlap `(width_a+width_b)/abs(v_res_a-v_res_b)` based on
  carrier centers. Coincident centers or a zero-force member produce NaN;
  phase frequency is NaN at
  zero amplitude or with fewer than two samples. Phase derivatives need temporal
  resolution and can be noisy near zero amplitude.
- Final weighted histograms `F0`, `delta_f`, `f`, and `v_edges`. They use
  `histogram(weights)/N/bin_width`, not histogram normalization that would
  erase the sign/integral of delta f. Under collisions w/p is a stochastic
  marker diagnostic, not a unique pointwise distribution. Default bins cover
  ub +/- 6 vb; `velocity_bins` may supply explicit edges. Particles outside
  the range are excluded from the histograms.
- Wave energy history sum(|E_a|^2)/2, final particle energy estimated with
  (nb/ne)*(p+w)*v^2/2, mode/physics metadata, and RNG state.

CPU saves atomic full checkpoints; GPU saves generation-tagged particle shards
and an atomic manifest. Restart with `--restart <checkpoint-or-manifest>`.
GPU manifests require the same rank count and matching shards. CPU checkpoints
can restart on GPU; loading a GPU manifest on CPU is not implemented.

Strict continuation validates mode configuration, distribution, collisions,
timestep, splitting, and particle count, then restores all field amplitudes.
Initial amplitudes in the input do not override restored amplitudes.
To deliberately change secondary waves use `restart_mode_policy: "branch"`:
it preserves particle/RNG state and the primary amplitude, while using the new
input amplitudes for all secondary modes. Primary k, omega, coupling must match.
Legacy scalar full-collision checkpoints require this explicit branch policy;
their physics provenance cannot be checked because they lack that metadata.

The CUDA kernels keep particles on device; only small mode arrays and current
reductions cross the host boundary per step. No modes-by-particles phase matrix
is allocated. Snapshots transfer particles and compute distribution diagnostics.
CUDA atomic reductions need not be bitwise reproducible on real hardware.

## Verification

For a two-node/eight-GPU interactive allocation, production-duration estimates,
and continuation commands, see [PERLMUTTER_8GPU.md](PERLMUTTER_8GPU.md) and
`jobscript_perlmutter_interactive_8gpu.sh`. The scan writes periodic per-case
checkpoints when `checkpoint_interval` is positive.

```bash
NUMBA_NUM_THREADS=2 python -m pytest -q test_secondary_resonance.py
NUMBA_NUM_THREADS=2 NUMBA_ENABLE_CUDASIM=1 python -m pytest -q \
  test_secondary_resonance.py -k gpu
```

Tests cover scalar parity for both splittings with/without collisions,
zero-coupling invariance, exact CPU restart, invalid-mode/restart rejection,
analytic combined impulses and current signs, exact weight transfer,
prescribed forcing, actual secondary effects on primary evolution, damping,
distribution normalization, and convergence against an independently integrated
nonlinear pendulum. A closed-system limit checks second-order energy-error
convergence, and fixed CPU current reductions are tested across thread counts.
CUDA simulator tests cover CPU parity, restart, CPU-to-GPU
restart, and shard-generation rejection. Simulator success does not establish
real CUDA compilation, multi-rank MPI correctness, or performance.

For physics conclusions, scan dt at a fixed physical endpoint, particle count,
and random seed; extend duration to the nonlinear regime. Measure mode growth
and trapped-orbit frequencies against the chosen dispersion/orbit model. The
overlap estimate alone is not a convergence or chaos criterion. The physical
motivation for coupled bump-on-tail experiments is discussed in Berk et al.,
[Nonlinear response of driven systems in weak turbulence theory](https://digital.library.unt.edu/ark:/67531/metadc669068/).
