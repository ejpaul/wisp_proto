# Two-node interactive Perlmutter run

From a Perlmutter login shell, request two GPU nodes. Replace the example
account with your NERSC GPU account:

```bash
salloc --nodes=2 --qos=interactive --time=04:00:00 --constraint=gpu \
  --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32 \
  --signal=USR1@300 --account=mXXXX_g
```

Inside the allocation, change to your Perlmutter checkout's
`examples/nonlinear_delta_f_collision/second_resonance` directory and run:

```bash
bash jobscript_perlmutter_interactive_8gpu.sh
```

This uses the example unchanged: 1,000 steps and 20,000 **global** particles.
It runs baseline, prescribed-secondary, and self-consistent-secondary cases
sequentially, each with eight ranks on eight GPUs. This small configuration is
a functionality check, not an efficient eight-GPU workload.

The launcher loads the same modules as the existing four-GPU launcher and
activates `wisp_proto_env`. Set `WISP_CONDA_ENV` to another environment name,
or `WISP_PYTHON` to an existing compatible Python executable to skip activation.
The environment must already contain WISP, NumPy, Numba/CUDA, Matplotlib,
and mpi4py linked to a Perlmutter-compatible MPI. The preflight prints the
actual Python/MPI provenance, checks two hosts with four distinct GPUs each,
compiles/runs a CUDA drift kernel, and verifies a cross-rank host reduction.
The first simulation still has to compile the other CUDA kernels.

Custom input and explicit resolution overrides:

```bash
# First longer exploratory run; N_PARTICLES is a GLOBAL total.
N_STEPS=100000 N_PARTICLES=1000000 \
  RUN_DIR="$SCRATCH/second_resonance/exploratory_8gpu" \
  bash jobscript_perlmutter_interactive_8gpu.sh secondary_resonance_example.json

# Candidate production duration, once the short run and resolution checks pass.
N_STEPS=500000 N_PARTICLES=1000000 \
  RUN_DIR="$SCRATCH/second_resonance/production_candidate_8gpu" \
  bash jobscript_perlmutter_interactive_8gpu.sh secondary_resonance_example.json
```

One million particles is an initial resolution candidate, not a demonstrated
requirement. Compare at least two particle counts and multiple random seeds.
No parameters are silently changed: only specified environment overrides are
applied, and the exact input is saved as `params.effective.json`. Existing run
directories are rejected. `launch.json` records input/source hashes and job ID;
`preflight.log` and `run.log` capture execution. Optional `WARMUP_STEPS` defaults
to zero; a nonzero value evolves the common primary before branching.

Periodic per-case checkpoints default to 300 seconds; override with
`CHECKPOINT_INTERVAL`. The scanner now saves `warmup_checkpoint.npz` and
`<case>_checkpoint.npz` with rank shards, in addition to final outputs.
SIGUSR1/SIGTERM requests stop after the current timestep and save a checkpoint.
The five-minute Slurm warning and periodic checkpoints reduce lost work, but
a large checkpoint may take longer than the warning interval.

To continue one interrupted case, select its saved JSON and checkpoint manifest:

```bash
RUN_MODE=single N_STEPS=200000 \
  RESTART_FILE="$SCRATCH/second_resonance/production_candidate_8gpu/comparison/prescribed_checkpoint.npz" \
  RUN_DIR="$SCRATCH/second_resonance/prescribed_continuation_8gpu" \
  bash jobscript_perlmutter_interactive_8gpu.sh \
  "$SCRATCH/second_resonance/production_candidate_8gpu/comparison/prescribed.json"
```

Here 200,000 means **additional** steps, not a target cumulative step count;
replace it with the remaining work. Pass the manifest, not a rank shard, and
retain all eight matching shards. Single-mode launcher operation means *one
case*, not one wave. It forces strict continuation so secondary amplitudes are
restored rather than reset by the original scan's branch policy. Outputs and
checkpoints go to the new run directory. The supplied JSON remains unchanged.
Use `RUN_MODE=single` without a restart to run only the supplied two-mode case.

## Choosing the number of steps

For the current JSON, dt=0.1, gamma_L=0.00195614799652 and primary
gamma_d=0.00176053319687. The isolated-mode small-signal reference rate is
gamma_net=gamma_L-gamma_d=0.000195614799652. Thus

```text
T = n_steps * dt
steps per reference net-growth e-fold = 1 / (gamma_net * dt) = 51,121
initial primary bounce period = 2*pi/sqrt(k1*g1*|E1|) = 1,986.9
initial secondary bounce period = 3,141.6
```

| Steps per case | Physical duration T | gamma_net*T | Intended use |
| --- | --- | --- | --- |
| 1,000 | 100 | 0.0196 | Functionality/compiler/MPI check |
| 100,000 | 10,000 | 1.96 | First nonlinear exploration, about five initial primary bounce periods |
| 500,000 | 50,000 | 9.78 | Candidate production duration |
| 1,000,000 | 100,000 | 19.56 | Longer evolution and duration-sensitivity comparison |

A first production-duration estimate is **500,000 to 1,000,000 steps per
case**, not a measured convergence requirement. The initial primary amplitude
already gives omega_B/gamma_L approximately 1.62, so the small-signal growth
clock is a reference, not a prediction of a long exponential startup. Mode
coupling, prescribed forcing, and trapping alter subsequent evolution. The
example has zero collisions; increasing duration alone does not establish a
collisional chirping regime. Select the endpoint from the actual observable
(redistribution, mode competition, trapping, or a demonstrated chirping cycle).

At fixed T, halving dt doubles the required step count. Compare dt=0.1 and
0.05 (e.g. 500,000 versus 1,000,000 steps for T=50,000), particle counts, and
seeds before using results for physics conclusions. For weak seeds, an onset
estimate is log(E_target/E_seed)/(gamma_net*dt), plus the desired number of
nonlinear periods; this estimate does not apply unchanged to the present
trapping-strength seed or a strongly driven secondary.

No eight-GPU throughput has been measured. Benchmark with the intended particle
count, allowing for JIT startup and checkpoint I/O; do not assume this duration
fits one four-hour allocation. The three-case scan uses three times the per-case
work, plus warmup. If measured throughput is R steps/second, its integration
time is approximately (warmup_steps + 3*n_steps)/R, before checkpoint/output
overheads. Two thousand five hundred particles per rank in the default example
will likely be dominated by launch and MPI overhead rather than GPU arithmetic.

Launch layout follows NERSC's [interactive allocation guidance](https://docs.nersc.gov/jobs/interactive/),
[GPU architecture](https://docs.nersc.gov/systems/perlmutter/architecture/), and
[CPU/GPU affinity guidance](https://docs.nersc.gov/jobs/affinity/).
Local syntax/routing tests are not evidence of a successful Perlmutter run.
