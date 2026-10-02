#!/bin/bash
# First allocate on Perlmutter (replace mXXXX_g with your GPU project account):
# salloc --nodes=2 --qos=interactive --time=04:00:00 --constraint=gpu \
#   --ntasks-per-node=4 --gpus-per-task=1 --cpus-per-task=32 \
#   --signal=USR1@300 --account=mXXXX_g
# Then: bash jobscript_perlmutter_interactive_8gpu.sh [params.json]
# Defaults: run the three matched cases, each on all 8 GPUs, at input resolution.
# Overrides: N_STEPS, N_PARTICLES, RUN_DIR, WARMUP_STEPS, CHECKPOINT_INTERVAL,
#            WISP_CONDA_ENV, WISP_PYTHON. RUN_MODE=single supports RESTART_FILE.
set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: run inside a two-node GPU salloc allocation (see script header)." >&2
    exit 2
fi
if [[ "${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-0}}" != 2 ]]; then
    echo "ERROR: this launcher requires exactly two allocated nodes." >&2
    exit 2
fi
if (( $# > 1 )); then
    echo "Usage: bash $0 [params.json]" >&2
    exit 2
fi
WISP_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export WISP_PARAMS_INPUT="${1:-${PARAMS_FILE:-${WISP_SCRIPT_DIR}/secondary_resonance_example.json}}"
export RUN_MODE="${RUN_MODE:-scan}"
export RUN_DIR="${RUN_DIR:-${PWD}/second_resonance_8gpu_${SLURM_JOB_ID}_$(date +%Y%m%d_%H%M%S)}"
export CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-300}"
export WARMUP_STEPS="${WARMUP_STEPS:-0}"
export N_STEPS="${N_STEPS:-}"
export N_PARTICLES="${N_PARTICLES:-}"
export RESTART_FILE="${RESTART_FILE:-}"

module load PrgEnv-gnu cray-mpich cudatoolkit python
if [[ -z "${WISP_PYTHON:-}" ]]; then
    conda activate "${WISP_CONDA_ENV:-wisp_proto_env}"
    WISP_PYTHON="${CONDA_PREFIX}/bin/python"
fi
[[ -x "${WISP_PYTHON}" ]] || { echo "ERROR: Python is not executable: ${WISP_PYTHON}" >&2; exit 2; }
# Preserve invocation-relative input/output paths, then make all paths absolute.
export WISP_PARAMS_INPUT RUN_DIR RESTART_FILE
WISP_PARAMS_INPUT="$("${WISP_PYTHON}" -c 'import os; print(os.path.abspath(os.environ["WISP_PARAMS_INPUT"]))')"
RUN_DIR="$("${WISP_PYTHON}" -c 'import os; print(os.path.abspath(os.environ["RUN_DIR"]))')"
if [[ -n "${RESTART_FILE}" ]]; then
    RESTART_FILE="$("${WISP_PYTHON}" -c 'import os; print(os.path.abspath(os.environ["RESTART_FILE"]))')"
fi
cd "${WISP_SCRIPT_DIR}"
unset NUMBA_ENABLE_CUDASIM WISP_GPU_SERIAL
export MPICH_GPU_SUPPORT_ENABLED=0  # MPI reductions use host arrays.
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR="/tmp/wisp-numba-${USER}-${SLURM_JOB_ID}"
export MPLCONFIGDIR="/tmp/wisp-mpl-${USER}-${SLURM_JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}" "${MPLCONFIGDIR}"

# Write a new effective input; never modify the supplied JSON.
"${WISP_PYTHON}" - <<'PY'
import hashlib
import json
import math
import os
from pathlib import Path
from secondary_runner import _configuration

mode = os.environ['RUN_MODE']
if mode not in ('scan', 'single'):
    raise SystemExit('RUN_MODE must be scan or single')
source = Path(os.environ['WISP_PARAMS_INPUT'])
params = json.loads(source.read_text())
for env, key in [('N_STEPS', 'n_steps'), ('N_PARTICLES', 'n_particles')]:
    if os.environ[env]:
        params[key] = int(os.environ[env])
warmup = int(os.environ['WARMUP_STEPS'])
if warmup < 0:
    raise SystemExit('WARMUP_STEPS must be nonnegative')
interval = float(os.environ['CHECKPOINT_INTERVAL'])
if not math.isfinite(interval) or interval < 0:
    raise SystemExit('CHECKPOINT_INTERVAL must be finite and nonnegative')
params['checkpoint_interval'] = interval
params['expected_gpus'] = 8
params.pop('checkpoint_file', None)
restart = os.environ['RESTART_FILE'] or params.get('restart_file')
if mode == 'scan' and (restart or len(params.get('modes', [])) != 2):
    raise SystemExit('Scan needs exactly two modes and no restart. For continuation use RUN_MODE=single.')
if mode == 'single' and restart:
    restart = str(Path(restart).resolve())
    if not Path(restart).is_file():
        raise SystemExit(f'Restart manifest not found: {restart}')
    params['restart_file'] = restart
    params['restart_mode_policy'] = 'strict'  # Do not reseed secondary fields on continuation.
    params.pop('t_start', None)
_configuration(params)
destination = Path(os.environ['RUN_DIR'])
destination.mkdir(parents=True, exist_ok=False)
(destination / 'params.effective.json').write_text(json.dumps(params, indent=2) + '\n')
provenance = dict(input_path=str(source), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  job_id=os.environ['SLURM_JOB_ID'], run_mode=mode, warmup_steps=warmup,
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in Path('.').glob('*.py')})
(destination / 'launch.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(f'Run directory: {destination}')
print(f'Per case: {params["n_steps"]} steps, dt={params["dt"]}, {params["n_particles"]} GLOBAL particles')
print('Three sequential eight-GPU cases' if mode == 'scan' else 'One eight-GPU case')
PY

SRUN_ARGS=(--nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=32
           --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=1)
# Create rank-local caches on BOTH nodes, then verify MPI, placement, and CUDA.
srun "${SRUN_ARGS[@]}" bash -c \
    'mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"; exec "$@"' \
    bash "${WISP_PYTHON}" perlmutter_8gpu_preflight.py 2>&1 | tee "${RUN_DIR}/preflight.log"

if [[ "${RUN_MODE}" == scan ]]; then
    srun "${SRUN_ARGS[@]}" "${WISP_PYTHON}" secondary_resonance_scan.py \
        --backend gpu --params "${RUN_DIR}/params.effective.json" \
        --warmup-steps "${WARMUP_STEPS}" --output-dir "${RUN_DIR}/comparison" \
        2>&1 | tee "${RUN_DIR}/run.log"
else
    srun "${SRUN_ARGS[@]}" "${WISP_PYTHON}" core_collision_full_gpu.py \
        --backend gpu --expected-gpus 8 --params "${RUN_DIR}/params.effective.json" \
        --output "${RUN_DIR}/output.npz" --checkpoint "${RUN_DIR}/checkpoint.npz" \
        2>&1 | tee "${RUN_DIR}/run.log"
fi
echo "Outputs and logs: ${RUN_DIR}"
