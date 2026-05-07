#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=48:00:00
#SBATCH --signal=B:TERM@120
#SBATCH -J hcs_ng_paper
#SBATCH --array=0-999%64
#SBATCH -o logs/hcs_ng_%A_%a.out
#SBATCH -e logs/hcs_ng_%A_%a.err

# ============================================================
# DSMC_0D — scaled-HCS non-Gaussian paper campaign
# Scalar speed/energy diagnostics, one realization per Slurm task.
#
# Grid:
#   AR     = 1.0, 1.1, 1.5, 2.0, 2.5, 3.0
#   alpha  = 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
#   seeds  = 100 fixed seeds
#
# Total tasks: 6 * 8 * 100 = 4800
# Default concurrency: %64 keeps per-task throughput predictable.
#
# Submit full campaign from project root with:
#   bash hpc/submit_hcs_ng_campaign.sh
#
# Negishi rejects
# array indices above the site array limit, so TASK_OFFSET maps each chunk back
# onto the global 0..4799 campaign task id.  If submitting manually, keep
# chunks dependent or submit one at a time:
#   TASK_OFFSET=0    sbatch --array=0-999%64 hpc/job_hcs_ng_paper_array.slurm
#   TASK_OFFSET=1000 sbatch --array=0-999%64 hpc/job_hcs_ng_paper_array.slurm
#   TASK_OFFSET=2000 sbatch --array=0-999%64 hpc/job_hcs_ng_paper_array.slurm
#   TASK_OFFSET=3000 sbatch --array=0-999%64 hpc/job_hcs_ng_paper_array.slurm
#   TASK_OFFSET=4000 sbatch --array=0-799%64 hpc/job_hcs_ng_paper_array.slurm
#
# Pilot examples:
#   OUTPUT_ROOT=runs/hcs_ng_pilot_500s T_END_OVERRIDE=500 DT_OVERRIDE=0.05 \
#       DOMAIN_OVERRIDE=100,100,100 sbatch --array=0-2%3 hpc/job_hcs_ng_paper_array.slurm
#   T_END_OVERRIDE=0.2 DOMAIN_OVERRIDE=20,20,20 START_TAU_OVERRIDE=0.0 \
#       sbatch --array=0-3%4 hpc/job_hcs_ng_paper_array.slurm
#   DT_OVERRIDE=0.01 SAMPLE_START_TAU_OVERRIDE=50 SAMPLE_END_TAU_OVERRIDE=150 \
#       sbatch --array=0-2%3 hpc/job_hcs_ng_paper_array.slurm
#   HCS_RESCALE_OVERRIDE=0 sbatch --array=0-3%4 hpc/job_hcs_ng_paper_array.slurm
#   sbatch --array=900-914%15 hpc/job_hcs_ng_paper_array.slurm
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
CONFIG="${CONFIG:-config/default.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RCAC_SCRATCH:-${SCRATCH:-runs}}/hcs_ng_long_window}"
mkdir -p logs "${OUTPUT_ROOT}"

if [[ -n "${TMPDIR:-}" && -d "${TMPDIR}" ]]; then
  LOCAL_WORKDIR="${LOCAL_WORKDIR:-${TMPDIR}/hcs_ng_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
else
  LOCAL_WORKDIR="${LOCAL_WORKDIR:-/tmp/${USER}_hcs_ng_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
fi
mkdir -p "${LOCAL_WORKDIR}"
cleanup_local_workdir() {
  if [[ -n "${LOCAL_WORKDIR}" && "${LOCAL_WORKDIR}" != "/" ]]; then
    rm -rf -- "${LOCAL_WORKDIR}"
  fi
}
trap cleanup_local_workdir EXIT

# Prevent thread oversubscription inside each one-core realization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=131072
export PYTHONUNBUFFERED=1

# ---- Python environment bootstrap (Conda on Negishi) ----
CONDA_BASE="/apps/external/apps/conda/2025.09"
CONDA_ENV_PATH="/home/${USER}/.conda/envs/dsmc_0d"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -d "${CONDA_ENV_PATH}" ]]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV_PATH}"
  echo "Create it once on the login node:"
  echo "  ${CONDA_BASE}/bin/conda create -y -p ${CONDA_ENV_PATH} python=3.11"
  echo "  ${CONDA_BASE}/bin/conda run -p ${CONDA_ENV_PATH} pip install numpy scipy matplotlib pyyaml scikit-learn"
  exit 1
fi

conda activate "${CONDA_ENV_PATH}"
echo "Python: $(command -v python)"
python --version

ARS=(1.0 1.1 1.5 2.0 2.5 3.0)
ALPHAS=(0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
read -r -a SEEDS <<< "$(python run_hcs_ng_realization.py --print-seeds)"

N_ALPHA="${#ALPHAS[@]}"
N_SEED="${#SEEDS[@]}"
TASK_OFFSET="${TASK_OFFSET:-0}"
TASK_ID=$(( SLURM_ARRAY_TASK_ID + TASK_OFFSET ))
TOTAL_TASKS=$(( ${#ARS[@]} * N_ALPHA * N_SEED ))

if (( TASK_ID < 0 || TASK_ID >= TOTAL_TASKS )); then
  echo "ERROR: global TASK_ID=${TASK_ID} is outside 0-$(( TOTAL_TASKS - 1 ))."
  echo "       SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}, TASK_OFFSET=${TASK_OFFSET}"
  exit 1
fi

AR_INDEX=$(( TASK_ID / (N_ALPHA * N_SEED) ))
REM=$(( TASK_ID % (N_ALPHA * N_SEED) ))
ALPHA_INDEX=$(( REM / N_SEED ))
SEED_INDEX=$(( REM % N_SEED ))

AR="${ARS[$AR_INDEX]}"
ALPHA="${ALPHAS[$ALPHA_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
REALIZATION_INDEX=$(( SEED_INDEX + 1 ))

echo "Task ${TASK_ID}: AR=${AR}, alpha=${ALPHA}, seed=${SEED}, R=${REALIZATION_INDEX}"
echo "Array task ${SLURM_ARRAY_TASK_ID} with TASK_OFFSET=${TASK_OFFSET}"

EXTRA_ARGS=()
if [[ -n "${DOMAIN_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--domain "${DOMAIN_OVERRIDE}")
fi
if [[ -n "${T_END_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--t-end "${T_END_OVERRIDE}")
fi
if [[ -n "${TAU_END_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--tau-end "${TAU_END_OVERRIDE}")
fi
if [[ -n "${DT_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--dt "${DT_OVERRIDE}")
fi
if [[ -n "${START_TAU_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--sample-start-tau "${START_TAU_OVERRIDE}")
fi
if [[ -n "${SAMPLE_START_TAU_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--sample-start-tau "${SAMPLE_START_TAU_OVERRIDE}")
fi
if [[ -n "${SAMPLE_END_TAU_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--sample-end-tau "${SAMPLE_END_TAU_OVERRIDE}")
fi
if [[ -n "${SAMPLE_DELTA_TAU_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--sample-delta-tau "${SAMPLE_DELTA_TAU_OVERRIDE}")
fi
if [[ "${HCS_RESCALE_OVERRIDE:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-hcs-rescale)
fi
if [[ -n "${MEMORY_DIAGNOSTICS_INTERVAL_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--memory-diagnostics-interval-steps "${MEMORY_DIAGNOSTICS_INTERVAL_STEPS}")
fi

python -u run_hcs_ng_realization.py \
  --config "${CONFIG}" \
  --AR "${AR}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --realization-index "${REALIZATION_INDEX}" \
  --output-root "${OUTPUT_ROOT}" \
  --local-workdir "${LOCAL_WORKDIR}" \
  --skip-manifest \
  "${EXTRA_ARGS[@]}"
