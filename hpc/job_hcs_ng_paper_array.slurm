#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=48:00:00
#SBATCH -J hcs_ng_paper
#SBATCH --array=0-2799%256
#SBATCH -o logs/hcs_ng_%A_%a.out
#SBATCH -e logs/hcs_ng_%A_%a.err

# ============================================================
# DSMC_0D — HCS non-Gaussian paper campaign
#
# Grid:
#   AR     = 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0
#   alpha  = 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
#   seeds  = 50 fixed seeds
#
# Total tasks: 7 * 8 * 50 = 2800
# Concurrency: %256 uses up to 256 cores at once.
#
# Submit full campaign from project root:
#   sbatch hpc/job_hcs_ng_paper_array.slurm
#
# Pilot examples:
#   T_END_OVERRIDE=0.2 DOMAIN_OVERRIDE=20,20,20 START_TAU_OVERRIDE=0.0 \
#       sbatch --array=0-3%4 hpc/job_hcs_ng_paper_array.slurm
#   sbatch --array=900-914%15 hpc/job_hcs_ng_paper_array.slurm
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/hcs_ng_paper}"
mkdir -p logs "${OUTPUT_ROOT}"

# Prevent thread oversubscription inside each one-core realization.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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

ARS=(1.0 1.1 1.25 1.5 2.0 2.5 3.0)
ALPHAS=(0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
read -r -a SEEDS <<< "$(python run_hcs_ng_realization.py --print-seeds)"

N_ALPHA="${#ALPHAS[@]}"
N_SEED="${#SEEDS[@]}"
TASK_ID="${SLURM_ARRAY_TASK_ID}"

AR_INDEX=$(( TASK_ID / (N_ALPHA * N_SEED) ))
REM=$(( TASK_ID % (N_ALPHA * N_SEED) ))
ALPHA_INDEX=$(( REM / N_SEED ))
SEED_INDEX=$(( REM % N_SEED ))

if (( AR_INDEX >= ${#ARS[@]} )); then
  echo "ERROR: SLURM_ARRAY_TASK_ID=${TASK_ID} is outside the campaign grid."
  exit 1
fi

AR="${ARS[$AR_INDEX]}"
ALPHA="${ALPHAS[$ALPHA_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
REALIZATION_INDEX=$(( SEED_INDEX + 1 ))

echo "Task ${TASK_ID}: AR=${AR}, alpha=${ALPHA}, seed=${SEED}, R=${REALIZATION_INDEX}"

EXTRA_ARGS=()
if [[ -n "${DOMAIN_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--domain "${DOMAIN_OVERRIDE}")
fi
if [[ -n "${T_END_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--t-end "${T_END_OVERRIDE}")
fi
if [[ -n "${START_TAU_OVERRIDE:-}" ]]; then
  EXTRA_ARGS+=(--start-tau "${START_TAU_OVERRIDE}")
fi

python run_hcs_ng_realization.py \
  --AR "${AR}" \
  --alpha "${ALPHA}" \
  --seed "${SEED}" \
  --realization-index "${REALIZATION_INDEX}" \
  --output-root "${OUTPUT_ROOT}" \
  "${EXTRA_ARGS[@]}"
