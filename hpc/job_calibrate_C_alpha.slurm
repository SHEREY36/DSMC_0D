#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH -J calib_C_alpha
#SBATCH --array=0-9
#SBATCH -o logs/calib_C_alpha_%A_%a.out
#SBATCH -e logs/calib_C_alpha_%A_%a.err

# ============================================================
# SLURM array job: calibrate C(alpha) for a given AR.
# One task per alpha — all 10 run simultaneously.
#
# ---- To change AR: edit the single line below ----
AR="2.0"
# --------------------------------------------------
#
# Array index -> alpha mapping (0-indexed):
#   0=0.50  1=0.55  2=0.60  3=0.65  4=0.70
#   5=0.75  6=0.80  7=0.85  8=0.90  9=0.95
#
# Submit from the project root:
#   cd /path/to/DSMC_0D
#   sbatch hpc/job_calibrate_C_alpha.slurm
#
# After ALL tasks finish, merge results on the login node:
#   python hpc/merge_C_alpha_results.py --AR 2.0
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs/calib_C_alpha

# Prevent thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Python environment (Conda on Negishi) ----
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
  echo "  conda activate ${CONDA_ENV_PATH}"
  echo "  pip install numpy scipy matplotlib pyyaml scikit-learn"
  exit 1
fi

conda activate "${CONDA_ENV_PATH}"
echo "Python: $(which python) $(python --version)"

# ---- Derive AR integer (e.g. 2.0 -> 20, 1.5 -> 15) ----
AR_INT=$(python -c "print(f'{round(float(\"${AR}\")*10):02d}')")

# ---- Alpha lookup ----
ALPHAS=(0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95)
ALPHA="${ALPHAS[${SLURM_ARRAY_TASK_ID}]}"
ALPHA_INT=$(python -c "print(f'{round(float(\"${ALPHA}\")*100):03d}')")

echo "Task ${SLURM_ARRAY_TASK_ID}: AR=${AR}, alpha=${ALPHA}"

# NOTE: Before submitting this job array, run once on the login node:
#   python extract_theta_target.py --config config/default.yaml --AR ${AR}
# This pre-extracts theta* values from LAMMPS data into:
#   models/theta_target_table_AR${AR_INT}.json
THETA_TABLE="models/theta_target_table_AR${AR_INT}.json"

# Each task writes to its own JSON to avoid race conditions.
# merge_C_alpha_results.py --AR ${AR} combines them afterward.
OUT_FILE="runs/calib_C_alpha/C_alpha_AR${AR_INT}_${ALPHA_INT}.json"
TMP_DIR="runs/calib_C_alpha/tmp_AR${AR_INT}_${ALPHA_INT}"

python calibrate_C_alpha.py \
    --config      config/default.yaml \
    --AR          "${AR}" \
    --alphas      "${ALPHA}" \
    --force \
    --theta-table "${THETA_TABLE}" \
    --output      "${OUT_FILE}" \
    --output-dir  "${TMP_DIR}"

echo "Task ${SLURM_ARRAY_TASK_ID} done. Result: ${OUT_FILE}"
