#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_ar1_usf
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D  — AR=1 (sphere) USF sweep (per-alpha shear rate)
#
# Parallelism: workers = n_alpha × n_seeds_per_batch
#   Current config: 11 alpha values × 4 seeds = 44 total tasks
#   WORKERS=20  → 20 simultaneous (alpha,seed) pairs  (~3 batches)
#   WORKERS=44  → all 44 run at once (set cpus-per-task=44, mem=96G)
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Prevent thread oversubscription inside multiprocessing workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Python environment bootstrap (Conda on Negishi) ----
# One-time setup (run on login node before first job):
#   /apps/external/apps/conda/2025.09/bin/conda create -y -p /home/${USER}/.conda/envs/dsmc_0d python=3.11
#   /apps/external/apps/conda/2025.09/bin/conda run -p /home/${USER}/.conda/envs/dsmc_0d \
#       pip install numpy scipy matplotlib pyyaml scikit-learn

CONDA_BASE="/apps/external/apps/conda/2025.09"
CONDA_ENV_PATH="/home/${USER}/.conda/envs/dsmc_0d"

if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh"
  exit 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if [[ ! -d "${CONDA_ENV_PATH}" ]]; then
  echo "ERROR: Conda env not found at ${CONDA_ENV_PATH}"
  echo "Create it once on the login node, then resubmit:"
  echo "  ${CONDA_BASE}/bin/conda create -y -p ${CONDA_ENV_PATH} python=3.11"
  echo "  ${CONDA_BASE}/bin/conda run -p ${CONDA_ENV_PATH} pip install numpy scipy matplotlib pyyaml scikit-learn"
  exit 1
fi

conda activate "${CONDA_ENV_PATH}"
echo "Using python: $(command -v python)"
python --version

# Run the sphere USF sweep — 20 workers = ~20 simultaneous (alpha,seed) pairs
python run_alpha_sweep.py \
    --config config/sphere_usf_sweep.yaml \
    --workers 20 \
    --skip-post
