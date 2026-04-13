#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -J dsmc_ar2_eta
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

# Move to project root (directory where this script was submitted from).
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

# Prevent thread oversubscription inside multiprocessing workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# ---- Python environment bootstrap (Conda on Negishi) ----
# One-time setup command (run on login node before first job):
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
  echo "Create it once on login node, then resubmit:"
  echo "  ${CONDA_BASE}/bin/conda create -y -p ${CONDA_ENV_PATH} python=3.11"
  echo "  ${CONDA_BASE}/bin/conda run -p ${CONDA_ENV_PATH} pip install numpy scipy matplotlib pyyaml scikit-learn"
  exit 1
fi

conda activate "${CONDA_ENV_PATH}"
PYTHON_CMD="$(command -v python)"
echo "Using python: ${PYTHON_CMD}"
python --version

# Option A: run all active alphas from config/default.yaml
"${PYTHON_CMD}" run_alpha_sweep.py --config config/default.yaml --workers 8

# Option B: run only selected alphas (example)
# "${PYTHON_CMD}" run_alpha_sweep.py --config config/default.yaml --workers 8 --alphas 0.8,0.85,1.0
