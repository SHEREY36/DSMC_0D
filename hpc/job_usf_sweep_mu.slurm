#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_ar2_usf_mu
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D  — AR=2 USF sweep with mu-conditioned scattering
#
# Uses config/usf_sweep_mu.yaml (use_mu_scattering: true) and
# the recalibrated C_alpha table models/C_alpha_table_AR20_mu.json.
#
# Prerequisite: run job_calibrate_C_alpha_mu.slurm first and
# merge results into models/C_alpha_table_AR20_mu.json.
#
# Parallelism: workers = n_alpha × n_seeds_per_batch
#   Current config: 10 alpha values × 4 seeds = 40 total tasks
#   WORKERS=20  → 20 simultaneous (alpha,seed) pairs  (2 batches of 20)
#   WORKERS=40  → all 40 run at once (set cpus-per-task=40, mem=96G)
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

# Run the USF sweep — 20 workers = 10 alphas × 2 seeds simultaneously
python run_alpha_sweep.py \
    --config config/usf_sweep_mu.yaml \
    --workers 20 \
    --skip-post
