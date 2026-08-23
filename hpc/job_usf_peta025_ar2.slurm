#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_ar2_usf_peta025
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D — AR=2 USF sweep with p_eta = 0.25
#
# Angular model:
#   stress_weight with angular_transport_probability_override = 0.25
#   Full angular scatter: probability p_eta
#   Scalar-only branch:   probability 1 - p_eta
#
# Grid: 10 alpha values x 4 seeds = 40 realizations
# Workers: 20 (runs 2 batches of 20 simultaneously)
#
# Submit from project root:
#   sbatch hpc/job_usf_peta025_ar2.slurm
#
# Override worker count:
#   sbatch --export=ALL,WORKERS=10 hpc/job_usf_peta025_ar2.slurm
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

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
echo "Using python: $(command -v python)"
python --version

WORKERS="${WORKERS:-20}"

python -m scripts.public.run_alpha_sweep \
    --config config/usf_sweep_peta025.yaml \
    --workers "${WORKERS}" \
    --skip-post
