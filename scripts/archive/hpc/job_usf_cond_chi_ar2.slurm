#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_ar2_usf_cond_chi
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D  — AR=2 USF sweep, conditional chi scattering model
#
# Scattering model: p(chi | mu, alpha, AR) — conditional Beta
# with mu = |eij . ghat| from the NTC-accepted geometry.
# Direction: mu_plane_post_relative (g' in (g, eij) plane).
#
# Grid: 10 alpha values x 4 seeds = 40 realizations
# Workers: 20 (runs 2 batches of 20 simultaneously)
#
# Submit from project root:
#   sbatch hpc/job_usf_cond_chi_ar2.slurm
#
# Pilot (2 alpha, 1 seed each, short run):
#   T_END_OVERRIDE=50 sbatch --export=ALL,T_END_OVERRIDE=50 \
#       hpc/job_usf_cond_chi_ar2.slurm
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
    --config config/usf_sweep_cond_chi.yaml \
    --workers "${WORKERS}" \
    --skip-post
