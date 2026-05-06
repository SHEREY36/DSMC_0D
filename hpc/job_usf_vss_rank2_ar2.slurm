#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_ar2_usf_vss
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D — AR=2 USF sweep with VSS rank-2 angular transport
#
# Angular model:
#   simulation.angular_transport_model = vss_rank2
#   alpha_eff table = models/vss_alpha_eff_table_AR20_peta060.json
#   cos(chi) = 2*R^(1/alpha_eff) - 1, eps uniform in [0, 2*pi)
#
# Grid: 10 alpha values x 4 seeds = 40 realizations
# Workers: 20 (runs 2 batches of 20 simultaneously)
#
# Submit from project root:
#   sbatch hpc/job_usf_vss_rank2_ar2.slurm
#
# Override worker count:
#   sbatch --export=ALL,WORKERS=10 hpc/job_usf_vss_rank2_ar2.slurm
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

python run_alpha_sweep.py \
    --config config/usf_sweep_vss_rank2.yaml \
    --workers "${WORKERS}" \
    --skip-post
