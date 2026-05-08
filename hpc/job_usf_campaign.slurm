#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_usf_campaign
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D — parameterized USF campaign sweep
#
# Required/typical variables:
#   AR          1.5, 2.0, 2.5, or 3.0
#   MODE        current | stress_weight | vss_rank2
#   PETA        required for MODE=stress_weight
#   VSS_TABLE   required for MODE=vss_rank2
#   OUTPUT_ROOT optional; defaults by AR/MODE/PETA
#   CONFIG_OUT  optional; defaults under runs/generated_configs/
#   WORKERS     optional; default 20
#
# Examples:
#   sbatch --export=ALL,AR=1.5,MODE=current hpc/job_usf_campaign.slurm
#   sbatch --export=ALL,AR=2.5,MODE=stress_weight,PETA=0.60 hpc/job_usf_campaign.slurm
#   sbatch --export=ALL,AR=3.0,MODE=vss_rank2,VSS_TABLE=models/vss_alpha_eff_table_AR30_peta060.json hpc/job_usf_campaign.slurm
# ============================================================

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs runs/generated_configs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

CONDA_BASE="${CONDA_BASE:-/apps/external/apps/conda/2025.09}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/home/${USER}/.conda/envs/dsmc_0d}"

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

AR="${AR:-2.0}"
MODE="${MODE:-current}"
WORKERS="${WORKERS:-20}"
BASE_CONFIG="${BASE_CONFIG:-config/usf_sweep.yaml}"

PETA_ARGS=()
if [[ "${MODE}" == "stress_weight" ]]; then
  if [[ -z "${PETA:-}" ]]; then
    echo "ERROR: PETA is required when MODE=stress_weight"
    exit 1
  fi
  PETA_ARGS=(--p-eta "${PETA}")
fi

VSS_ARGS=()
if [[ "${MODE}" == "vss_rank2" ]]; then
  if [[ -z "${VSS_TABLE:-}" ]]; then
    echo "ERROR: VSS_TABLE is required when MODE=vss_rank2"
    exit 1
  fi
  VSS_ARGS=(--vss-table "${VSS_TABLE}")
fi

CONFIG_OUT="${CONFIG_OUT:-runs/generated_configs/usf_AR${AR}_${MODE}${PETA:+_peta${PETA}}.yaml}"
OUTPUT_ARGS=()
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  OUTPUT_ARGS=(--output-root "${OUTPUT_ROOT}")
fi

python prepare_usf_campaign_config.py \
  --base-config "${BASE_CONFIG}" \
  --config-out "${CONFIG_OUT}" \
  --AR "${AR}" \
  --mode "${MODE}" \
  --workers "${WORKERS}" \
  "${PETA_ARGS[@]}" \
  "${VSS_ARGS[@]}" \
  "${OUTPUT_ARGS[@]}"

python run_alpha_sweep.py \
  --config "${CONFIG_OUT}" \
  --workers "${WORKERS}" \
  --skip-post
