#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=44
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH -J hcs_ng_sweep
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D — one-AR HCS non-Gaussian sweep
#
# One Slurm job runs all alpha/seed realizations for one AR using a local
# ProcessPoolExecutor.  Edit hcs_ng_sweep.AR in config/hcs_ng_sweep.yaml, or
# override it at submission:
#
#   sbatch --export=ALL,HCS_AR=2.0,WORKERS=44 hpc/job_hcs_ng_sweep_ar.slurm
#
# Pilot:
#   HCS_AR=1.0 WORKERS=4 ALPHAS=0.60 SEEDS=42,123 T_END=120 \
#       SAMPLE_START_TAU=5 SAMPLE_END_TAU=10 \
#       sbatch --export=ALL hpc/job_hcs_ng_sweep_ar.slurm
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

CONFIG="${CONFIG:-config/hcs_ng_sweep.yaml}"
WORKERS="${WORKERS:-44}"

EXTRA_ARGS=()
if [[ -n "${HCS_AR:-}" ]]; then
  EXTRA_ARGS+=(--AR "${HCS_AR}")
fi
if [[ -n "${ALPHAS:-}" ]]; then
  EXTRA_ARGS+=(--alphas "${ALPHAS}")
fi
if [[ -n "${SEEDS:-}" ]]; then
  EXTRA_ARGS+=(--seeds "${SEEDS}")
fi
if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--output-root "${OUTPUT_ROOT}")
fi
if [[ -n "${DOMAIN:-}" ]]; then
  EXTRA_ARGS+=(--domain "${DOMAIN}")
fi
if [[ -n "${DT:-}" ]]; then
  EXTRA_ARGS+=(--dt "${DT}")
fi
if [[ -n "${T_END:-}" ]]; then
  EXTRA_ARGS+=(--t-end "${T_END}")
fi
if [[ -n "${TAU_END:-}" ]]; then
  EXTRA_ARGS+=(--tau-end "${TAU_END}")
fi
if [[ -n "${SAMPLE_START_TAU:-}" ]]; then
  EXTRA_ARGS+=(--sample-start-tau "${SAMPLE_START_TAU}")
fi
if [[ -n "${SAMPLE_END_TAU:-}" ]]; then
  EXTRA_ARGS+=(--sample-end-tau "${SAMPLE_END_TAU}")
fi
if [[ -n "${SAMPLE_DELTA_TAU:-}" ]]; then
  EXTRA_ARGS+=(--sample-delta-tau "${SAMPLE_DELTA_TAU}")
fi
if [[ "${HCS_RESCALE:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-hcs-rescale)
fi

python -m scripts.internal.run_hcs_ng_sweep \
  --config "${CONFIG}" \
  --workers "${WORKERS}" \
  "${EXTRA_ARGS[@]}"
