#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_usf_vss_probe
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D -- USF finite-difference f_tr probe sweep
#
# Default probe:
#   AR=2.0
#   FTR_PROBE_DELTA=-0.10
#   output root=runs/AR2_usf_vss_rank2_probe_m010
#
# Submit:
#   sbatch hpc/job_usf_vss_rank2_probe.slurm
#
# Reuse:
#   sbatch --export=ALL,AR=2.5,FTR_PROBE_DELTA=-0.10 hpc/job_usf_vss_rank2_probe.slurm
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
WORKERS="${WORKERS:-20}"
BASE_CONFIG="${BASE_CONFIG:-config/usf_sweep.yaml}"
FTR_PROBE_DELTA="${FTR_PROBE_DELTA:--0.10}"

AR_TAG="$(python -c 'from src.simulation.particle import model_ar_tag; import sys; print(model_ar_tag(float(sys.argv[1])))' "${AR}")"
PROBE_TAG="$(python -c 'from prepare_usf_campaign_config import probe_tag; import sys; print(probe_tag(float(sys.argv[1])))' "${FTR_PROBE_DELTA}")"
if [[ "${AR_TAG}" == "AR20" ]]; then
  OUTPUT_TAG="AR2"
else
  OUTPUT_TAG="${AR_TAG}"
fi

if [[ -z "${VSS_TABLE:-}" ]]; then
  case "${AR_TAG}" in
    AR15) VSS_TABLE="models/vss_alpha_eff_table_AR15_peta090.json" ;;
    AR20) VSS_TABLE="models/vss_alpha_eff_table_AR20_peta060.json" ;;
    AR25) VSS_TABLE="models/vss_alpha_eff_table_AR25_peta070.json" ;;
    AR30) VSS_TABLE="models/vss_alpha_eff_table_AR30_peta070.json" ;;
    *)
      echo "ERROR: No default VSS_TABLE for AR=${AR} (${AR_TAG})."
      exit 1
      ;;
  esac
fi

CONFIG_OUT="${CONFIG_OUT:-runs/generated_configs/usf_${AR_TAG}_vss_rank2_${PROBE_TAG}.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/${OUTPUT_TAG}_usf_vss_rank2_${PROBE_TAG}}"

if [[ ! -f "${VSS_TABLE}" ]]; then
  echo "ERROR: VSS_TABLE not found: ${VSS_TABLE}"
  exit 1
fi

TIME_ARGS=()
if [[ -n "${T_END_OVERRIDE:-}" ]]; then
  TIME_ARGS=(--t-end "${T_END_OVERRIDE}")
fi

ALPHA_PREP_ARGS=()
ALPHA_RUN_ARGS=()
if [[ -n "${ALPHAS:-}" ]]; then
  ALPHA_PREP_ARGS=(--alpha-values "${ALPHAS}")
  ALPHA_RUN_ARGS=(--alphas "${ALPHAS}")
fi

python prepare_usf_campaign_config.py \
  --base-config "${BASE_CONFIG}" \
  --config-out "${CONFIG_OUT}" \
  --AR "${AR}" \
  --mode vss_rank2 \
  --vss-table "${VSS_TABLE}" \
  --ftr-rank0-probe-delta "${FTR_PROBE_DELTA}" \
  --output-root "${OUTPUT_ROOT}" \
  --workers "${WORKERS}" \
  "${TIME_ARGS[@]}" \
  "${ALPHA_PREP_ARGS[@]}"

python run_alpha_sweep.py \
  --config "${CONFIG_OUT}" \
  --workers "${WORKERS}" \
  --skip-post \
  "${ALPHA_RUN_ARGS[@]}"
