#!/bin/bash -l
#SBATCH -A morri353
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH -J dsmc_usf_vss_C2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ============================================================
# DSMC_0D -- USF sweep with VSS rank-2 transport and C2 correction
#
# Defaults test AR=2.0 with:
#   VSS_TABLE=models/angular_transport/vss_alpha_eff_table_AR20_peta060.json
#   C2_TABLE=models/relaxation/C2_table_AR20.json
#
# Submit AR=2 default:
#   sbatch hpc/job_usf_vss_rank2_C2.slurm
#
# Reuse for other AR values:
#   sbatch --export=ALL,AR=2.5 hpc/job_usf_vss_rank2_C2.slurm
#   sbatch --export=ALL,AR=3.0,WORKERS=10 hpc/job_usf_vss_rank2_C2.slurm
#
# Useful overrides:
#   OUTPUT_ROOT=runs/my_sweep
#   CONFIG_OUT=runs/generated_configs/my_sweep.yaml
#   VSS_TABLE=models/angular_transport/vss_alpha_eff_table_AR25_peta070.json
#   C2_TABLE=models/relaxation/C2_table_AR25.json
#   ALPHAS=0.50,0.60,0.80
#   T_END_OVERRIDE=500
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

AR_TAG="$(python -c 'from src.simulation.particle import model_ar_tag; import sys; print(model_ar_tag(float(sys.argv[1])))' "${AR}")"
if [[ "${AR_TAG}" == "AR20" ]]; then
  OUTPUT_TAG="AR2"
else
  OUTPUT_TAG="${AR_TAG}"
fi

if [[ -z "${VSS_TABLE:-}" ]]; then
  case "${AR_TAG}" in
    AR15) VSS_TABLE="models/angular_transport/vss_alpha_eff_table_AR15_peta090.json" ;;
    AR20) VSS_TABLE="models/angular_transport/vss_alpha_eff_table_AR20_peta060.json" ;;
    AR25) VSS_TABLE="models/angular_transport/vss_alpha_eff_table_AR25_peta070.json" ;;
    AR30) VSS_TABLE="models/angular_transport/vss_alpha_eff_table_AR30_peta070.json" ;;
    *)
      echo "ERROR: No default VSS_TABLE for AR=${AR} (${AR_TAG})."
      echo "Pass VSS_TABLE=models/angular_transport/vss_alpha_eff_table_${AR_TAG}_*.json"
      exit 1
      ;;
  esac
fi

C2_TABLE="${C2_TABLE:-models/relaxation/C2_table_${AR_TAG}.json}"
CONFIG_OUT="${CONFIG_OUT:-runs/generated_configs/usf_${AR_TAG}_vss_rank2_C2.yaml}"

if [[ ! -f "${VSS_TABLE}" ]]; then
  echo "ERROR: VSS_TABLE not found: ${VSS_TABLE}"
  exit 1
fi
if [[ ! -f "${C2_TABLE}" ]]; then
  echo "ERROR: C2_TABLE not found: ${C2_TABLE}"
  exit 1
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-runs/${OUTPUT_TAG}_usf_vss_rank2_C2}"
OUTPUT_ARGS=(--output-root "${OUTPUT_ROOT}")

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

python -m scripts.internal.prepare_usf_campaign_config \
  --base-config "${BASE_CONFIG}" \
  --config-out "${CONFIG_OUT}" \
  --AR "${AR}" \
  --mode vss_rank2 \
  --vss-table "${VSS_TABLE}" \
  --rank2-correction-enabled \
  --C2-table "${C2_TABLE}" \
  --workers "${WORKERS}" \
  "${OUTPUT_ARGS[@]}" \
  "${TIME_ARGS[@]}" \
  "${ALPHA_PREP_ARGS[@]}"

python -m scripts.public.run_alpha_sweep \
  --config "${CONFIG_OUT}" \
  --workers "${WORKERS}" \
  --skip-post \
  "${ALPHA_RUN_ARGS[@]}"
