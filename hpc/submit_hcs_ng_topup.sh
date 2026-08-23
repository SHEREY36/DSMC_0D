#!/bin/bash
set -euo pipefail

# Top up the HCS non-Gaussian campaign: resubmit ONLY the realizations that
# have not reached sample_end_tau (sampling_complete=false or missing
# ng_summary.json), instead of the full 4800-task array. Reuses the existing
# per-task job script (hpc/job_hcs_ng_paper_array.slurm) and its
# TASK_OFFSET/AR/alpha/seed indexing scheme; only the --array index list and
# --time are different from a fresh submission.
#
# Why this exists: at fixed 48h walltime, low-alpha (more dissipative)
# realizations stall well short of tau=1500 (see hpc/find_incomplete_hcs_ng_tasks.py
# docstring / project notes) because T_rot collapses relative to the
# rescaled T_trans, making the collision search progressively more
# expensive per unit of simulated tau. Giving the SAME realizations more
# wall-clock time (same seed, same deterministic RNG stream) lets them
# pick up further than they got before; whatever tau they reach is still
# usable via the theta-divergence-aware truncation in
# src/postprocessing/non_gaussian.py.
#
# Usage (from the repo root, after `conda activate` per job_hcs_ng_paper_array.slurm):
#   bash hpc/submit_hcs_ng_topup.sh
#
# Env overrides:
#   OUTPUT_ROOT   default: ${RCAC_SCRATCH:-${SCRATCH:-runs}}/hcs_ng_long_window
#   CONCURRENCY   default: 64   (max simultaneously running array tasks)
#   TOPUP_TIME    default: 96:00:00  (passed to sbatch --time, overrides the
#                 48:00:00 in job_hcs_ng_paper_array.slurm)
#
# This does NOT touch the realizations that already reached sampling_complete
# -- it only (re)runs task_ids reported incomplete by
# find_incomplete_hcs_ng_tasks.py against the CURRENT contents of
# OUTPUT_ROOT. If OUTPUT_ROOT was purged since the original campaign ran,
# this degrades gracefully to resubmitting everything (with the same seeds,
# so previously-complete realizations reproduce identically).

JOB_SCRIPT="${JOB_SCRIPT:-hpc/job_hcs_ng_paper_array.slurm}"
CONCURRENCY="${CONCURRENCY:-64}"
TOPUP_TIME="${TOPUP_TIME:-96:00:00}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RCAC_SCRATCH:-${SCRATCH:-runs}}/hcs_ng_long_window}"
MANIFEST="${OUTPUT_ROOT}/campaign_manifest.csv"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "No manifest at ${MANIFEST}; writing one (does not run anything)."
  python -m scripts.internal.run_hcs_ng_realization \
    --output-root "${OUTPUT_ROOT}" \
    --write-manifest-only
fi

echo "Scanning ${OUTPUT_ROOT} for incomplete realizations..."
PLAN_FILE="$(mktemp)"
python hpc/find_incomplete_hcs_ng_tasks.py \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  > "${PLAN_FILE}"
cat "${PLAN_FILE}"

if ! grep -q '^TASK_OFFSET=' "${PLAN_FILE}"; then
  echo "Nothing incomplete -- campaign already fully sampled. Nothing to submit."
  rm -f "${PLAN_FILE}"
  exit 0
fi

echo "Submitting top-up with CONCURRENCY=${CONCURRENCY} TOPUP_TIME=${TOPUP_TIME}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

prev_job=""
while IFS= read -r line; do
  [[ "${line}" == TASK_OFFSET=* ]] || continue
  # line looks like: TASK_OFFSET=2000 ARRAY='0-99,400-832,...'  # 595 tasks
  offset="$(echo "${line}" | sed -n "s/^TASK_OFFSET=\([0-9]*\).*/\1/p")"
  array_spec="$(echo "${line}" | sed -n "s/.*ARRAY='\([^']*\)'.*/\1/p")"

  if [[ -n "${prev_job}" ]]; then
    job_id="$(
      sbatch --parsable \
        --time="${TOPUP_TIME}" \
        --dependency="afterany:${prev_job}" \
        --export=ALL,TASK_OFFSET="${offset}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
        --array="${array_spec}%${CONCURRENCY}" \
        "${JOB_SCRIPT}"
    )"
  else
    job_id="$(
      sbatch --parsable \
        --time="${TOPUP_TIME}" \
        --export=ALL,TASK_OFFSET="${offset}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
        --array="${array_spec}%${CONCURRENCY}" \
        "${JOB_SCRIPT}"
    )"
  fi
  echo "chunk offset=${offset} array=${array_spec} job=${job_id} dependency=${prev_job:-none}"
  prev_job="${job_id}"
done < "${PLAN_FILE}"

rm -f "${PLAN_FILE}"
echo "Submitted top-up campaign chain, final job=${prev_job}"
