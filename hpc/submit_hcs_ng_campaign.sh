#!/bin/bash
set -euo pipefail

# Submit the 4800-task HCS non-Gaussian campaign as dependent chunks.
# This keeps global concurrency near CONCURRENCY instead of multiplying it
# across independently submitted Slurm arrays.

JOB_SCRIPT="${JOB_SCRIPT:-hpc/job_hcs_ng_paper_array.slurm}"
CONCURRENCY="${CONCURRENCY:-64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RCAC_SCRATCH:-${SCRATCH:-runs}}/hcs_ng_long_window}"

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "ERROR: job script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

submit_chunk() {
  local offset="$1"
  local array_spec="$2"
  local dependency="${3:-}"
  local job_id

  if [[ -n "${dependency}" ]]; then
    job_id="$(
      sbatch --parsable \
        --dependency="afterany:${dependency}" \
        --export=ALL,TASK_OFFSET="${offset}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
        --array="${array_spec}%${CONCURRENCY}" \
        "${JOB_SCRIPT}"
    )"
  else
    job_id="$(
      sbatch --parsable \
        --export=ALL,TASK_OFFSET="${offset}",OUTPUT_ROOT="${OUTPUT_ROOT}" \
        --array="${array_spec}%${CONCURRENCY}" \
        "${JOB_SCRIPT}"
    )"
  fi

  echo "${job_id}"
}

mkdir -p "${OUTPUT_ROOT}"
python run_hcs_ng_realization.py \
  --output-root "${OUTPUT_ROOT}" \
  --write-manifest-only

echo "Submitting HCS NG campaign with CONCURRENCY=${CONCURRENCY}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"

job0="$(submit_chunk 0 0-999)"
echo "chunk offset=0    job=${job0}"

job1="$(submit_chunk 1000 0-999 "${job0}")"
echo "chunk offset=1000 job=${job1} dependency=afterany:${job0}"

job2="$(submit_chunk 2000 0-999 "${job1}")"
echo "chunk offset=2000 job=${job2} dependency=afterany:${job1}"

job3="$(submit_chunk 3000 0-999 "${job2}")"
echo "chunk offset=3000 job=${job3} dependency=afterany:${job2}"

job4="$(submit_chunk 4000 0-799 "${job3}")"
echo "chunk offset=4000 job=${job4} dependency=afterany:${job3}"

echo "Submitted dependent campaign chain: ${job0} -> ${job1} -> ${job2} -> ${job3} -> ${job4}"
