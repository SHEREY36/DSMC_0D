#!/usr/bin/env python3
"""Find HCS non-Gaussian campaign task_ids that have not reached sample_end_tau.

Reads the campaign manifest (task_id, AR, alpha, seed, realization_index,
output_dir) and, for each row, checks the corresponding *_ng_summary.json
for sampling_complete=true. Rows with no summary file (never ran, or died
before writing output) count as incomplete too.

Prints either:
  - one task_id per line (--format ids), or
  - SLURM --array specs grouped into the campaign's 1000-wide TASK_OFFSET
    windows, range-compressed (--format slurm, the default).

Usage (run from the repo root, after activating the campaign's conda env):
    python hpc/find_incomplete_hcs_ng_tasks.py \
        --manifest runs/hcs_ng_long_window/campaign_manifest.csv \
        --output-root runs/hcs_ng_long_window
"""
import argparse
import csv
import json
import os


def ar_dirname(ar):
    return f"AR{round(float(ar) * 100):03d}"


def alpha_dirname(alpha):
    return f"alpha_{round(float(alpha) * 100):03d}"


def summary_filename(ar, alpha, realization_index):
    return (
        f"AR{round(float(ar) * 100):03d}"
        f"_COR{round(float(alpha) * 100):03d}"
        f"_R{int(realization_index):03d}_ng_summary.json"
    )


def is_complete(output_dir, ar, alpha, realization_index):
    path = os.path.join(
        output_dir, summary_filename(ar, alpha, realization_index)
    )
    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(data.get("sampling_complete"))


def compress_ranges(sorted_ints):
    """[0,1,2,5,6,9] -> ['0-2', '5-6', '9']"""
    ranges = []
    start = prev = None
    for n in sorted_ints:
        if start is None:
            start = prev = n
            continue
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    if start is not None:
        ranges.append((start, prev))
    return [f"{a}-{b}" if a != b else f"{a}" for a, b in ranges]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--output-root", required=True,
        help="Campaign output root (its AR*/alpha_*/results dirs are read).",
    )
    parser.add_argument(
        "--window-size", type=int, default=1000,
        help="TASK_OFFSET chunk width used by submit_hcs_ng_campaign.sh.",
    )
    parser.add_argument(
        "--format", choices=["slurm", "ids"], default="slurm",
    )
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    incomplete = []
    for row in rows:
        task_id = int(row["task_id"])
        ar = float(row["AR"])
        alpha = float(row["alpha"])
        realization_index = int(row["realization_index"])
        output_dir = os.path.join(
            args.output_root, ar_dirname(ar), alpha_dirname(alpha), "results"
        )
        if not is_complete(output_dir, ar, alpha, realization_index):
            incomplete.append(task_id)

    incomplete.sort()

    if args.format == "ids":
        for t in incomplete:
            print(t)
        return

    print(f"# {len(incomplete)} / {len(rows)} realizations incomplete "
          f"(no ng_summary.json or sampling_complete=false)")
    by_window = {}
    for t in incomplete:
        window = (t // args.window_size) * args.window_size
        by_window.setdefault(window, []).append(t - window)
    for window in sorted(by_window):
        local_ids = sorted(by_window[window])
        spec = ",".join(compress_ranges(local_ids))
        print(f"TASK_OFFSET={window} ARRAY='{spec}'  # {len(local_ids)} tasks")


if __name__ == "__main__":
    main()
