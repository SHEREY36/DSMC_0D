#!/usr/bin/env python3
"""Merge per-alpha calibration JSONs written by the SLURM array job into the
main C_alpha_table_AR{ar_int}.json.

Run this on the login node after ALL array tasks have finished:
    python hpc/merge_C_alpha_results.py --AR 2.0
    python hpc/merge_C_alpha_results.py --AR 1.5   # for a different AR

Options:
    --AR            Aspect ratio (default: 2.0). Derives default output path
                    and file pattern automatically.
    --results-dir   Directory containing C_alpha_AR*_*.json files
                    (default: runs/calib_C_alpha)
    --output        Destination table file. Default derived from --AR:
                    models/C_alpha_table_AR{ar_int}.json
    --dry-run       Print what would be merged without writing
"""
import argparse
import glob
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Merge per-alpha C(alpha) JSONs")
    parser.add_argument(
        "--AR", type=float, default=2.0,
        help="Aspect ratio being merged (default: 2.0)"
    )
    parser.add_argument(
        "--results-dir", default="runs/calib_C_alpha",
        help="Directory containing C_alpha_AR*_*.json files"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output table path. Default: models/C_alpha_table_AR{ar_int}.json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print merged table without writing to disk"
    )
    args = parser.parse_args()

    ar_int = int(round(args.AR * 10))
    output_path = args.output or f"models/C_alpha_table_AR{ar_int:02d}.json"
    pattern = os.path.join(args.results_dir, f"C_alpha_AR{ar_int:02d}_*.json")

    # Load existing table (if any) as base — preserves entries not re-calibrated
    merged = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            merged = json.load(f)
        print(f"Loaded existing table ({len(merged)} entries): {output_path}")

    # Find and merge per-alpha result files
    result_files = sorted(glob.glob(pattern))

    if not result_files:
        print(f"No result files found matching: {pattern}")
        return

    print(f"\nFound {len(result_files)} result file(s):")
    updated = []
    for path in result_files:
        with open(path) as f:
            data = json.load(f)
        for key, value in data.items():
            old = merged.get(key)
            merged[key] = float(value)
            status = "NEW" if old is None else f"updated {old:.5f} -> {value:.5f}"
            print(f"  {key}: C = {value:.5f}  [{status}]")
            updated.append(key)

    print(f"\nMerged {len(updated)} entr(ies) into table ({len(merged)} total).")

    if args.dry_run:
        print("Dry run — nothing written.")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dict(sorted(merged.items())), f, indent=2)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
