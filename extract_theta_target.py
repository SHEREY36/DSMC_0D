#!/usr/bin/env python3
"""Extract steady-state theta* = T_trans/T_rot from LAMMPS HCS data and save to a JSON table.

Run this once (on login node for HPC, or locally) before running calibrate_C_alpha.py.
The table can be updated incrementally as new AR/alpha data becomes available.

Usage:
    python extract_theta_target.py --config config/default.yaml --AR 2.0
    python extract_theta_target.py --config config/default.yaml --AR 2.0 --alphas 0.80,0.85,0.90
    python extract_theta_target.py --config config/default.yaml --AR 2.0 --force

Output: models/theta_target_table_AR20.json  (key: "(alpha, AR)" -> float theta*)
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml


def extract_theta_star(lammps_root, alpha, tail_fraction=0.2):
    """Extract steady-state theta* = T_trans/T_rot from a LAMMPS HCS temperature file.

    Reads `hcs_temperatures_B_e*.dat` in the alpha subdirectory.
    Expected columns: time  T_trans  T_rot  T_tr/T_rot  T_total

    Returns the mean of the last `tail_fraction` of the theta column.
    """
    alpha_int = int(round(alpha * 100))
    folder = os.path.join(lammps_root, f"e_{alpha_int:03d}")
    candidates = glob.glob(os.path.join(folder, "hcs_temperatures_B_e*.dat"))
    if not candidates:
        raise FileNotFoundError(
            f"No hcs_temperatures_B_e*.dat found in {folder}"
        )
    data = np.loadtxt(candidates[0], comments='#')
    theta_col = data[:, 3]   # column 3: T_trans / T_rot
    n_tail = max(10, int(tail_fraction * len(theta_col)))
    return float(np.mean(theta_col[-n_tail:]))


def main():
    parser = argparse.ArgumentParser(
        description="Extract theta* from LAMMPS HCS data and save to a JSON lookup table"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--AR", type=float, default=None,
        help="Aspect ratio to extract. Default: from config (preprocessing.zr_eff.AR)."
    )
    parser.add_argument(
        "--alphas", default=None,
        help="Comma-separated alpha values, e.g. 0.65,0.70. Default: all in config."
    )
    parser.add_argument(
        "--lammps-root", default=None,
        help="Root directory of LAMMPS data (contains e_050/, e_060/, ...). "
             "Default: config preprocessing.zr_eff.lammps_root."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: models/theta_target_table_AR{ar_int}.json"
    )
    parser.add_argument(
        "--tail-fraction", type=float, default=0.2,
        help="Fraction of tail used to estimate steady-state theta* (default 0.2)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract and overwrite entries already present in the table."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # AR
    AR = args.AR if args.AR is not None else float(
        config['preprocessing']['zr_eff'].get('AR', 2.0)
    )
    ar_int = int(round(AR * 10))

    # Alpha values
    if args.alphas:
        alpha_values = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    else:
        alpha_values = config['preprocessing']['zr_eff']['alpha_values']
        alpha_values = [a for a in alpha_values if a < 1.0]

    # LAMMPS root
    lammps_root = args.lammps_root or config['preprocessing']['zr_eff']['lammps_root']

    # Output path
    output_path = args.output or config.get('calibration', {}).get(
        'theta_target_table_file',
        f"models/theta_target_table_AR{ar_int:02d}.json"
    )

    # Load existing table (resume support)
    if os.path.exists(output_path):
        with open(output_path) as f:
            table = json.load(f)
        print(f"Loaded existing table: {output_path} ({len(table)} entries)")
    else:
        table = {}

    print(f"\nExtracting theta* for AR={AR}, alphas={alpha_values}")
    print(f"LAMMPS root:    {lammps_root}")
    print(f"Tail fraction:  {args.tail_fraction}")
    print(f"Output:         {output_path}\n")

    changed = False
    for alpha in alpha_values:
        key = f"({alpha:.3f}, {AR:.1f})"
        if key in table and not args.force:
            print(f"  [{alpha:.2f}] Already in table (theta*={table[key]:.5f}), skipping.")
            continue
        try:
            theta_star = extract_theta_star(lammps_root, alpha, args.tail_fraction)
        except FileNotFoundError as e:
            print(f"  [{alpha:.2f}] Skipping: {e}")
            continue
        table[key] = theta_star
        changed = True
        print(f"  [{alpha:.2f}] theta* = {theta_star:.6f}")

    if not changed:
        print("\nNothing new to extract.")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dict(sorted(table.items())), f, indent=2)
        print(f"\nSaved {len(table)}-entry table to {output_path}")

    print("\nSummary:")
    for key, val in sorted(table.items()):
        print(f"  {key}: theta* = {val:.6f}")


if __name__ == "__main__":
    main()
