#!/usr/bin/env python3
"""Extract steady-state theta* = T_trans/T_rot and case-specific t_end from
LAMMPS HCS data, and save both to a JSON lookup table.

Run this once before running calibrate_C_alpha.py.  The table stores, per
(alpha, AR) case, the steady-state temperature ratio AND the simulation
end-time the DSMC calibration should use — derived from the LAMMPS settling
time plus a safety margin.

Usage:
    python extract_theta_target.py --config config/default.yaml --AR 2.0
    python extract_theta_target.py --config config/default.yaml --AR 1.1 \\
        --lammps-root LAMMPS_data/HCS/Calibrate_r/modeB_e_sweep11 --force
    python extract_theta_target.py --config config/default.yaml --AR 2.0 \\
        --alphas 0.80,0.85,0.90

Output schema  (models/theta_target_table_AR{ar_int}.json):
    {
      "(0.500, 2.0)": {"theta_star": 0.8736, "t_end": 374.0},
      ...
    }

Backward compatibility: calibrate_C_alpha.py also accepts the old flat-float
schema {"(alpha, AR)": float} and falls back to config t_end in that case.
"""
import argparse
import glob
import json
import os

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# LAMMPS data reading
# ---------------------------------------------------------------------------

def _read_lammps_hcs(lammps_root, alpha):
    """Read a LAMMPS HCS temperature file.

    Returns (time, theta) arrays, or raises FileNotFoundError.
    Expected columns: time  T_trans  T_rot  T_tr/T_rot  T_total
    """
    alpha_int = int(round(alpha * 100))
    folder = os.path.join(lammps_root, f"e_{alpha_int:03d}")
    candidates = glob.glob(os.path.join(folder, "hcs_temperatures_B_e*.dat"))
    if not candidates:
        raise FileNotFoundError(
            f"No hcs_temperatures_B_e*.dat found in {folder}"
        )
    data = np.loadtxt(candidates[0], comments='#')
    return data[:, 0], data[:, 3]   # time, theta = T_trans/T_rot


# ---------------------------------------------------------------------------
# theta* extraction
# ---------------------------------------------------------------------------

def extract_theta_star(time, theta, tail_fraction=0.2):
    """Return the mean of the last tail_fraction of the theta column."""
    n_tail = max(10, int(tail_fraction * len(theta)))
    return float(np.mean(theta[-n_tail:]))


# ---------------------------------------------------------------------------
# Settling-time detection
# ---------------------------------------------------------------------------

def detect_settling_time(t, theta, theta_star,
                         window_frac=0.02, k_sigma=3.0, min_rel_tol=0.01):
    """Return the last simulation time at which the smoothed theta was still
    outside the steady-state tolerance band.

    The band is centred on theta_star with half-width:
        tol = max(k_sigma * noise_floor, min_rel_tol * |theta_star|)

    where noise_floor = std of the smoothed signal over the last 20% of the
    run (an empirical measure of the steady-state fluctuation amplitude).

    Uses scipy.ndimage.uniform_filter1d with mode='nearest' to avoid the
    zero-padding edge artefacts of np.convolve(..., mode='same').

    Args:
        t            : 1-D time array (same units as DSMC t_end)
        theta        : 1-D theta = T_trans/T_rot array
        theta_star   : steady-state value (pre-computed from tail mean)
        window_frac  : smoothing window as fraction of n  (default 0.02 = 2%)
        k_sigma      : band half-width in noise-floor sigmas         (default 3)
        min_rel_tol  : minimum band half-width as fraction of theta* (default 1%)

    Returns:
        t_settle (float): last time the smooth signal left the band.
                          Returns t[0] if the signal is always within the band.
    """
    from scipy.ndimage import uniform_filter1d

    n = len(theta)
    window = max(200, int(window_frac * n))
    smooth = uniform_filter1d(theta.astype(float), size=window, mode='nearest')

    n_tail = max(10, int(0.2 * n))
    noise = np.std(smooth[-n_tail:])
    tol = max(k_sigma * noise, min_rel_tol * abs(theta_star))

    outside = np.abs(smooth - theta_star) > tol
    if not np.any(outside):
        return float(t[0])

    return float(t[np.where(outside)[0][-1]])


def compute_t_end(t_settle, margin=0.3, min_t_end=50.0):
    """Apply a safety margin above the detected settling time."""
    return max(t_settle * (1.0 + margin), min_t_end)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract theta* and case-specific t_end from LAMMPS HCS data "
            "and save to a JSON lookup table used by calibrate_C_alpha.py."
        )
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--AR", type=float, default=None,
        help="Aspect ratio to extract. Default: from config."
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
        help="Fraction of run used to estimate theta* (default 0.2 = last 20%%)."
    )
    # Settling-time parameters
    parser.add_argument(
        "--margin", type=float, default=0.3,
        help="Multiplicative safety margin above t_settle: "
             "t_end = t_settle * (1 + margin).  Default: 0.3 (30%%)."
    )
    parser.add_argument(
        "--min-t-end", type=float, default=50.0,
        help="Minimum t_end regardless of settling time (default 50.0). "
             "Prevents near-zero t_end for fast-settling or near-elastic cases."
    )
    parser.add_argument(
        "--settle-window-frac", type=float, default=0.02,
        help="Smoothing window for settling detection as fraction of data "
             "length (default 0.02 = 2%%)."
    )
    parser.add_argument(
        "--settle-k-sigma", type=float, default=3.0,
        help="Tolerance band half-width in noise-floor sigma units (default 3.0)."
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

    print(f"\nExtracting theta* and t_end for AR={AR}, alphas={alpha_values}")
    print(f"LAMMPS root:        {lammps_root}")
    print(f"Tail fraction:      {args.tail_fraction}")
    print(f"Settling margin:    {args.margin:.0%}  (min t_end = {args.min_t_end})")
    print(f"Settling window:    {args.settle_window_frac:.0%} of data, "
          f"k_sigma={args.settle_k_sigma}")
    print(f"Output:             {output_path}\n")

    changed = False
    for alpha in alpha_values:
        key = f"({alpha:.3f}, {AR:.1f})"

        # Check if we can skip
        if key in table and not args.force:
            entry = table[key]
            if isinstance(entry, dict):
                print(f"  [{alpha:.2f}] Already in table "
                      f"(theta*={entry['theta_star']:.5f}, "
                      f"t_end={entry['t_end']:.1f}), skipping.")
            else:
                # Old flat-float entry: re-extract to add t_end
                print(f"  [{alpha:.2f}] Old float entry found — "
                      f"upgrading to dict (use --force to skip this message).")
            if isinstance(entry, dict):
                continue

        try:
            time_arr, theta_arr = _read_lammps_hcs(lammps_root, alpha)
        except FileNotFoundError as e:
            print(f"  [{alpha:.2f}] Skipping: {e}")
            continue

        theta_star = extract_theta_star(time_arr, theta_arr, args.tail_fraction)
        t_settle = detect_settling_time(
            time_arr, theta_arr, theta_star,
            window_frac=args.settle_window_frac,
            k_sigma=args.settle_k_sigma,
        )
        t_end = compute_t_end(t_settle, margin=args.margin, min_t_end=args.min_t_end)

        table[key] = {"theta_star": theta_star, "t_end": t_end}
        changed = True
        print(f"  [{alpha:.2f}] theta*={theta_star:.6f}  "
              f"t_settle={t_settle:.1f}  t_end={t_end:.1f}")

    if not changed:
        print("\nNothing new to extract.")
    else:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dict(sorted(table.items())), f, indent=2)
        print(f"\nSaved {len(table)}-entry table to {output_path}")

    print("\nSummary:")
    for key, val in sorted(table.items()):
        if isinstance(val, dict):
            print(f"  {key}: theta*={val['theta_star']:.6f}  t_end={val['t_end']:.1f}")
        else:
            print(f"  {key}: theta*={val:.6f}  (no t_end — old format)")


if __name__ == "__main__":
    main()
