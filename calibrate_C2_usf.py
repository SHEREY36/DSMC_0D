#!/usr/bin/env python3
import argparse
from pathlib import Path

from src.simulation.usf_c2_calibration import (
    build_usf_C2_table,
    default_paths_for_AR,
)


def _parse_ar_values(raw):
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise argparse.ArgumentTypeError("at least one AR value is required")
    return values


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate rank-2 C2 tables from USF DSMC/LAMMPS theta gaps."
    )
    parser.add_argument(
        "--AR",
        type=_parse_ar_values,
        default=_parse_ar_values("1.1,1.5,2.0,2.5,3.0"),
        help="Comma-separated aspect ratios to calibrate.",
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--lammps-usf-root", default="LAMMPS_data/USF2")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--stats-fraction", type=float, default=0.50)
    parser.add_argument("--plateau-threshold", type=float, default=5.0e-4)
    parser.add_argument("--smooth-window", type=int, default=51)
    parser.add_argument("--lammps-tail-fraction", type=float, default=0.30)
    parser.add_argument("--min-a2", type=float, default=1.0e-12)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a requested AR is missing DSMC, LAMMPS, or C_alpha input.",
    )
    args = parser.parse_args()

    built = []
    skipped = []
    for AR in args.AR:
        paths = default_paths_for_AR(
            AR,
            runs_root=args.runs_root,
            lammps_usf_root=args.lammps_usf_root,
            models_dir=args.models_dir,
        )
        required = [
            paths["dsmc_root"],
            paths["lammps_root"],
            paths["C_alpha_table_file"],
        ]
        missing = [str(path) for path in required if not Path(path).exists()]
        if missing:
            message = f"AR={AR:g}: missing inputs: {', '.join(missing)}"
            if args.strict:
                raise FileNotFoundError(message)
            print(f"SKIP {message}")
            skipped.append((AR, missing))
            continue

        payload = build_usf_C2_table(
            dsmc_root=paths["dsmc_root"],
            lammps_root=paths["lammps_root"],
            C_alpha_table_file=paths["C_alpha_table_file"],
            AR=AR,
            output_path=paths["output_path"],
            stats_fraction=args.stats_fraction,
            plateau_threshold=args.plateau_threshold,
            smooth_window=args.smooth_window,
            lammps_tail_fraction=args.lammps_tail_fraction,
            min_a2=args.min_a2,
        )
        rows = payload["rows"]
        valid = sum(1 for row in rows if row.get("valid", True))
        c2_values = [row["C2"] for row in rows if row.get("valid", True)]
        if c2_values:
            c2_span = f"{min(c2_values):.6g}..{max(c2_values):.6g}"
        else:
            c2_span = "none"
        print(
            f"WROTE {paths['output_path']} "
            f"rows={len(rows)} valid={valid} C2_range={c2_span}"
        )
        built.append(paths["output_path"])

    print(f"Built {len(built)} table(s); skipped {len(skipped)} AR value(s).")


if __name__ == "__main__":
    main()
