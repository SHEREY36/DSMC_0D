#!/usr/bin/env python3
"""Generate PoF-style HCS paper figures one at a time."""

import argparse
from pathlib import Path

from src.postprocessing.pof_hcs_figures import (
    DEFAULT_AR4_DSMC,
    DEFAULT_DEM_ROOT,
    DEFAULT_ROOT,
    FIGURE_NAMES,
    generate_pof_figures,
    validate_available_data,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=FIGURE_NAMES, required=True)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--macro-root", default=None)
    parser.add_argument("--hist-root", default=None)
    parser.add_argument("--dem-root", default=str(DEFAULT_DEM_ROOT))
    parser.add_argument("--ar4-dsmc-file", default=str(DEFAULT_AR4_DSMC))
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"])
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Print the discovered input data counts before plotting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.check_data:
        summary = validate_available_data(
            root=args.root,
            macro_root=args.macro_root,
            hist_root=args.hist_root,
            dem_root=args.dem_root,
            ar4_dsmc_file=args.ar4_dsmc_file,
        )
        print("Input data summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    written = generate_pof_figures(
        args.figure,
        root=args.root,
        macro_root=args.macro_root,
        hist_root=args.hist_root,
        dem_root=args.dem_root,
        ar4_dsmc_file=args.ar4_dsmc_file,
        models_dir=args.models_dir,
        out_dir=args.out_dir,
        formats=args.formats,
    )
    for path in written:
        print(f"Wrote {Path(path)}")


if __name__ == "__main__":
    main()
