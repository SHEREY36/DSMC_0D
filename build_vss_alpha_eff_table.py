#!/usr/bin/env python3
"""Build VSS-style rank-2 alpha_eff calibration tables from CTC chi data."""
import argparse

from src.simulation.vss_rank2 import build_vss_alpha_eff_table


def main():
    parser = argparse.ArgumentParser(
        description="Build alpha_eff table for the VSS rank-2 scattering kernel"
    )
    parser.add_argument(
        "--ctc-root",
        default="/home/muhammed/Documents/Thesis/Coll_Models/results",
        help="Root containing alpha_*_r1.00_AR* CTC result folders",
    )
    parser.add_argument("--AR", type=float, default=2.0)
    parser.add_argument("--p-eta", type=float, default=0.6)
    parser.add_argument(
        "--output",
        default="models/vss_alpha_eff_table_AR20_peta060.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    payload = build_vss_alpha_eff_table(
        args.ctc_root, args.AR, args.p_eta, output_path=args.output
    )
    rows = payload["rows"]
    print(
        f"Wrote {len(rows)} VSS alpha_eff row(s) for AR={args.AR:g}, "
        f"p_eta={args.p_eta:g}: {args.output}"
    )
    print(
        f"  alpha range [{rows[0]['alpha']:.2f}, {rows[-1]['alpha']:.2f}], "
        f"alpha_eff range [{min(r['alpha_eff'] for r in rows):.6f}, "
        f"{max(r['alpha_eff'] for r in rows):.6f}]"
    )


if __name__ == "__main__":
    main()
