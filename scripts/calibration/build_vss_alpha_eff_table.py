#!/usr/bin/env python3
"""Build VSS-style rank-2 alpha_eff calibration tables from CTC chi data."""
import argparse
import glob
import os

from src.simulation.particle import model_ar_tag
from src.simulation.vss_rank2 import build_vss_alpha_eff_table, parse_case_dir


DEFAULT_ALPHA_VALUES = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
]


def _parse_alpha_values(text):
    if not text:
        return list(DEFAULT_ALPHA_VALUES)
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def peta_tag(p_eta):
    return f"peta{int(round(float(p_eta) * 100.0)):03d}"


def default_output_path(AR, p_eta):
    return (
        f"models/angular_transport/vss_alpha_eff_table_{model_ar_tag(AR)}_"
        f"{peta_tag(p_eta)}.json"
    )


def discover_case_alphas(ctc_root, AR):
    alphas = []
    pattern = os.path.join(str(ctc_root), "alpha_*_r1.00_AR*")
    for case_dir in sorted(glob.glob(pattern)):
        parsed = parse_case_dir(case_dir)
        if parsed is None:
            continue
        alpha, case_AR = parsed
        if abs(float(case_AR) - float(AR)) <= 1.0e-12:
            alphas.append(float(alpha))
    return sorted(set(alphas))


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
        default=None,
        help="Output JSON path. Defaults to models/angular_transport/vss_alpha_eff_table_AR{tag}_peta{tag}.json",
    )
    parser.add_argument(
        "--alpha-values",
        default=None,
        help="Expected alpha values for preflight reporting, comma-separated.",
    )
    args = parser.parse_args()

    output = args.output or default_output_path(args.AR, args.p_eta)
    expected_alphas = _parse_alpha_values(args.alpha_values)
    found_alphas = discover_case_alphas(args.ctc_root, args.AR)
    missing = [
        alpha for alpha in expected_alphas
        if not any(abs(alpha - found) <= 1.0e-12 for found in found_alphas)
    ]
    print(
        f"Preflight: found {len(found_alphas)} CTC alpha row(s) for "
        f"AR={args.AR:g} ({model_ar_tag(args.AR)}) under {args.ctc_root}"
    )
    if found_alphas:
        print("  found:", ", ".join(f"{alpha:.2f}" for alpha in found_alphas))
    if missing:
        print("  warning: missing expected alpha values:",
              ", ".join(f"{alpha:.2f}" for alpha in missing))

    payload = build_vss_alpha_eff_table(
        args.ctc_root, args.AR, args.p_eta, output_path=output
    )
    rows = payload["rows"]
    print(
        f"Wrote {len(rows)} VSS alpha_eff row(s) for AR={args.AR:g}, "
        f"p_eta={args.p_eta:g}: {output}"
    )
    print(
        f"  alpha range [{rows[0]['alpha']:.2f}, {rows[-1]['alpha']:.2f}], "
        f"alpha_eff range [{min(r['alpha_eff'] for r in rows):.6f}, "
        f"{max(r['alpha_eff'] for r in rows):.6f}]"
    )


if __name__ == "__main__":
    main()
