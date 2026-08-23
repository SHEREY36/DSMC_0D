#!/usr/bin/env python3
"""Build CTC-only rank-zero C^(0)(alpha, AR) routing tables.

The estimator uses the elastic CTC pass as the conservative translational
exchange baseline.  It does not use raw translational energy loss.  Instead it
converts the signed Fortran output in ``ftr_data.txt`` to the positive DSMC
removal convention and moment-matches

    C0 = sum(w * f_req * delta_E) / sum(w * B(theta) * delta_E),
    B(theta) = 3*theta/(3*theta+2).

The current CTC data under Thesis/Coll_Models/results are sampled at r=theta=1,
so the output table is a scalar CTC-derived replacement candidate for
``C_alpha_table_file`` rather than evidence for theta dependence.
"""

import argparse
from pathlib import Path

from src.calibration.ctc_c0 import (
    DEFAULT_CTC_ROOT,
    ar_label,
    build_c0_table,
    load_theta_table,
    write_outputs,
)
from src.simulation.collision import CollisionModels


def _parse_alphas(value):
    if not value:
        return None
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_CTC_ROOT))
    parser.add_argument("--AR", type=float, required=True)
    parser.add_argument(
        "--alphas",
        default=None,
        help="Comma-separated alpha values. Default: all available for AR.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output table path. Default depends on --estimator.",
    )
    parser.add_argument("--diagnostics-json", default=None)
    parser.add_argument("--diagnostics-csv", default=None)
    parser.add_argument(
        "--compare-table",
        default=None,
        help="Optional existing C_alpha table used only for diagnostics.",
    )
    parser.add_argument(
        "--microscopic-table",
        default=None,
        help="Optional C0 microscopic table used only for hcs-balance diagnostics.",
    )
    parser.add_argument(
        "--fixed-point-table",
        default=None,
        help="Optional fixed-point C0 table used only for grid diagnostics.",
    )
    parser.add_argument(
        "--estimator",
        choices=(
            "microscopic-ftr",
            "hcs-balance",
            "hcs-fixed-point",
            "hcs-self-consistent",
            "hcs-grid-self-consistent",
            "hcs-grid-scaled-self-consistent",
            "hcs-grid-ctc-balance",
            "hcs-grid-ctc-balance-regularized",
        ),
        default="microscopic-ftr",
        help="Which C0 estimator to build.",
    )
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--gmm-npz", default=None)
    parser.add_argument("--theta-table", default=None)
    parser.add_argument("--beta-a", type=float, default=1.21)
    parser.add_argument("--beta-b", type=float, default=3.67)
    parser.add_argument(
        "--fixed-point-samples",
        type=int,
        default=20000,
        help="Number of analytic DSMC accepted-pair samples for HCS estimators.",
    )
    parser.add_argument(
        "--theta-scan-points",
        type=int,
        default=28,
        help="Number of theta scan points for self-consistent HCS estimators.",
    )
    parser.add_argument(
        "--alpha-regularization-strength",
        type=float,
        default=10.0,
        help="Second-difference smoothing strength for regularized grid estimators.",
    )
    parser.add_argument(
        "--weight-mode",
        choices=("area", "uniform-b"),
        default="area",
        help="Collision-measure correction. Use area for current init_part.f90 data.",
    )
    parser.add_argument(
        "--include-multihit",
        action="store_true",
        help="Include all NPhit rows. Default uses only NPhit == 1.",
    )
    parser.add_argument("--min-delta-E", type=float, default=1.0e-12)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output
    if output is None:
        stem = (
            "C0_hcs_balance_table"
            if args.estimator == "hcs-balance"
            else "C0_hcs_fixed_point_table"
            if args.estimator == "hcs-fixed-point"
            else "C0_hcs_self_consistent_table"
            if args.estimator == "hcs-self-consistent"
            else "C0_hcs_grid_self_consistent_table"
            if args.estimator == "hcs-grid-self-consistent"
            else "C0_hcs_grid_scaled_self_consistent_table"
            if args.estimator == "hcs-grid-scaled-self-consistent"
            else "C0_hcs_grid_ctc_balance_table"
            if args.estimator == "hcs-grid-ctc-balance"
            else "C0_hcs_grid_ctc_balance_regularized_table"
            if args.estimator == "hcs-grid-ctc-balance-regularized"
            else "C0_ctc_table"
        )
        output = Path("models") / "relaxation" / f"{stem}_{ar_label(args.AR)}.json"

    compare_table = args.compare_table
    if compare_table is None:
        candidate = Path("models") / "relaxation" / f"C_alpha_table_{ar_label(args.AR)}.json"
        compare_table = candidate if candidate.exists() else None

    microscopic_table = args.microscopic_table
    if microscopic_table is None and args.estimator in (
        "hcs-balance",
        "hcs-fixed-point",
        "hcs-self-consistent",
        "hcs-grid-self-consistent",
        "hcs-grid-scaled-self-consistent",
        "hcs-grid-ctc-balance",
        "hcs-grid-ctc-balance-regularized",
    ):
        candidate = Path("models") / "relaxation" / f"C0_ctc_table_{ar_label(args.AR)}.json"
        microscopic_table = candidate if candidate.exists() else None

    fixed_point_table = args.fixed_point_table
    if fixed_point_table is None and args.estimator in (
        "hcs-grid-self-consistent",
        "hcs-grid-scaled-self-consistent",
        "hcs-grid-ctc-balance",
        "hcs-grid-ctc-balance-regularized",
    ):
        candidate = (
            Path("models")
            / "relaxation"
            / f"C0_hcs_fixed_point_table_{ar_label(args.AR)}.json"
        )
        fixed_point_table = candidate if candidate.exists() else None

    theta_table = None
    models = None
    if args.estimator in ("hcs-balance", "hcs-fixed-point"):
        theta_path = args.theta_table
        if theta_path is None:
            theta_path = (
                Path(args.models_dir)
                / "targets"
                / f"theta_target_table_{ar_label(args.AR)}.json"
            )
        theta_table = load_theta_table(theta_path)
        gmm_npz = args.gmm_npz
        if gmm_npz is None:
            gmm_npz = (
                Path(args.models_dir)
                / "exchange_gmm"
                / f"gmm_cond_{ar_label(args.AR)}.npz"
            )
        models = CollisionModels(
            args.models_dir,
            gmm_npz_path=str(gmm_npz),
            c_alpha_path=str(compare_table) if compare_table else None,
        )
    elif args.estimator in (
        "hcs-grid-self-consistent",
        "hcs-grid-scaled-self-consistent",
        "hcs-grid-ctc-balance",
        "hcs-grid-ctc-balance-regularized",
    ):
        theta_path = args.theta_table
        if theta_path is None:
            candidate = (
                Path(args.models_dir)
                / "targets"
                / f"theta_target_table_{ar_label(args.AR)}.json"
            )
            theta_path = candidate if candidate.exists() else None
        theta_table = load_theta_table(theta_path) if theta_path else None
        gmm_npz = args.gmm_npz
        if gmm_npz is None:
            gmm_npz = (
                Path(args.models_dir)
                / "exchange_gmm"
                / f"gmm_cond_{ar_label(args.AR)}.npz"
            )
        models = CollisionModels(
            args.models_dir,
            gmm_npz_path=str(gmm_npz),
            c_alpha_path=str(compare_table) if compare_table else None,
        )
    elif args.estimator == "hcs-self-consistent":
        gmm_npz = args.gmm_npz
        if gmm_npz is None:
            gmm_npz = (
                Path(args.models_dir)
                / "exchange_gmm"
                / f"gmm_cond_{ar_label(args.AR)}.npz"
            )
        models = CollisionModels(
            args.models_dir,
            gmm_npz_path=str(gmm_npz),
            c_alpha_path=str(compare_table) if compare_table else None,
        )

    payload = build_c0_table(
        args.root,
        AR=args.AR,
        alphas=_parse_alphas(args.alphas),
        weight_mode=args.weight_mode,
        single_hit_only=not args.include_multihit,
        min_delta_E=args.min_delta_E,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        compare_table=compare_table,
        estimator=args.estimator,
        models=models,
        theta_table=theta_table,
        beta_a=args.beta_a,
        beta_b=args.beta_b,
        microscopic_table=microscopic_table,
        fixed_point_table=fixed_point_table,
        fixed_point_samples=args.fixed_point_samples,
        theta_scan_points=args.theta_scan_points,
        alpha_regularization_strength=args.alpha_regularization_strength,
    )
    written = write_outputs(
        payload,
        output,
        diagnostics_json_path=args.diagnostics_json,
        diagnostics_csv_path=args.diagnostics_csv,
    )
    rows = payload["diagnostics"]["rows"]
    print(
        f"Wrote {args.estimator} C0 table for AR={args.AR:g}: {written[0]} "
        f"({len(rows)} alpha row(s))"
    )
    print(f"Wrote diagnostics JSON: {written[1]}")
    print(f"Wrote diagnostics CSV: {written[2]}")
    for row in rows:
        extra = ""
        if "comparison_C_alpha" in row:
            extra += f" old={row['comparison_C_alpha']:.6g}"
        if "comparison_C_mic" in row:
            extra += f" mic={row['comparison_C_mic']:.6g}"
        if "comparison_C_fixed_point" in row:
            extra += f" fixed={row['comparison_C_fixed_point']:.6g}"
        if "lambda_scale" in row and row["lambda_scale"] == row["lambda_scale"]:
            extra += f" lambda={row['lambda_scale']:.6g}"
        if "C0_raw" in row:
            extra += f" raw={row['C0_raw']:.6g}"
        if "theta_pred" in row and row["theta_pred"] == row["theta_pred"]:
            extra += f" theta={row['theta_pred']:.6g}"
        print(
            f"  alpha={row['alpha']:.3f} C0={row['C0']:.6g} "
            f"n_used={row['n_used']} status={row['status']}{extra}"
        )


if __name__ == "__main__":
    main()
