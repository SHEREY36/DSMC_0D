#!/usr/bin/env python3
"""Build rank-2 C2 correction tables from current CTC artifacts."""
import argparse
import os

from src.simulation.particle import model_ar_tag
from src.simulation.rank2_correction import build_C2_table


DEFAULT_ARS = [1.1, 1.5, 2.0, 2.5, 3.0]


def _parse_csv_floats(text):
    if text is None or str(text).strip() == "":
        return None
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Build C2(alpha, AR) rank-2 f_tr correction tables."
    )
    parser.add_argument(
        "--ctc-root", default="../Coll_Models/results",
        help="Root containing alpha_*_r1.00_AR* CTC result folders.",
    )
    parser.add_argument(
        "--AR", dest="ars", default=None,
        help="Comma-separated AR list. Defaults to 1.1,1.5,2.0,2.5,3.0.",
    )
    parser.add_argument(
        "--epsilons", default="0.02,0.04,0.06,0.08",
        help="Comma-separated positive epsilon perturbation magnitudes.",
    )
    parser.add_argument(
        "--output-dir", default="models",
        help="Directory for C2_table_AR*.json outputs.",
    )
    parser.add_argument(
        "--bootstrap-samples", type=int, default=200,
        help="Bootstrap samples per CTC case for slope uncertainty.",
    )
    parser.add_argument(
        "--min-abs-dissipation", type=float, default=1.0e-30,
        help="Discard rows with |delta_E_diss| at or below this value.",
    )
    parser.add_argument("--rng-seed", type=int, default=12345)
    args = parser.parse_args()

    ars = _parse_csv_floats(args.ars) or list(DEFAULT_ARS)
    epsilons = _parse_csv_floats(args.epsilons)
    os.makedirs(args.output_dir, exist_ok=True)

    for AR in ars:
        ar_tag = model_ar_tag(AR)
        output_path = os.path.join(args.output_dir, f"C2_table_{ar_tag}.json")
        payload = build_C2_table(
            args.ctc_root,
            AR=AR,
            epsilon_values=epsilons,
            output_path=output_path,
            min_abs_dissipation=args.min_abs_dissipation,
            bootstrap_samples=args.bootstrap_samples,
            rng_seed=args.rng_seed,
        )
        rows = payload["rows"]
        print(f"Wrote {output_path} ({len(rows)} alpha rows)")
        for row in rows:
            print(
                f"  alpha={row['alpha']:.2f} C2={row['C2']:.6g} "
                f"stderr={row['stderr']:.3g} n={row['n_valid']}"
            )


if __name__ == "__main__":
    main()
