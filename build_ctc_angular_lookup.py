#!/usr/bin/env python3
"""Build the CTC-conditioned angular scattering lookup artifact."""
import argparse

from src.simulation.ctc_angular import (
    DEFAULT_SOURCE_ROOT,
    build_ctc_angular_lookup,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build CTC angular conditional lookup from Coll_Models results."
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT,
                        help="Root containing alpha_*_r1.00_AR* CTC folders.")
    parser.add_argument("--output", default="models/ctc_angular_conditional_ARge15.npz",
                        help="Output .npz model artifact.")
    parser.add_argument("--n-mu-bins", type=int, default=20,
                        help="Equal-DSMC-probability mu bins.")
    parser.add_argument("--min-AR", type=float, default=1.5,
                        help="Minimum AR to include.")
    parser.add_argument("--norm-tol", type=float, default=5.0e-3,
                        help="Allowed max |norm(ghat_post)-1|.")
    args = parser.parse_args()

    output = build_ctc_angular_lookup(
        args.source_root,
        args.output,
        n_mu_bins=args.n_mu_bins,
        min_AR=args.min_AR,
        norm_tol=args.norm_tol,
    )
    print(f"Wrote CTC angular lookup: {output}")


if __name__ == "__main__":
    main()
