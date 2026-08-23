#!/usr/bin/env python3
"""Build the AR=2 mu-joint collision lookup from CTC output files."""
import argparse

from src.simulation.mu_joint import build_mu_joint_ar2_lookup


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build models/mu_joint_ar2_lookup.npz from AR=2 CTC data"
    )
    parser.add_argument("--source-root", default="../Coll_Models/results")
    parser.add_argument("--output", default="models/mu_joint_ar2_lookup.npz")
    parser.add_argument("--n-mu-bins", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    path = build_mu_joint_ar2_lookup(
        args.source_root, args.output, n_mu_bins=args.n_mu_bins
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
