#!/usr/bin/env python3
"""Entry point for pre-processing: fit all collision sub-models from CTC data."""
import argparse
import yaml

from src.preprocessing.fit_all import run_all


def main():
    parser = argparse.ArgumentParser(
        description="Fit collision sub-models from CTC data"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate validation/inspection plots"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    run_all(config, plot=args.plot)


if __name__ == "__main__":
    main()
