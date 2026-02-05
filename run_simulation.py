#!/usr/bin/env python3
"""Entry point for running the 0D DSMC simulation."""
import argparse
import yaml

from src.simulation.collision import CollisionModels
from src.simulation.dsmc import run_all_realizations


def main():
    parser = argparse.ArgumentParser(
        description="Run 0D DSMC simulation for spherocylinder HCS"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = config['preprocessing']['model_output_dir']
    gmm_npz = config['preprocessing']['gmm'].get('gmm_cond_file')
    print(f"Loading models from {model_dir}...")
    models = CollisionModels(model_dir, gmm_npz_path=gmm_npz)

    run_all_realizations(config, models)
    print("All realizations complete.")


if __name__ == "__main__":
    main()
