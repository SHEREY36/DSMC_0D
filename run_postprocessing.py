#!/usr/bin/env python3
"""Entry point for post-processing: analyze and plot DSMC results."""
import argparse
import os
import yaml

from src.postprocessing.plotting import (
    plot_temperature_evolution,
    plot_temperature_components,
    plot_temperature_ratio_evolution
)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process and plot DSMC simulation results"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results_dir = config['postprocessing']['results_dir']
    figures_dir = config['postprocessing']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)

    # Find all result files
    result_files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith('.txt')
    ])

    if not result_files:
        print(f"No result files found in {results_dir}")
        return

    labels = [os.path.basename(f).replace('.txt', '') for f in result_files]
    print(f"Found {len(result_files)} result files")

    plot_temperature_evolution(
        result_files, labels,
        os.path.join(figures_dir, "temperature_evolution.png")
    )

    plot_temperature_components(
        result_files, labels,
        os.path.join(figures_dir, "temperature_components.png")
    )

    plot_temperature_ratio_evolution(
        result_files, labels,
        os.path.join(figures_dir, "temperature_ratio.png")
    )

    print(f"All plots saved to {figures_dir}/")


if __name__ == "__main__":
    main()
