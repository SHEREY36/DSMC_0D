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
from src.postprocessing.sweep_plotting import run_sweep_postprocessing


def _parse_alpha_list(alpha_csv):
    if not alpha_csv:
        return None
    return [float(x.strip()) for x in alpha_csv.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Post-process and plot DSMC simulation results"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--sweep-root", default=None,
        help="Optional root folder with alpha_XXX subfolders for sweep plots"
    )
    parser.add_argument(
        "--alphas", default=None,
        help="Optional comma-separated include set for sweep (e.g. 0.65,0.7,0.75)"
    )
    parser.add_argument(
        "--exclude-alphas", default=None,
        help="Optional comma-separated alphas to skip in sweep (e.g. 0.5,0.55,0.6)"
    )
    parser.add_argument(
        "--min-realizations", type=int, default=None,
        help="Minimum number of seed files required per alpha case"
    )
    parser.add_argument(
        "--figures-dir", default=None,
        help="Override output directory for sweep figures"
    )
    parser.add_argument(
        "--reference-realization", type=int, default=None,
        help="Realization index used for Haff log-log plot (default from config)"
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    post_cfg = config['postprocessing']
    sweep_root_from_cli = args.sweep_root is not None
    sweep_root = args.sweep_root or post_cfg.get('sweep_root')
    if sweep_root and os.path.isdir(sweep_root):
        alpha_filter = _parse_alpha_list(args.alphas)
        if alpha_filter is None:
            alpha_filter = post_cfg.get('active_alphas', [])
        alpha_exclude = _parse_alpha_list(args.exclude_alphas)
        if alpha_exclude is None:
            alpha_exclude = post_cfg.get('exclude_alphas', [])

        min_realizations = args.min_realizations
        if min_realizations is None:
            min_realizations = int(post_cfg.get('sweep_min_realizations', 1))
        reference_realization = args.reference_realization
        if reference_realization is None:
            reference_realization = int(
                post_cfg.get('reference_realization_index', 0)
            )
        figures_dir = args.figures_dir or post_cfg.get(
            'sweep_figures_dir', os.path.join(sweep_root, "figures")
        )

        generated = run_sweep_postprocessing(
            sweep_root=sweep_root,
            figures_dir=figures_dir,
            alpha_filter=alpha_filter,
            alpha_exclude=alpha_exclude,
            n_time_points=int(post_cfg.get('sweep_n_time_points', 1200)),
            asymptotic_tail_fraction=float(
                post_cfg.get('asymptotic_tail_fraction', 0.2)
            ),
            normalize_total_temperature=bool(
                post_cfg.get('normalize_total_temperature', True)
            ),
            min_realizations=min_realizations,
            reference_realization_index=reference_realization,
        )
        if generated or sweep_root_from_cli:
            return

    results_dir = post_cfg['results_dir']
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
