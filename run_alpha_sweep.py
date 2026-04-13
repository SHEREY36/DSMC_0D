#!/usr/bin/env python3
"""Prepare and run AR-specific alpha sweep cases for eta calibration."""
import argparse
import yaml

from src.simulation.alpha_sweep import prepare_sweep_cases, run_prepared_cases
from src.postprocessing.sweep_plotting import run_sweep_postprocessing


def _parse_alphas(alpha_csv):
    if not alpha_csv:
        return None
    return [float(x.strip()) for x in alpha_csv.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Prepare/run alpha sweep calibration cases"
    )
    parser.add_argument(
        "--config", default="config/default.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--prepare-only", action="store_true",
        help="Only create/update sweep case folders and per-case config files"
    )
    parser.add_argument(
        "--alphas", default=None,
        help="Optional comma-separated subset, e.g. 0.8,0.85,1.0"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers for running cases"
    )
    parser.add_argument(
        "--skip-post", action="store_true",
        help="Skip sweep postprocessing plots after simulation"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cli_alphas = _parse_alphas(args.alphas)
    sweep_info = prepare_sweep_cases(config, cli_alphas=cli_alphas)

    print(f"Prepared {len(sweep_info['all_cases'])} case folders under:")
    print(f"  {sweep_info['output_root']}")
    print(f"Active alphas: {sweep_info['active_alphas']}")

    if args.prepare_only:
        return

    sweep_cfg = config["calibration_sweep"]
    workers = args.workers
    if workers is None:
        workers = int(sweep_cfg.get("parallel_workers", 1))
    workers = max(1, workers)

    print(f"Running {len(sweep_info['run_cases'])} active case(s) with workers={workers}")
    run_prepared_cases(sweep_info["run_cases"], workers=workers)
    print("Sweep simulation complete.")

    if not args.skip_post:
        post_cfg = config.get("postprocessing", {})
        figures_dir = post_cfg.get(
            "sweep_figures_dir",
            f"{sweep_info['output_root']}/figures"
        )
        run_sweep_postprocessing(
            sweep_root=sweep_info["output_root"],
            figures_dir=figures_dir,
            alpha_filter=sweep_info["active_alphas"],
            n_time_points=int(post_cfg.get("sweep_n_time_points", 1200)),
            asymptotic_tail_fraction=float(
                post_cfg.get("asymptotic_tail_fraction", 0.2)
            ),
            normalize_total_temperature=bool(
                post_cfg.get("normalize_total_temperature", True)
            )
        )


if __name__ == "__main__":
    main()
