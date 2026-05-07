#!/usr/bin/env python3
"""Run one-AR HCS non-Gaussian sweeps with local worker parallelism."""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

from run_hcs_ng_realization import (
    ALPHAS,
    SEEDS,
    build_campaign_config,
    build_output_paths,
    load_models,
)
from src.simulation.dsmc import run_simulation


def _parse_float_csv(value):
    if value is None or value == "":
        return None
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_int_csv(value):
    if value is None or value == "":
        return None
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_domain(value):
    if value is None or value == "":
        return None
    domain = [float(x.strip()) for x in value.split(",") if x.strip()]
    if len(domain) != 3:
        raise ValueError("--domain must have exactly three comma-separated values")
    return domain


def _cfg_get(sweep_cfg, key, default=None):
    return sweep_cfg[key] if key in sweep_cfg else default


def build_tasks(base_config, sweep_cfg, args):
    AR = float(args.AR if args.AR is not None else sweep_cfg["AR"])
    output_root = args.output_root or sweep_cfg["output_root"]
    alpha_values = (
        _parse_float_csv(args.alphas)
        or [float(x) for x in _cfg_get(sweep_cfg, "alpha_values", ALPHAS)]
    )
    seeds = (
        _parse_int_csv(args.seeds)
        or [int(x) for x in _cfg_get(sweep_cfg, "seeds", SEEDS)]
    )
    domain = _parse_domain(args.domain) or _cfg_get(sweep_cfg, "domain")
    dt = args.dt if args.dt is not None else _cfg_get(sweep_cfg, "dt")
    t_end = args.t_end if args.t_end is not None else _cfg_get(sweep_cfg, "t_end")
    tau_end = (
        args.tau_end if args.tau_end is not None else _cfg_get(sweep_cfg, "tau_end")
    )
    sample_start_tau = (
        args.sample_start_tau
        if args.sample_start_tau is not None
        else _cfg_get(sweep_cfg, "sample_start_tau")
    )
    sample_end_tau = (
        args.sample_end_tau
        if args.sample_end_tau is not None
        else _cfg_get(sweep_cfg, "sample_end_tau")
    )
    sample_delta_tau = (
        args.sample_delta_tau
        if args.sample_delta_tau is not None
        else _cfg_get(sweep_cfg, "sample_delta_tau")
    )
    hcs_rescale = not args.no_hcs_rescale
    if args.no_hcs_rescale is False and "hcs_rescale_temperature" in sweep_cfg:
        hcs_rescale = bool(sweep_cfg["hcs_rescale_temperature"])

    tasks = []
    for alpha in alpha_values:
        for realization_index, seed in enumerate(seeds, start=1):
            config, output_dir = build_campaign_config(
                base_config, AR, alpha, seed, output_root,
                domain=domain, t_end=t_end, tau_end=tau_end, dt=dt,
                sample_start_tau=sample_start_tau,
                sample_end_tau=sample_end_tau,
                sample_delta_tau=sample_delta_tau,
                hcs_rescale=hcs_rescale,
            )
            output_path, pressure_path = build_output_paths(
                output_dir, AR, alpha, realization_index
            )
            tasks.append({
                "AR": AR,
                "alpha": float(alpha),
                "seed": int(seed),
                "realization_index": int(realization_index),
                "output_dir": str(output_dir),
                "output_path": str(output_path),
                "pressure_path": str(pressure_path),
                "config": config,
            })
    return tasks


def write_manifest(tasks, output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "hcs_ng_sweep_manifest.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "AR", "alpha", "seed", "realization_index",
            "output_dir", "output_path",
        ])
        for task in tasks:
            writer.writerow([
                f"{task['AR']:g}", f"{task['alpha']:.2f}", task["seed"],
                task["realization_index"], task["output_dir"],
                task["output_path"],
            ])
    return path


_worker_models = None


def _worker_init(config_for_models):
    global _worker_models
    _worker_models = load_models(config_for_models)


def _run_task(task):
    config = task["config"]
    output_dir = Path(task["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"  [AR={task['AR']:g} alpha={task['alpha']:.2f} "
        f"R{task['realization_index']} seed={task['seed']}] starting..."
    )
    models = _worker_models if _worker_models is not None else load_models(config)
    run_simulation(
        config, models, task["seed"], task["output_path"], task["pressure_path"]
    )
    return task["output_path"]


def run_tasks(tasks, workers):
    workers = max(1, int(workers))
    if workers == 1:
        _worker_init(tasks[0]["config"])
        return [_run_task(task) for task in tasks]

    completed = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(tasks[0]["config"],),
    ) as executor:
        futures = {executor.submit(_run_task, task): task for task in tasks}
        for future in as_completed(futures):
            completed.append(future.result())
    return completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one-AR HCS non-Gaussian sweep"
    )
    parser.add_argument("--config", default="config/hcs_ng_sweep.yaml")
    parser.add_argument("--AR", type=float, default=None)
    parser.add_argument("--alphas", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--tau-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--sample-start-tau", type=float, default=None)
    parser.add_argument("--sample-end-tau", type=float, default=None)
    parser.add_argument("--sample-delta-tau", type=float, default=None)
    parser.add_argument("--no-hcs-rescale", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config_file = yaml.safe_load(f)

    base_config = config_file["base_config"]
    sweep_cfg = config_file["hcs_ng_sweep"]
    tasks = build_tasks(base_config, sweep_cfg, args)
    if not tasks:
        print("No HCS NG sweep tasks to run.")
        return

    output_root = Path(tasks[0]["output_dir"]).parents[2]
    manifest = write_manifest(tasks, output_root)
    print(f"Wrote manifest: {manifest}")
    print(
        f"Prepared {len(tasks)} HCS NG task(s): "
        f"AR={tasks[0]['AR']:g}, alphas={sorted({t['alpha'] for t in tasks})}"
    )

    if args.prepare_only:
        return

    workers = args.workers
    if workers is None:
        workers = int(sweep_cfg.get("workers", 1))
    print(f"Running with workers={workers}")
    run_tasks(tasks, workers)
    print("HCS NG sweep complete.")


if __name__ == "__main__":
    main()
