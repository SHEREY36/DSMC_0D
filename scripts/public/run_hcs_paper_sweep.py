#!/usr/bin/env python3
"""Run the local HCS DSMC sweep needed for the paper figures."""

import argparse
import copy
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.internal.usf.dsmc import run_simulation as run_hist_simulation
from src.simulation.collision import CollisionModels
from src.simulation.dsmc import run_simulation as run_macro_simulation


DEFAULT_SEEDS = (42, 123, 999, 1234, 5678)
MACRO_CASES = (
    (1.5, 0.95),
    (2.0, 0.95),
    (2.5, 0.95),
    (3.0, 0.95),
    (2.0, 0.90),
    (2.0, 0.70),
    (2.0, 0.50),
)
HIST_CASES = (
    (2.0, 0.95),
    (2.0, 0.80),
    (2.0, 0.60),
)
MODEL_LABELS = {
    1.5: "AR15",
    2.0: "AR20",
    2.5: "AR25",
    3.0: "AR30",
}


@dataclass(frozen=True)
class SweepTask:
    task_id: int
    group: str
    AR: float
    alpha: float
    seed: int
    realization_index: int
    t_end: float
    dt: float
    output_path: str
    pressure_path: str
    config_path: str
    gmm_cond_file: str
    c_alpha_table_file: str
    sample_axis: str = ""
    sample_start_t: float | None = None
    sample_end_t: float | None = None
    sample_delta_t: float | None = None


def ar_folder_label(AR):
    return f"AR{int(round(float(AR) * 100)):03d}"


def alpha_folder_label(alpha):
    return f"alpha_{int(round(float(alpha) * 100)):03d}"


def _model_label_for_ar(AR):
    AR = float(AR)
    if AR not in MODEL_LABELS:
        raise ValueError(f"No paper-sweep model mapping for AR={AR:g}")
    return MODEL_LABELS[AR]


def _load_base_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw["base_config"] if "base_config" in raw else raw


def _parse_domain(value):
    if value is None:
        return [200, 200, 200]
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError("--domain must have exactly three comma-separated values")
    return parts


def _case_output_paths(output_root, group, AR, alpha, realization_index):
    result_dir = (
        Path(output_root)
        / group
        / ar_folder_label(AR)
        / alpha_folder_label(alpha)
        / "results"
    )
    stem = (
        f"{ar_folder_label(AR)}_COR{int(round(float(alpha) * 100)):03d}"
        f"_R{int(realization_index):03d}"
    )
    return (
        result_dir / f"{stem}.txt",
        result_dir / f"{stem}_pressure.txt",
        result_dir / f"{stem}_config.yaml",
    )


def _model_paths(models_dir, AR):
    label = _model_label_for_ar(AR)
    models_dir = Path(models_dir)
    gmm_path = models_dir / "exchange_gmm" / f"gmm_cond_{label}.npz"
    c_alpha_path = models_dir / "relaxation" / f"C_alpha_table_{label}.json"
    for path, description in [
        (gmm_path, f"GMM conditional model for AR={AR:g}"),
        (c_alpha_path, f"C_alpha table for AR={AR:g}"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    return str(gmm_path), str(c_alpha_path)


def _build_config(base_config, task, models_dir, domain, phi):
    cfg = copy.deepcopy(base_config)
    cfg["particle"]["AR"] = float(task.AR)
    cfg["system"]["alpha"] = float(task.alpha)
    cfg["system"]["kTt"] = 1.0
    cfg["system"]["kTr"] = 1.0
    cfg["system"]["phi"] = float(phi)
    cfg["system"]["domain"] = domain
    cfg["system"]["C_alpha"] = None

    cfg["time"]["dt"] = float(task.dt)
    cfg["time"]["dtau"] = 0.1 if task.group == "macro" else 5.0
    cfg["time"]["t_end"] = float(task.t_end)
    cfg["time"]["tau_end"] = None
    cfg["time"]["equilibration_time"] = 0.0

    cfg.setdefault("flow", {})
    cfg["flow"]["mode"] = "hcs"
    cfg["flow"]["shear_rate"] = 0.0

    cfg.setdefault("simulation", {})
    cfg["simulation"]["seeds"] = [int(task.seed)]
    cfg["simulation"]["output_dir"] = str(Path(task.output_path).parent)
    cfg["simulation"]["sphere_collision"] = False
    cfg["simulation"]["use_isotropic_eps"] = True
    cfg["simulation"]["angular_transport_model"] = "current"
    cfg["simulation"]["hcs_rescale_temperature"] = False
    cfg["simulation"]["hcs_rescale_reference"] = "initial"
    cfg["simulation"]["hcs_rescale_vrmax_policy"] = "reset"
    cfg["simulation"]["malloc_trim_interval_steps"] = 1000

    cfg.setdefault("preprocessing", {}).setdefault("gmm", {})
    cfg.setdefault("preprocessing", {}).setdefault("dissipation", {})
    cfg["preprocessing"]["model_output_dir"] = str(models_dir)
    cfg["preprocessing"]["gmm"]["gmm_cond_file"] = task.gmm_cond_file
    cfg.setdefault("calibration", {})
    cfg["calibration"]["C_alpha_table_file"] = task.c_alpha_table_file

    cfg.setdefault("diagnostics", {})["non_gaussian"] = {"enabled": False}
    if task.group == "hist":
        cfg["diagnostics"]["non_gaussian"] = {
            "enabled": True,
            "sample_axis": "t",
            "sample_start_t": task.sample_start_t,
            "sample_end_t": task.sample_end_t,
            "sample_delta_t": task.sample_delta_t,
            "sample_every_outputs": 1,
            "hist_speed_bins": 256,
            "hist_speed_range": [0.0, 7.0],
            "hist_rot_speed_bins": 256,
            "hist_rot_speed_range": [0.0, 7.0],
            "hist_energy_tr_bins": 256,
            "hist_energy_tr_range": [0.0, 16.0],
            "hist_energy_rot_bins": 256,
            "hist_energy_rot_range": [0.0, 16.0],
            "hist_energy_coupling_bins": 256,
            "hist_energy_coupling_range": [0.0, 64.0],
            "write_time_series": True,
            "write_histograms": True,
            "write_progress": False,
        }
    return cfg


def _validate_task_model_paths(task, cfg):
    label = _model_label_for_ar(task.AR)
    checks = [
        ("particle.AR", float(cfg["particle"]["AR"]), float(task.AR)),
        ("system.alpha", float(cfg["system"]["alpha"]), float(task.alpha)),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"Task/config mismatch for task {task.task_id}: "
                f"{name}={actual:g}, expected {expected:g}"
            )

    path_checks = {
        "preprocessing.gmm.gmm_cond_file": cfg["preprocessing"]["gmm"][
            "gmm_cond_file"
        ],
        "calibration.C_alpha_table_file": cfg["calibration"][
            "C_alpha_table_file"
        ],
    }
    for name, path in path_checks.items():
        if label not in Path(path).name:
            raise ValueError(
                f"Task/model mismatch for task {task.task_id} "
                f"(AR={task.AR:g}): {name}={path!r} does not contain {label}"
            )


def build_tasks(
    output_root,
    seeds=DEFAULT_SEEDS,
    groups=("macro", "hist"),
    dt=0.05,
    macro_t_end=1000.0,
    hist_t_end=1200.0,
    sample_start_t=200.0,
    sample_end_t=1200.0,
    sample_delta_t=20.0,
    models_dir="models",
):
    tasks = []
    task_id = 0
    selected = {group.strip() for group in groups if group.strip()}
    for group, cases, t_end in [
        ("macro", MACRO_CASES, macro_t_end),
        ("hist", HIST_CASES, hist_t_end),
    ]:
        if group not in selected:
            continue
        for AR, alpha in cases:
            gmm_path, c_alpha_path = _model_paths(models_dir, AR)
            for realization_index, seed in enumerate(seeds, start=1):
                output_path, pressure_path, config_path = _case_output_paths(
                    output_root, group, AR, alpha, realization_index
                )
                tasks.append(
                    SweepTask(
                        task_id=task_id,
                        group=group,
                        AR=float(AR),
                        alpha=float(alpha),
                        seed=int(seed),
                        realization_index=realization_index,
                        t_end=float(t_end),
                        dt=float(dt),
                        output_path=str(output_path),
                        pressure_path=str(pressure_path),
                        config_path=str(config_path),
                        gmm_cond_file=gmm_path,
                        c_alpha_table_file=c_alpha_path,
                        sample_axis="t" if group == "hist" else "",
                        sample_start_t=sample_start_t if group == "hist" else None,
                        sample_end_t=sample_end_t if group == "hist" else None,
                        sample_delta_t=sample_delta_t if group == "hist" else None,
                    )
                )
                task_id += 1
    return tasks


def _write_manifest(output_root, tasks):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [asdict(task) for task in tasks]
    json_path = root / "paper_sweep_manifest.json"
    csv_path = root / "paper_sweep_manifest.csv"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)
        f.write("\n")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return csv_path, json_path


def _task_complete(task):
    output = Path(task.output_path)
    if not output.exists() or output.stat().st_size == 0:
        return False
    if task.group != "hist":
        return True
    root, _ = os.path.splitext(task.output_path)
    required = [
        f"{root}_ng_summary.json",
        f"{root}_ng_hist_speed.txt",
        f"{root}_ng_hist_energy_rot.txt",
    ]
    return all(Path(path).exists() and Path(path).stat().st_size > 0 for path in required)


def _run_task(task_dict, base_config, models_dir, domain, phi, skip_existing, force):
    task = SweepTask(**task_dict)
    if skip_existing and not force and _task_complete(task):
        return {"task_id": task.task_id, "status": "skipped", "output_path": task.output_path}

    cfg = _build_config(base_config, task, models_dir, domain, phi)
    _validate_task_model_paths(task, cfg)
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(task.config_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(
        f"[start] task={task.task_id} group={task.group} AR={task.AR:g} "
        f"alpha={task.alpha:.2f} seed={task.seed} "
        f"C_alpha={cfg['calibration']['C_alpha_table_file']} "
        f"GMM={cfg['preprocessing']['gmm']['gmm_cond_file']} "
        f"output={task.output_path}",
        flush=True,
    )
    models = CollisionModels(
        cfg["preprocessing"]["model_output_dir"],
        gmm_npz_path=cfg["preprocessing"]["gmm"]["gmm_cond_file"],
        ftr_params_path="",
        c_alpha_path=cfg["calibration"]["C_alpha_table_file"],
    )
    if task.group == "hist":
        run_hist_simulation(
            cfg, models, task.seed, task.output_path, task.pressure_path
        )
    else:
        run_macro_simulation(cfg, models, task.seed, task.output_path)
    return {"task_id": task.task_id, "status": "ran", "output_path": task.output_path}


def _parse_groups(value):
    groups = tuple(group.strip() for group in value.split(",") if group.strip())
    allowed = {"macro", "hist"}
    unknown = sorted(set(groups) - allowed)
    if unknown:
        raise ValueError(f"--groups may contain only macro,hist; got {unknown}")
    return groups


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/release_hcs.yaml")
    parser.add_argument("--output-root", default="runs/paper_hcs_sweep")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--groups", default="macro,hist")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--phi", type=float, default=0.01)
    parser.add_argument("--macro-t-end", type=float, default=1000.0)
    parser.add_argument("--hist-t-end", type=float, default=400.0)
    parser.add_argument("--sample-start-t", type=float, default=150.0)
    parser.add_argument("--sample-end-t", type=float, default=400.0)
    parser.add_argument("--sample-delta-t", type=float, default=25.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    groups = _parse_groups(args.groups)
    domain = _parse_domain(args.domain)
    base_config = _load_base_config(args.config)
    tasks = build_tasks(
        args.output_root,
        seeds=args.seeds,
        groups=groups,
        dt=args.dt,
        macro_t_end=args.macro_t_end,
        hist_t_end=args.hist_t_end,
        sample_start_t=args.sample_start_t,
        sample_end_t=args.sample_end_t,
        sample_delta_t=args.sample_delta_t,
        models_dir=args.models_dir,
    )
    csv_path, json_path = _write_manifest(args.output_root, tasks)
    print(f"Wrote manifest: {csv_path}")
    print(f"Wrote manifest: {json_path}")
    print(
        f"Prepared {len(tasks)} task(s): "
        f"{sum(task.group == 'macro' for task in tasks)} macro, "
        f"{sum(task.group == 'hist' for task in tasks)} hist"
    )
    if args.dry_run:
        for task in tasks:
            print(
                f"[dry-run] {task.group} task={task.task_id} AR={task.AR:g} "
                f"alpha={task.alpha:.2f} seed={task.seed} -> {task.output_path}"
            )
        return

    task_dicts = [asdict(task) for task in tasks]
    if args.workers <= 1:
        results = [
            _run_task(
                task, base_config, args.models_dir, domain, args.phi,
                args.skip_existing, args.force
            )
            for task in task_dicts
        ]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    _run_task, task, base_config, args.models_dir, domain,
                    args.phi, args.skip_existing, args.force
                )
                for task in task_dicts
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{result['status']}] task={result['task_id']} "
                    f"{result['output_path']}",
                    flush=True,
                )
    ran = sum(result["status"] == "ran" for result in results)
    skipped = sum(result["status"] == "skipped" for result in results)
    print(f"Finished paper sweep: ran={ran}, skipped={skipped}")


if __name__ == "__main__":
    main()
