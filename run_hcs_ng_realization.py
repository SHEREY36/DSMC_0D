#!/usr/bin/env python3
"""Run one HCS non-Gaussian paper-campaign realization.

This entry point is intended for Slurm job arrays: one process, one seed, one
(AR, alpha) case.  Slurm owns the parallelism.
"""
import argparse
import copy
import csv
import os
from pathlib import Path

import yaml

from src.simulation.collision import CollisionModels
from src.simulation.dsmc import run_simulation


ARS = [1.0, 1.1, 1.5, 2.0, 2.5, 3.0]
ALPHAS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
SEEDS = [
    42, 123, 456, 789, 1001, 2024, 3141, 2718, 5772, 8119,
    10103, 12109, 14153, 16127, 18131, 20143, 22147, 24151, 26161, 28163,
    30169, 32189, 34211, 36217, 38219, 40231, 42239, 44249, 46261, 48271,
    50273, 52289, 54311, 56333, 58337, 60343, 62347, 64381, 66383, 68411,
    70423, 72431, 74453, 76463, 78467, 80473, 82483, 84499, 86501, 88513,
    90523, 92531, 94543, 96557, 98561, 100573, 102587, 104593, 106619, 108631,
    110641, 112657, 114671, 116681, 118687, 120691, 122699, 124703, 126713, 128717,
    130729, 132739, 134743, 136751, 138763, 140779, 142789, 144791, 146807, 148829,
    150833, 152837, 154841, 156859, 158869, 160873, 162887, 164893, 166909, 168929,
    170933, 172933, 174949, 176959, 178963, 180979, 182983, 184997, 187003, 189017,
]

MODEL_LABELS = {
    1.1: "AR11",
    1.5: "AR15",
    2.0: "AR20",
    2.5: "AR25",
    3.0: "AR30",
}


def ar_folder_label(AR):
    return f"AR{int(round(float(AR) * 100)):03d}"


def alpha_folder_label(alpha):
    return f"alpha_{int(round(float(alpha) * 100)):03d}"


def decode_task_id(task_id):
    task_id = int(task_id)
    n_alpha = len(ALPHAS)
    n_seed = len(SEEDS)
    ar_index = task_id // (n_alpha * n_seed)
    rem = task_id % (n_alpha * n_seed)
    alpha_index = rem // n_seed
    seed_index = rem % n_seed
    if ar_index >= len(ARS):
        raise ValueError(f"Task id {task_id} is outside 0-{total_tasks() - 1}")
    return ARS[ar_index], ALPHAS[alpha_index], SEEDS[seed_index], seed_index + 1


def total_tasks():
    return len(ARS) * len(ALPHAS) * len(SEEDS)


def _model_label_for_ar(AR):
    AR = float(AR)
    if AR not in MODEL_LABELS:
        raise ValueError(f"No spherocylinder model mapping for AR={AR}")
    return MODEL_LABELS[AR]


def _maybe_existing(path):
    return str(path) if path.exists() else None


def _require_existing(path, description):
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return str(path)


def build_campaign_config(base_config, AR, alpha, seed, output_root,
                          domain=None, t_end=None, tau_end=None,
                          dt=None, sample_start_tau=None,
                          sample_end_tau=None, sample_delta_tau=None,
                          hcs_rescale=True):
    cfg = copy.deepcopy(base_config)
    AR = float(AR)
    alpha = float(alpha)
    model_dir = Path(cfg["preprocessing"]["model_output_dir"])
    ar_dir = ar_folder_label(AR)
    alpha_dir = alpha_folder_label(alpha)
    output_dir = Path(output_root) / ar_dir / alpha_dir / "results"

    cfg["particle"]["AR"] = AR
    cfg["system"]["alpha"] = alpha
    cfg["system"]["kTt"] = 1.0
    cfg["system"]["kTr"] = 1.0
    cfg["system"]["phi"] = 0.01
    cfg["system"]["domain"] = domain or [100, 100, 100]
    cfg["system"]["C_alpha"] = None

    cfg["time"]["dt"] = 0.05 if dt is None else float(dt)
    cfg["time"]["dtau"] = (
        5.0 if sample_delta_tau is None else float(sample_delta_tau)
    )
    cfg["time"]["t_end"] = 30000.0 if t_end is None else float(t_end)
    cfg["time"]["tau_end"] = 1500.0 if tau_end is None else float(tau_end)
    cfg["time"]["equilibration_time"] = 0.0

    cfg.setdefault("flow", {})
    cfg["flow"]["mode"] = "hcs"
    cfg["flow"]["shear_rate"] = 0.0

    cfg.setdefault("simulation", {})
    cfg["simulation"]["seeds"] = [int(seed)]
    cfg["simulation"]["output_dir"] = str(output_dir)
    cfg["simulation"]["sphere_collision"] = (AR == 1.0)
    cfg["simulation"]["angular_transport_model"] = "current"
    cfg["simulation"]["angular_transport_probability_override"] = None
    cfg["simulation"]["hcs_rescale_temperature"] = bool(hcs_rescale)
    cfg["simulation"]["hcs_rescale_reference"] = "initial"
    cfg["simulation"].pop("use_zr_eff", None)
    cfg["simulation"].pop("use_stress_transport_weight", None)
    cfg["simulation"].pop("stress_transport_weight_file", None)
    cfg["simulation"].pop("ctc_angular_file", None)

    cfg.setdefault("diagnostics", {}).setdefault("non_gaussian", {})
    sample_start = 500.0 if sample_start_tau is None else float(sample_start_tau)
    sample_end = 1500.0 if sample_end_tau is None else float(sample_end_tau)
    sample_delta = 5.0 if sample_delta_tau is None else float(sample_delta_tau)
    cfg["diagnostics"]["non_gaussian"].update({
        "enabled": True,
        "start_tau": sample_start,
        "sample_start_tau": sample_start,
        "sample_end_tau": sample_end,
        "sample_delta_tau": sample_delta,
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
    })

    cfg.setdefault("preprocessing", {}).setdefault("gmm", {})
    cfg.setdefault("preprocessing", {}).setdefault("ftr", {})
    cfg.setdefault("calibration", {})

    if AR == 1.0:
        cfg["preprocessing"]["gmm"].pop("gmm_cond_file", None)
        cfg["preprocessing"]["ftr"]["ftr_params_file"] = ""
        cfg["calibration"].pop("C_alpha_table_file", None)
    else:
        model_label = _model_label_for_ar(AR)
        cfg["preprocessing"]["gmm"]["gmm_cond_file"] = _require_existing(
            model_dir / f"gmm_cond_{model_label}.npz",
            f"GMM conditional model for AR={AR:g}",
        )
        cfg["calibration"]["C_alpha_table_file"] = _require_existing(
            model_dir / f"C_alpha_table_{model_label}.json",
            f"C_alpha table for AR={AR:g}",
        )

        ftr_path = _maybe_existing(model_dir / f"ftr_params_{model_label}_r100.json")
        cfg["preprocessing"]["ftr"]["ftr_params_file"] = ftr_path or ""

    return cfg, output_dir


def build_output_paths(output_dir, AR, alpha, realization_index):
    ar_label = ar_folder_label(AR)
    alpha_label = int(round(float(alpha) * 100))
    stem = f"{ar_label}_COR{alpha_label:03d}_R{int(realization_index):03d}"
    output_path = Path(output_dir) / f"{stem}.txt"
    pressure_path = Path(output_dir) / f"{stem}_pressure.txt"
    return output_path, pressure_path


def load_models(config):
    if config.get("simulation", {}).get("sphere_collision", False):
        print("Sphere collision mode: skipping empirical model loading.")
        return None

    model_dir = config["preprocessing"]["model_output_dir"]
    gmm_npz = config["preprocessing"]["gmm"].get("gmm_cond_file")
    ftr_path = config["preprocessing"].get("ftr", {}).get("ftr_params_file")
    c_alpha_path = config.get("calibration", {}).get("C_alpha_table_file")
    vss_alpha_eff_path = config.get("simulation", {}).get(
        "vss_alpha_eff_table_file"
    )
    print(f"Loading models from {model_dir}...")
    return CollisionModels(
        model_dir, gmm_npz_path=gmm_npz, ftr_params_path=ftr_path,
        c_alpha_path=c_alpha_path, vss_alpha_eff_path=vss_alpha_eff_path
    )


def ensure_manifest(output_root):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "campaign_manifest.csv"
    try:
        with open(manifest_path, "x", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id", "AR", "alpha", "seed", "realization_index",
                "output_dir", "status_expected",
            ])
            for task_id in range(total_tasks()):
                AR, alpha, seed, realization_index = decode_task_id(task_id)
                output_dir = (
                    output_root / ar_folder_label(AR)
                    / alpha_folder_label(alpha) / "results"
                )
                writer.writerow([
                    task_id, f"{AR:g}", f"{alpha:.2f}", seed,
                    realization_index, output_dir, "pending",
                ])
    except FileExistsError:
        pass
    return manifest_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one HCS non-Gaussian campaign realization"
    )
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--AR", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--realization-index", type=int)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--output-root", default="runs/hcs_ng_long_window")
    parser.add_argument(
        "--domain", default=None,
        help="Optional comma-separated domain override, e.g. 20,20,20"
    )
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--tau-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--start-tau", type=float, default=None)
    parser.add_argument("--sample-start-tau", type=float, default=None)
    parser.add_argument("--sample-end-tau", type=float, default=None)
    parser.add_argument("--sample-delta-tau", type=float, default=None)
    parser.add_argument(
        "--no-hcs-rescale", action="store_true",
        help="Disable HCS temperature rescaling for debugging comparisons."
    )
    parser.add_argument("--write-manifest-only", action="store_true")
    parser.add_argument("--print-seeds", action="store_true")
    parser.add_argument("--print-total-tasks", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.print_seeds:
        print(" ".join(str(seed) for seed in SEEDS))
        return
    if args.print_total_tasks:
        print(total_tasks())
        return
    if args.write_manifest_only:
        path = ensure_manifest(args.output_root)
        print(f"Wrote manifest: {path}")
        return

    if args.task_id is not None:
        AR, alpha, seed, realization_index = decode_task_id(args.task_id)
    else:
        missing = [
            name for name, value in [
                ("--AR", args.AR),
                ("--alpha", args.alpha),
                ("--seed", args.seed),
                ("--realization-index", args.realization_index),
            ]
            if value is None
        ]
        if missing:
            raise ValueError(
                "Provide --task-id or all explicit run arguments: "
                + ", ".join(missing)
            )
        AR = args.AR
        alpha = args.alpha
        seed = args.seed
        realization_index = args.realization_index

    ensure_manifest(args.output_root)
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)
    domain = None
    if args.domain:
        domain = [float(x.strip()) for x in args.domain.split(",") if x.strip()]
        if len(domain) != 3:
            raise ValueError("--domain must have exactly three comma-separated values")
    sample_start_tau = (
        args.sample_start_tau
        if args.sample_start_tau is not None else args.start_tau
    )
    config, output_dir = build_campaign_config(
        base_config, AR, alpha, seed, args.output_root,
        domain=domain, t_end=args.t_end, tau_end=args.tau_end, dt=args.dt,
        sample_start_tau=sample_start_tau,
        sample_end_tau=args.sample_end_tau,
        sample_delta_tau=args.sample_delta_tau,
        hcs_rescale=not args.no_hcs_rescale
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path, pressure_path = build_output_paths(
        output_dir, AR, alpha, realization_index
    )
    print(
        f"Running HCS NG realization: AR={AR:g}, alpha={alpha:.2f}, "
        f"seed={seed}, R={realization_index}, output={output_dir}"
    )
    models = load_models(config)
    run_simulation(
        config, models, seed, str(output_path), str(pressure_path)
    )


if __name__ == "__main__":
    main()
