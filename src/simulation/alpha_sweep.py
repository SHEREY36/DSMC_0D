import copy
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from src.simulation.collision import CollisionModels
from src.simulation.dsmc import run_simulation


def _normalize_alpha(alpha):
    return round(float(alpha), 3)


def _alpha_dirname(alpha):
    return f"alpha_{int(round(alpha * 100)):03d}"


def _default_alpha_values():
    return [_normalize_alpha(0.5 + 0.05 * i) for i in range(11)]


def _resolve_alpha_values(sweep_cfg):
    values = sweep_cfg.get("alpha_values")
    if not values:
        return _default_alpha_values()
    return [_normalize_alpha(v) for v in values]


def _resolve_eta_map(alpha_values, sweep_cfg):
    if "eta_values" in sweep_cfg and sweep_cfg["eta_values"] is not None:
        eta_values = [float(v) for v in sweep_cfg["eta_values"]]
        if len(eta_values) != len(alpha_values):
            raise ValueError(
                "calibration_sweep.eta_values length must match alpha_values length"
            )
        return {a: eta for a, eta in zip(alpha_values, eta_values)}

    eta_by_alpha = sweep_cfg.get("eta_by_alpha")
    if eta_by_alpha:
        eta_map = {}
        for k, v in eta_by_alpha.items():
            eta_map[_normalize_alpha(k)] = float(v)
        missing = [a for a in alpha_values if a not in eta_map]
        if missing:
            raise ValueError(
                f"Missing eta values for alpha(s): {missing}"
            )
        return eta_map

    default_eta = float(sweep_cfg.get("default_eta", 2.0))
    return {a: default_eta for a in alpha_values}


def _resolve_shear_rate_map(alpha_values, sweep_cfg):
    if "shear_rate_values" in sweep_cfg and sweep_cfg["shear_rate_values"] is not None:
        sr_values = [float(v) for v in sweep_cfg["shear_rate_values"]]
        if len(sr_values) != len(alpha_values):
            raise ValueError(
                "calibration_sweep.shear_rate_values length must match alpha_values length"
            )
        return {a: sr for a, sr in zip(alpha_values, sr_values)}

    shear_rates_by_alpha = sweep_cfg.get("shear_rates_by_alpha")
    if shear_rates_by_alpha:
        sr_map = {}
        for k, v in shear_rates_by_alpha.items():
            sr_map[_normalize_alpha(k)] = float(v)
        missing = [a for a in alpha_values if a not in sr_map]
        if missing:
            raise ValueError(
                f"Missing shear_rate values for alpha(s): {missing}"
            )
        return sr_map

    default_sr = float(sweep_cfg.get("default_shear_rate",
                                     sweep_cfg.get("shear_rate", 0.0)))
    return {a: default_sr for a in alpha_values}


def _resolve_active_alphas(alpha_values, sweep_cfg, cli_alphas=None):
    if cli_alphas:
        active = [_normalize_alpha(v) for v in cli_alphas]
    else:
        active_cfg = sweep_cfg.get("active_alphas", [])
        active = [_normalize_alpha(v) for v in active_cfg] if active_cfg else list(alpha_values)

    unknown = [a for a in active if a not in alpha_values]
    if unknown:
        raise ValueError(
            f"Requested alpha(s) not in sweep alpha_values: {unknown}"
        )
    return sorted(set(active))


def _build_case_config(base_config, sweep_cfg, alpha, eta, gdot=None):
    cfg = copy.deepcopy(base_config)
    output_root = sweep_cfg["output_root"]
    case_dir = os.path.join(output_root, _alpha_dirname(alpha))

    cfg["particle"]["AR"] = float(sweep_cfg.get("AR", cfg["particle"]["AR"]))
    cfg["system"]["alpha"] = float(alpha)
    cfg["system"]["eta"] = float(eta)
    cfg["time"]["t_end"] = float(sweep_cfg.get("t_end", cfg["time"]["t_end"]))
    cfg["time"]["equilibration_time"] = float(
        sweep_cfg.get("equilibration_time", cfg["time"].get("equilibration_time", 0.0))
    )

    if gdot is not None:
        cfg.setdefault('flow', {})
        cfg['flow']['shear_rate'] = float(gdot)

    if "dt" in sweep_cfg:
        cfg["time"]["dt"] = float(sweep_cfg["dt"])
    if "dtau" in sweep_cfg:
        cfg["time"]["dtau"] = float(sweep_cfg["dtau"])
    if "seeds" in sweep_cfg and sweep_cfg["seeds"] is not None:
        cfg["simulation"]["seeds"] = list(sweep_cfg["seeds"])

    cfg["simulation"]["output_dir"] = os.path.join(case_dir, "results")
    cfg.setdefault("postprocessing", {})
    cfg["postprocessing"]["results_dir"] = os.path.join(case_dir, "results")
    cfg["postprocessing"]["figures_dir"] = os.path.join(case_dir, "figures")
    cfg["postprocessing"]["sweep_root"] = output_root

    return cfg, case_dir


def prepare_sweep_cases(base_config, cli_alphas=None):
    sweep_cfg = base_config.get("calibration_sweep")
    if not sweep_cfg:
        raise ValueError(
            "Missing `calibration_sweep` section in config."
        )

    output_root = sweep_cfg["output_root"]
    os.makedirs(output_root, exist_ok=True)

    alpha_values = _resolve_alpha_values(sweep_cfg)
    eta_map = _resolve_eta_map(alpha_values, sweep_cfg)
    sr_map = _resolve_shear_rate_map(alpha_values, sweep_cfg)
    active_alphas = _resolve_active_alphas(
        alpha_values, sweep_cfg, cli_alphas=cli_alphas
    )

    case_configs = []
    run_case_configs = []
    for alpha in alpha_values:
        eta = eta_map[alpha]
        gdot = sr_map[alpha]
        cfg, case_dir = _build_case_config(base_config, sweep_cfg, alpha, eta, gdot=gdot)

        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(cfg["simulation"]["output_dir"], exist_ok=True)
        os.makedirs(cfg["postprocessing"]["figures_dir"], exist_ok=True)

        config_path = os.path.join(case_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        case_info = {
            "alpha": alpha,
            "eta": eta,
            "gdot": gdot,
            "case_dir": case_dir,
            "config_path": config_path,
        }
        case_configs.append(case_info)
        if alpha in active_alphas:
            run_case_configs.append(case_info)

    eta_csv = os.path.join(output_root, "eta_profile.csv")
    with open(eta_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["alpha", "eta", "gdot", "case_dir"])
        for c in case_configs:
            writer.writerow([f"{c['alpha']:.3f}", f"{c['eta']:.8f}",
                             f"{c['gdot']:.6f}", c["case_dir"]])

    return {
        "output_root": output_root,
        "all_cases": case_configs,
        "run_cases": run_case_configs,
        "active_alphas": active_alphas,
    }


def _run_single_seed(task):
    """Run one (alpha, seed) realization. Each call is an independent worker task.

    task keys: config_path, seed, realization_idx
    """
    config_path = task["config_path"]
    seed = task["seed"]
    realization_idx = task["realization_idx"]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config.get("simulation", {}).get("sphere_collision", False):
        models = None
    else:
        model_dir = config["preprocessing"]["model_output_dir"]
        gmm_npz = config["preprocessing"]["gmm"].get("gmm_cond_file")
        ftr_path = config["preprocessing"].get("ftr", {}).get("ftr_params_file")
        models = CollisionModels(model_dir, gmm_npz_path=gmm_npz, ftr_params_path=ftr_path)

    AR = config["particle"]["AR"]
    alpha = config["system"]["alpha"]
    flow_tag = "_USF" if config.get("flow", {}).get("mode") == "usf" else ""
    filename = f"AR{AR:.0f}_COR{int(alpha * 100)}{flow_tag}_R{realization_idx}.txt"
    output_path = os.path.join(config["simulation"]["output_dir"], filename)
    pressure_path = os.path.join(
        config["simulation"]["output_dir"],
        f"AR{AR:.0f}_COR{int(alpha * 100)}{flow_tag}_R{realization_idx}_pressure.txt",
    )

    os.makedirs(config["simulation"]["output_dir"], exist_ok=True)
    print(f"  [alpha={alpha:.2f} R{realization_idx} seed={seed}] starting...")
    run_simulation(config, models, seed, output_path, pressure_path)
    return config_path, seed


def run_prepared_cases(run_cases, workers=1):
    """Expand run_cases into per-(alpha, seed) tasks and run with the given worker count.

    workers=1  → serial execution
    workers=N  → up to N simultaneous (alpha, seed) simulations via ProcessPoolExecutor
    """
    tasks = []
    for case in run_cases:
        with open(case["config_path"]) as f:
            cfg = yaml.safe_load(f)
        seeds = cfg["simulation"]["seeds"]
        for i, seed in enumerate(seeds, start=1):
            tasks.append({
                "config_path": case["config_path"],
                "seed": seed,
                "realization_idx": i,
            })

    if not tasks:
        return []

    workers = int(workers)
    if workers <= 1:
        for task in tasks:
            _run_single_seed(task)
        return [t["config_path"] for t in tasks]

    completed = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single_seed, t): t for t in tasks}
        for future in as_completed(futures):
            completed.append(future.result())
    return [c[0] for c in completed]
