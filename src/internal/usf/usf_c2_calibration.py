import json
import math
import os
import re
from pathlib import Path

import numpy as np
import yaml

from src.postprocessing.analysis import load_dsmc_results, load_pressure_results
from src.simulation.particle import (
    compute_particle_params,
    model_ar_tag,
    result_ar_tag,
)


_C_ALPHA_KEY_RE = re.compile(
    r"\(\s*([0-9.+\-eE]+)\s*,\s*([0-9.+\-eE]+)\s*\)"
)


def _as_float(value):
    return float(np.asarray(value, dtype=float))


def _tail_mask(values, tail_fraction):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    start = int(math.floor((1.0 - float(tail_fraction)) * values.size))
    start = max(0, min(start, values.size - 1))
    mask = np.zeros(values.size, dtype=bool)
    mask[start:] = True
    return mask, start


def _plateau_stats_mask(values, stats_fraction=0.50, threshold=5.0e-4,
                        smooth_window=51):
    """Return the steady averaging mask used for DSMC baseline statistics."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty 1-D array")
    n = values.size
    if n < 5:
        return np.ones(n, dtype=bool), 0, 0

    window = int(smooth_window)
    if window > n:
        window = n if n % 2 == 1 else n - 1
    if window < 5:
        smooth = values
    else:
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(values, kernel, mode="same")

    scale = max(abs(float(np.nanmean(smooth))), 1.0e-30)
    rel_step = np.abs(np.gradient(smooth)) / scale
    plateau_candidates = np.flatnonzero(rel_step < float(threshold))
    plateau_idx = int(plateau_candidates[0]) if plateau_candidates.size else 0

    stats_idx = max(plateau_idx, int(math.floor((1.0 - stats_fraction) * n)))
    stats_idx = max(0, min(stats_idx, n - 1))
    mask = np.zeros(n, dtype=bool)
    mask[stats_idx:] = True
    return mask, stats_idx, plateau_idx


def reduced_kinetic_a2(Pk_reduced):
    """Compute a2 from a reduced kinetic pressure tensor Pk/(n*Ttr)."""
    Pk = np.asarray(Pk_reduced, dtype=float)
    if Pk.shape == (3, 3):
        dev = Pk - np.eye(3)
        a2 = float(np.trace(dev @ dev) / 8.0)
        return 0.0 if not np.isfinite(a2) else max(0.0, a2)
    if Pk.ndim == 3 and Pk.shape[1:] == (3, 3):
        dev = Pk - np.eye(3)[None, :, :]
        a2 = np.einsum("nij,nji->n", dev, dev) / 8.0
        a2 = np.where(np.isfinite(a2), a2, 0.0)
        return np.maximum(a2, 0.0)
    raise ValueError(f"Pk_reduced must have shape (3, 3) or (N, 3, 3), got {Pk.shape}")


def rank0_ftr(C_alpha, theta):
    theta = max(float(theta), 1.0e-10)
    return float(C_alpha) * 3.0 * theta / (3.0 * theta + 2.0)


def infer_C2_from_theta_gap(theta_dsmc, theta_lammps, a2_steady, C_alpha,
                            theta_probe=None, probe_delta=None,
                            min_a2=1.0e-12, min_abs_chi=1.0e-12):
    """Invert the USF theta gap using a DSMC-measured probe sensitivity."""
    theta_dsmc = float(theta_dsmc)
    theta_lammps = float(theta_lammps)
    a2_steady = float(a2_steady)
    C_alpha = float(C_alpha)
    theta_probe = None if theta_probe is None else float(theta_probe)
    probe_delta = None if probe_delta is None else float(probe_delta)

    f_tr0 = rank0_ftr(C_alpha, theta_dsmc)
    delta_theta = theta_lammps - theta_dsmc
    status = "ok"
    valid = True
    C2 = 0.0
    chi = float("nan")

    required = [theta_dsmc, theta_lammps, a2_steady, C_alpha, f_tr0]
    if theta_probe is not None:
        required.append(theta_probe)
    if probe_delta is not None:
        required.append(probe_delta)
    if not all(np.isfinite(x) for x in required):
        status = "nonfinite_input"
        valid = False
    elif a2_steady <= float(min_a2):
        status = "near_zero_a2"
        valid = False
    elif theta_probe is None or probe_delta is None:
        status = "missing_probe_sensitivity"
        valid = False
    elif abs(probe_delta) <= 1.0e-30:
        status = "zero_probe_delta"
        valid = False
    else:
        chi = (theta_probe - theta_dsmc) / probe_delta
        if not np.isfinite(chi) or abs(chi) <= float(min_abs_chi):
            status = "invalid_chi"
            valid = False
        else:
            C2 = delta_theta / (chi * a2_steady)
            if not np.isfinite(C2):
                status = "nonfinite_C2"
                valid = False
                C2 = 0.0

    return {
        "C2": float(C2),
        "valid": bool(valid),
        "status": status,
        "delta_theta": float(delta_theta),
        "f_tr0": float(f_tr0),
        "chi": float(chi),
        "theta_probe": None if theta_probe is None else float(theta_probe),
        "probe_delta": None if probe_delta is None else float(probe_delta),
    }


def load_C_alpha_table(path):
    with open(path, "r") as f:
        payload = json.load(f)
    rows = []
    if isinstance(payload, dict) and "rows" in payload:
        for row in payload["rows"]:
            rows.append((float(row["alpha"]), float(row["AR"]), float(row["C_alpha"])))
    elif isinstance(payload, dict):
        for key, value in payload.items():
            match = _C_ALPHA_KEY_RE.match(str(key))
            if match is None:
                continue
            rows.append((float(match.group(1)), float(match.group(2)), float(value)))
    else:
        raise ValueError(f"Unsupported C_alpha table format: {path}")
    if not rows:
        raise ValueError(f"No C_alpha rows found in {path}")
    return rows


def lookup_C_alpha(rows, alpha, AR):
    alpha = float(alpha)
    AR = float(AR)
    pairs = sorted((a, c) for a, ar, c in rows if np.isclose(ar, AR, atol=1.0e-12))
    if not pairs:
        raise KeyError(f"C_alpha table has no rows for AR={AR:g}")
    alphas = np.array([item[0] for item in pairs], dtype=float)
    values = np.array([item[1] for item in pairs], dtype=float)
    idx = np.flatnonzero(np.isclose(alphas, alpha, atol=1.0e-12))
    if idx.size:
        return float(values[idx[0]])
    if alpha <= alphas[0]:
        return float(values[0])
    if alpha >= alphas[-1]:
        return float(values[-1])
    return float(np.interp(alpha, alphas, values))


def load_lammps_theta(case_dir, tail_fraction=0.30):
    path = Path(case_dir) / "temperature_stats.dat"
    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 3:
        raise ValueError(f"LAMMPS temperature file has too few columns: {path}")
    T_tr = data[:, 1]
    T_rot = data[:, 2]
    theta = T_tr / np.where(T_rot > 0.0, T_rot, np.nan)
    mask, start = _tail_mask(theta, tail_fraction)
    return {
        "theta": float(np.nanmean(theta[mask])),
        "theta_std": float(np.nanstd(theta[mask])),
        "n_samples": int(np.count_nonzero(mask)),
        "tail_start_index": int(start),
        "path": str(path),
    }


def load_lammps_stress(case_dir, tail_fraction=0.30):
    path = Path(case_dir) / "shear_stats.dat"
    if not path.exists():
        return None
    data = np.loadtxt(path, comments="#")
    data = np.atleast_2d(data)
    if data.shape[1] < 5:
        return None
    mask, start = _tail_mask(data[:, 0], tail_fraction)
    return {
        "Pxx": float(np.mean(data[mask, 1])),
        "Pyy": float(np.mean(data[mask, 2])),
        "Pzz": float(np.mean(data[mask, 3])),
        "Pxy": float(np.mean(data[mask, 4])),
        "n_samples": int(np.count_nonzero(mask)),
        "tail_start_index": int(start),
        "path": str(path),
    }


def _candidate_output_files(results_dir, AR, alpha, realization_index):
    tag = int(round(float(alpha) * 100.0))
    ar_tags = list(dict.fromkeys([result_ar_tag(AR), f"AR{int(round(float(AR)))}"]))
    candidates = []
    for ar_tag in ar_tags:
        stem = f"{ar_tag}_COR{tag}_USF_R{realization_index}"
        candidates.append((
            Path(results_dir) / f"{stem}.txt",
            Path(results_dir) / f"{stem}_pressure.txt",
        ))
    return candidates


def _number_density(config, params):
    phi = float(config["system"]["phi"])
    lx, ly, lz = config["system"]["domain"]
    volume = float(lx) * float(ly) * float(lz)
    Np = int(math.ceil(phi * volume / params.volume))
    return float(Np / volume), Np


def load_dsmc_usf_case(case_dir, stats_fraction=0.50, plateau_threshold=5.0e-4,
                       smooth_window=51):
    """Load one DSMC USF alpha case and compute steady theta and kinetic a2."""
    case_dir = Path(case_dir)
    cfg_path = case_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing DSMC config: {cfg_path}")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    alpha = float(config["system"]["alpha"])
    params = compute_particle_params(config)
    AR = float(params.AR)
    n_density, Np = _number_density(config, params)
    results_dir = Path(config.get("simulation", {}).get("output_dir", case_dir / "results"))
    if not results_dir.is_absolute():
        results_dir = case_dir / "results"

    seed_ids = config.get("simulation", {}).get("seeds", [])
    n_expected = len(seed_ids) if seed_ids else 4
    seed_rows = []
    for realization in range(1, n_expected + 1):
        existing = [
            (temp_path, pressure_path)
            for temp_path, pressure_path in _candidate_output_files(results_dir, AR, alpha, realization)
            if temp_path.exists() and pressure_path.exists()
        ]
        if not existing:
            continue
        temp_path, pressure_path = existing[0]
        t, _, T_tr, T_rot, T_total = load_dsmc_results(str(temp_path))
        pressure = load_pressure_results(str(pressure_path))

        mask_t, stats_idx, plateau_idx = _plateau_stats_mask(
            T_total,
            stats_fraction=stats_fraction,
            threshold=plateau_threshold,
            smooth_window=smooth_window,
        )
        stats_t_start = float(t[stats_idx])
        mask_p = pressure["t"] >= stats_t_start
        if not np.any(mask_p):
            continue

        T_tr_ss = float(np.mean(T_tr[mask_t]))
        T_rot_ss = float(np.mean(T_rot[mask_t]))
        if T_rot_ss <= 0.0 or T_tr_ss <= 0.0:
            continue
        Pk_mean = np.mean(pressure["pij_k"][mask_p], axis=0)
        Pk_reduced = Pk_mean / (n_density * T_tr_ss)
        a2 = reduced_kinetic_a2(Pk_reduced)

        Ptot_mean = np.mean(pressure["pij"][mask_p], axis=0)
        Ptot_reduced = Ptot_mean / (n_density * T_tr_ss)
        seed_rows.append({
            "realization": int(realization),
            "theta": float(T_tr_ss / T_rot_ss),
            "T_tr": T_tr_ss,
            "T_rot": T_rot_ss,
            "a2_kinetic": float(a2),
            "Pk_reduced": Pk_reduced.tolist(),
            "Ptot_reduced": Ptot_reduced.tolist(),
            "stats_t_start": stats_t_start,
            "stats_index": int(stats_idx),
            "plateau_index": int(plateau_idx),
            "n_pressure_samples": int(np.count_nonzero(mask_p)),
            "temperature_file": str(temp_path),
            "pressure_file": str(pressure_path),
        })

    if not seed_rows:
        raise FileNotFoundError(f"No usable DSMC result/pressure files in {case_dir}")

    theta_values = np.array([row["theta"] for row in seed_rows], dtype=float)
    a2_values = np.array([row["a2_kinetic"] for row in seed_rows], dtype=float)
    return {
        "alpha": alpha,
        "AR": AR,
        "theta": float(np.mean(theta_values)),
        "theta_std": float(np.std(theta_values, ddof=1)) if theta_values.size > 1 else 0.0,
        "a2_steady": float(np.mean(a2_values)),
        "a2_std": float(np.std(a2_values, ddof=1)) if a2_values.size > 1 else 0.0,
        "n_seeds": int(len(seed_rows)),
        "n_expected_seeds": int(n_expected),
        "Np": int(Np),
        "number_density": float(n_density),
        "seed_rows": seed_rows,
        "case_dir": str(case_dir),
    }


def _parse_alpha_from_case_dir(path):
    name = Path(path).name
    match = re.match(r"alpha_([0-9]+(?:\.[0-9]+)?)$", name)
    if match is None:
        return None
    raw = float(match.group(1))
    return raw / 100.0 if raw > 1.5 else raw


def lammps_case_for_alpha(lammps_root, alpha):
    tag = int(round(float(alpha) * 100.0))
    return Path(lammps_root) / f"e_{tag:03d}"


def build_usf_C2_table(
    dsmc_root,
    lammps_root,
    C_alpha_table_file,
    AR,
    probe_root,
    probe_delta=None,
    output_path=None,
    stats_fraction=0.50,
    plateau_threshold=5.0e-4,
    smooth_window=51,
    lammps_tail_fraction=0.30,
    min_a2=1.0e-12,
    min_abs_chi=1.0e-12,
):
    """Build a USF theta-gap calibrated C2 table for one aspect ratio."""
    AR = float(AR)
    dsmc_root = Path(dsmc_root)
    probe_root = Path(probe_root)
    lammps_root = Path(lammps_root)
    C_alpha_rows = load_C_alpha_table(C_alpha_table_file)

    rows = []
    for case_dir in sorted(dsmc_root.glob("alpha_*")):
        alpha = _parse_alpha_from_case_dir(case_dir)
        if alpha is None:
            continue
        dsmc = load_dsmc_usf_case(
            case_dir,
            stats_fraction=stats_fraction,
            plateau_threshold=plateau_threshold,
            smooth_window=smooth_window,
        )
        lammps_case = lammps_case_for_alpha(lammps_root, alpha)
        lammps = load_lammps_theta(lammps_case, tail_fraction=lammps_tail_fraction)
        lammps_stress = load_lammps_stress(lammps_case, tail_fraction=lammps_tail_fraction)
        probe_case = probe_root / case_dir.name
        probe = load_dsmc_usf_case(
            probe_case,
            stats_fraction=stats_fraction,
            plateau_threshold=plateau_threshold,
            smooth_window=smooth_window,
        )
        case_probe_delta = probe_delta
        if case_probe_delta is None:
            with open(probe_case / "config.yaml", "r") as f:
                probe_cfg = yaml.safe_load(f)
            case_probe_delta = probe_cfg.get("simulation", {}).get(
                "ftr_rank0_probe_delta"
            )
        C_alpha = lookup_C_alpha(C_alpha_rows, alpha, AR)
        inverted = infer_C2_from_theta_gap(
            dsmc["theta"], lammps["theta"], dsmc["a2_steady"], C_alpha,
            theta_probe=probe["theta"], probe_delta=case_probe_delta,
            min_a2=min_a2, min_abs_chi=min_abs_chi,
        )
        row = {
            "alpha": float(alpha),
            "AR": float(AR),
            "C2": float(inverted["C2"]),
            "valid": bool(inverted["valid"]),
            "status": inverted["status"],
            "theta_DSMC": float(dsmc["theta"]),
            "theta_DSMC_std": float(dsmc["theta_std"]),
            "theta_LAMMPS": float(lammps["theta"]),
            "theta_LAMMPS_std": float(lammps["theta_std"]),
            "theta_probe": float(probe["theta"]),
            "theta_probe_std": float(probe["theta_std"]),
            "delta_theta": float(inverted["delta_theta"]),
            "a2_steady": float(dsmc["a2_steady"]),
            "a2_steady_std": float(dsmc["a2_std"]),
            "C_alpha": float(C_alpha),
            "f_tr0": float(inverted["f_tr0"]),
            "chi_DSMC": float(inverted["chi"]),
            "probe_delta": float(case_probe_delta),
            "n_dsmc_seeds": int(dsmc["n_seeds"]),
            "n_probe_seeds": int(probe["n_seeds"]),
            "n_lammps_samples": int(lammps["n_samples"]),
            "dsmc_source_folder": str(case_dir),
            "probe_source_folder": str(probe_case),
            "lammps_source_folder": str(lammps_case),
            "dsmc_seed_rows": dsmc["seed_rows"],
            "probe_seed_rows": probe["seed_rows"],
            "lammps_stress_diagnostic": lammps_stress,
        }
        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No DSMC alpha folders found under {dsmc_root}")
    rows.sort(key=lambda item: item["alpha"])
    payload = {
        "metadata": {
            "model": "rank2_C2",
            "method": "usf_theta_gap_measured_sensitivity",
            "deployed_formula": (
                "f_tr = C_alpha * (1 + C2(alpha, AR) * a2_live) "
                "* 3*theta/(3*theta + 2)"
            ),
            "calibration_a2": "steady kinetic pressure tensor from DSMC USF baseline",
            "runtime_a2": "live particle-velocity invariant in dsmc.py",
            "sensitivity": (
                "chi_DSMC = (theta_probe - theta_DSMC) / probe_delta"
            ),
            "f_tr0": "C_alpha * 3*theta_DSMC/(3*theta_DSMC + 2)",
            "dsmc_root": str(dsmc_root),
            "probe_root": str(probe_root),
            "probe_delta": None if probe_delta is None else float(probe_delta),
            "lammps_root": str(lammps_root),
            "C_alpha_table_file": str(C_alpha_table_file),
            "AR": float(AR),
            "tail_policy": {
                "dsmc_stats_fraction": float(stats_fraction),
                "dsmc_plateau_threshold": float(plateau_threshold),
                "dsmc_smooth_window": int(smooth_window),
                "lammps_tail_fraction": float(lammps_tail_fraction),
                "seed_averaging": "per-seed steady values first, then arithmetic mean",
            },
            "pressure_normalization": "Pk_reduced = Pk_mean/(n*T_tr_steady)",
            "invalid_row_policy": (
                "Rows with near-zero a2 or singular measured sensitivity are retained "
                "with valid=false and C2=0.0"
            ),
        },
        "rows": rows,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    return payload


def default_paths_for_AR(AR, runs_root="runs", lammps_usf_root="LAMMPS_data/USF2",
                         models_dir="models", probe_tag="probe_m010"):
    result_tag = result_ar_tag(AR)
    model_tag = model_ar_tag(AR)
    run_candidates = [
        Path(runs_root) / f"{result_tag}_usf_vss_rank2",
        Path(runs_root) / f"{model_tag}_usf_vss_rank2",
    ]
    probe_candidates = [
        Path(runs_root) / f"{result_tag}_usf_vss_rank2_{probe_tag}",
        Path(runs_root) / f"{model_tag}_usf_vss_rank2_{probe_tag}",
    ]
    lammps_candidates = [
        Path(lammps_usf_root) / result_tag,
        Path(lammps_usf_root) / model_tag,
    ]
    dsmc_root = next((path for path in run_candidates if path.exists()), run_candidates[0])
    probe_root = next((path for path in probe_candidates if path.exists()), probe_candidates[0])
    lammps_root = next((path for path in lammps_candidates if path.exists()), lammps_candidates[0])
    return {
        "dsmc_root": dsmc_root,
        "probe_root": probe_root,
        "lammps_root": lammps_root,
        "C_alpha_table_file": Path(models_dir) / f"C_alpha_table_{model_tag}.json",
        "output_path": Path(models_dir) / f"C2_table_{model_tag}.json",
    }
