#!/usr/bin/env python3
"""Compare HCS cumulative collisions-per-particle across DSMC, MFIX, and LAMMPS.

Generates one figure per restitution coefficient for AR=2.0 spherocylinders.

Series shown on each figure:
  - MFIX         : line
  - DSMC         : markers only
  - LAMMPS (old) : markers only, using collision_events.dat
  - LAMMPS (new) : markers only, using collision_count.dat

Both LAMMPS counters are multiplied by 2/N so they are directly comparable to
DSMC tau = NColl / Np.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


LAMMPS_ROOT = "LAMMPS_data/HCS/calibrate_cpp/modeB_e_sweep2"
MFIX_ROOT = "MFIX_DEM_data/kT_1_kTr_1"
CALIB_DIR = "runs/calib_C_alpha"
C_TABLE_PATH = "models/C_alpha_table_AR20.json"
FIGURES_DIR = "figures"
DEFAULT_ALPHAS = (0.9, 0.8, 0.7)
DSMC_EQUIL_T = 2.0
T_MAX = 300.0
FALLBACK_LAMMPS_N = 2003


def set_aspect_one(ax):
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    if yr > 0:
        ax.set_aspect(xr / yr, adjustable="box")


def _format_ax(ax):
    ax.tick_params(axis="both", direction="in", which="both", right=True, top=True)
    ax.grid(True, linestyle="--", alpha=0.3)


def _prepend_origin_if_needed(x, y):
    if len(x) == 0:
        return x, y
    if x[0] <= 0.0:
        return x, y
    return np.insert(x, 0, 0.0), np.insert(y, 0, 0.0)


def _clip_time_window(x, y, t_max):
    mask = x <= t_max
    x = np.asarray(x)[mask]
    y = np.asarray(y)[mask]
    return x, y


def _marker_slice(x, y, n_target=55):
    if len(x) <= n_target:
        return x, y
    step = max(1, int(np.ceil(len(x) / n_target)))
    return x[::step], y[::step]


def _read_dt_from_log(folder):
    log_path = os.path.join(folder, "log.lammps")
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            match = re.search(r"dt=([0-9eE+\-\.]+)", line)
            if match:
                return float(match.group(1))
    raise RuntimeError(f"Could not find 'dt=...' in {log_path}")


def _read_npart_from_log(folder):
    log_path = os.path.join(folder, "log.lammps")
    with open(log_path, "r", encoding="utf-8") as fh:
        for line in fh:
            match = re.search(r"\bN=(\d+),\s*phi=", line)
            if match:
                return int(match.group(1))
    return FALLBACK_LAMMPS_N


def _load_lammps_counter(alpha, filename, t_max):
    alpha_int = int(round(alpha * 100))
    folder = os.path.join(LAMMPS_ROOT, f"e_{alpha_int:03d}")
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    raw = np.loadtxt(path, comments="#")
    dt = _read_dt_from_log(folder)
    n_part = _read_npart_from_log(folder)

    t = raw[:, 0] * dt
    tau = 2.0 * raw[:, 2] / float(n_part)
    t, tau = _clip_time_window(t, tau, t_max)
    return _prepend_origin_if_needed(t, tau)


def _mfix_folder(alpha):
    alpha_int = int(round(alpha * 100))
    stripped = str(alpha_int).rstrip("0") or "0"
    candidates = [
        f"Alpha_{alpha_int:03d}",
        f"Alpha_{alpha_int}",
        f"Alpha_{stripped.zfill(2)}",
        f"Alpha_{stripped}",
    ]
    for cand in candidates:
        folder = os.path.join(MFIX_ROOT, cand, "AR20")
        if os.path.isdir(folder):
            return folder
    raise FileNotFoundError(f"Could not resolve MFIX folder for alpha={alpha:.2f}")


def _load_mfix(alpha, t_max):
    path = os.path.join(_mfix_folder(alpha), "T.txt")
    raw = np.loadtxt(path)
    t = raw[:, 0]
    tau = raw[:, 1]
    t, tau = _clip_time_window(t, tau, t_max)
    return _prepend_origin_if_needed(t, tau)


def _load_dsmc(path, t_max):
    raw = np.loadtxt(path)
    t = raw[:, 0]
    tau = raw[:, 1]

    i_eq = int(np.searchsorted(t, DSMC_EQUIL_T))
    i_eq = min(i_eq, len(t) - 1)
    t = t[i_eq:] - t[i_eq]
    tau = tau[i_eq:] - tau[i_eq]
    t, tau = _clip_time_window(t, tau, t_max)
    return _prepend_origin_if_needed(t, tau)


def _find_calibrated_dsmc_path(alpha, c_table, seed=42):
    alpha_int = int(round(alpha * 100))
    target = float(c_table[f"({alpha:.3f}, 2.0)"])
    pattern = os.path.join(CALIB_DIR, f"calib_C*_a{alpha_int:03d}_s{seed}.txt")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No DSMC calibration files found for alpha={alpha:.2f}")

    candidates = []
    for path in matches:
        match = re.search(r"calib_C([0-9.]+)_a", os.path.basename(path))
        if match:
            c_val = float(match.group(1))
            candidates.append((abs(c_val - target), c_val, path))
    if not candidates:
        raise FileNotFoundError(f"Could not parse DSMC calibration filenames for alpha={alpha:.2f}")

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _axis_limits(series_list, x_max, pad_frac=0.04):
    y_all = np.concatenate([np.asarray(y) for _, y in series_list if len(y) > 0])
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    y_rng = y_max - y_min
    y_pad = pad_frac * y_rng if y_rng > 0 else max(0.1, 0.03 * max(abs(y_max), 1.0))
    return (0.0, x_max), (max(0.0, y_min - y_pad), y_max + y_pad)


def plot_alpha(alpha, c_table, output_dir, t_max):
    dsmc_path = _find_calibrated_dsmc_path(alpha, c_table)
    dsmc_t, dsmc_tau = _load_dsmc(dsmc_path, t_max)
    mfix_t, mfix_tau = _load_mfix(alpha, t_max)
    lmp_old_t, lmp_old_tau = _load_lammps_counter(alpha, "collision_events.dat", t_max)
    lmp_new_t, lmp_new_tau = _load_lammps_counter(alpha, "collision_count.dat", t_max)

    dsmc_t_m, dsmc_tau_m = _marker_slice(dsmc_t, dsmc_tau, n_target=40)
    lmp_old_t_m, lmp_old_tau_m = _marker_slice(lmp_old_t, lmp_old_tau, n_target=55)
    lmp_new_t_m, lmp_new_tau_m = _marker_slice(lmp_new_t, lmp_new_tau, n_target=55)

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    ax.plot(mfix_t, mfix_tau, color="black", linewidth=1.8, label="MFIX", zorder=2)
    ax.plot(
        dsmc_t_m, dsmc_tau_m,
        linestyle="None", marker="o", markersize=4.6,
        color="#1f77b4", label="DSMC", zorder=4,
    )
    ax.plot(
        lmp_old_t_m, lmp_old_tau_m,
        linestyle="None", marker="s", markersize=4.2,
        markerfacecolor="none", markeredgewidth=1.2,
        color="#d95f02", label="LAMMPS (old)", zorder=3,
    )
    ax.plot(
        lmp_new_t_m, lmp_new_tau_m,
        linestyle="None", marker="^", markersize=4.8,
        markerfacecolor="none", markeredgewidth=1.2,
        color="#1b9e77", label="LAMMPS (new)", zorder=3,
    )

    xlim, ylim = _axis_limits(
        [
            (mfix_t, mfix_tau),
            (dsmc_t, dsmc_tau),
            (lmp_old_t, lmp_old_tau),
            (lmp_new_t, lmp_new_tau),
        ],
        x_max=t_max,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    set_aspect_one(ax)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Cumulative collisions per particle, $\tau$")
    ax.set_title(f"HCS Spherocylinder AR=2, e={alpha:.1f}")
    _format_ax(ax)
    ax.legend(loc="upper left", frameon=True)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"hcs_cpp_compare_e{int(round(alpha * 100)):03d}.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path, dsmc_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot HCS cumulative collisions-per-particle comparisons for AR=2."
    )
    parser.add_argument(
        "--alphas",
        default="0.9,0.8,0.7",
        help="Comma-separated restitution coefficients to plot.",
    )
    parser.add_argument(
        "--output-dir",
        default=FIGURES_DIR,
        help="Directory for the output figures.",
    )
    parser.add_argument(
        "--t-max",
        type=float,
        default=T_MAX,
        help="Maximum physical time shown on the plots.",
    )
    args = parser.parse_args()

    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]

    with open(C_TABLE_PATH, "r", encoding="utf-8") as fh:
        c_table = json.load(fh)

    os.makedirs(args.output_dir, exist_ok=True)

    for alpha in alphas:
        out_path, dsmc_path = plot_alpha(alpha, c_table, args.output_dir, args.t_max)
        print(f"Saved {out_path} using {os.path.basename(dsmc_path)}")


if __name__ == "__main__":
    main()
