#!/usr/bin/env python3
"""Diagnostics and DSMC-LAMMPS overlay for AR=2.0 0D USF sweep."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

if __package__ is None or __package__ == "":
    THIS_FILE = Path(__file__).resolve()
    DSMC0D_ROOT = THIS_FILE.parents[2]
    if str(DSMC0D_ROOT) not in sys.path:
        sys.path.insert(0, str(DSMC0D_ROOT))

from src.postprocessing.analysis import load_dsmc_results, load_pressure_results
from src.simulation.particle import compute_particle_params


@dataclass
class CaseSummary:
    alpha: float
    case_dir: str
    result_file: str
    pressure_file: str
    t_end: float
    steady_start: float
    settle_05: float
    settle_10: float
    settle_15: float
    pxx: float
    pyy: float
    pzz: float
    pxy: float
    ttr_mean: float
    trot_mean: float
    tmix_mean: float
    ratio_mean: float


def average_last_fraction(arr: np.ndarray, frac: float) -> float:
    n = len(arr)
    if n == 0:
        return float("nan")
    i0 = int((1.0 - frac) * n)
    i0 = max(0, min(i0, n - 1))
    return float(np.mean(arr[i0:]))


def tail_slice(arr: np.ndarray, frac: float) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr
    i0 = int((1.0 - frac) * n)
    i0 = max(0, min(i0, n - 1))
    return arr[i0:]


def thin_series(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    step = max(1, int(np.ceil(len(x) / max_points)))
    return x[::step], y[::step]


def moving_average(arr: np.ndarray, window: int = 11) -> np.ndarray:
    n = len(arr)
    if n < 3:
        return arr.copy()
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return arr.copy()
    kernel = np.ones(w, dtype=float) / float(w)
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def settle_time(
    time: np.ndarray,
    series: np.ndarray,
    tail_frac: float = 0.2,
    tol_frac: float = 0.10,
    remain_frac: float = 0.85,
    hold_frac: float = 0.02,
) -> float:
    n = len(time)
    tail_n = max(10, int(round(n * tail_frac)))
    ref = float(np.mean(series[-tail_n:]))
    smooth = moving_average(series, 11)
    tol = tol_frac * abs(ref)
    inside = np.abs(smooth - ref) <= tol
    hold_n = max(10, int(round(hold_frac * n)))

    for i in range(max(0, n - tail_n - hold_n)):
        if not inside[i]:
            continue
        if np.mean(inside[i:i + hold_n]) < 0.95:
            continue
        if np.mean(inside[i:]) >= remain_frac:
            return float(time[i])
    return float("nan")


def choose_steady_start(time: np.ndarray, tmix: np.ndarray) -> tuple[float, float, float, float]:
    settle_05 = settle_time(time, tmix, tol_frac=0.05)
    settle_10 = settle_time(time, tmix, tol_frac=0.10)
    settle_15 = settle_time(time, tmix, tol_frac=0.15)
    fallback = float(time[max(0, int(0.8 * len(time)))])
    for val in (settle_05, settle_10, settle_15):
        if not np.isnan(val):
            return val, settle_05, settle_10, settle_15
    return fallback, settle_05, settle_10, settle_15


def result_paths(case_dir: Path, alpha: float) -> tuple[Path, Path]:
    tag = int(round(alpha * 100.0))
    result_file = case_dir / "results" / f"AR2_COR{tag}_USF_R1.txt"
    pressure_file = case_dir / "results" / f"AR2_COR{tag}_USF_R1_pressure.txt"
    return result_file, pressure_file


def summarize_case(case_dir: Path) -> CaseSummary:
    with open(case_dir / "config.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    alpha = float(cfg["system"]["alpha"])
    result_file, pressure_file = result_paths(case_dir, alpha)
    t, tau, ttr, trot, tmix = load_dsmc_results(str(result_file))
    pressure = load_pressure_results(str(pressure_file))

    params = compute_particle_params(cfg)
    lx, ly, lz = cfg["system"]["domain"]
    vol = float(lx * ly * lz)
    phi = float(cfg["system"]["phi"])
    npart = int(np.ceil(phi * vol / params.volume))
    nden = npart / vol

    steady_start, settle_05, settle_10, settle_15 = choose_steady_start(t, tmix)
    mask_t = t >= steady_start
    mask_p = pressure["t"] >= steady_start

    ttr_mean = float(np.mean(ttr[mask_t]))
    trot_mean = float(np.mean(trot[mask_t]))
    tmix_mean = float(np.mean(tmix[mask_t]))
    ratio_mean = ttr_mean / max(trot_mean, 1.0e-30)

    pij = pressure["pij"][mask_p]
    p_mean = np.mean(pij, axis=0)
    denom = nden * ttr_mean

    return CaseSummary(
        alpha=alpha,
        case_dir=str(case_dir),
        result_file=str(result_file),
        pressure_file=str(pressure_file),
        t_end=float(t[-1]),
        steady_start=float(steady_start),
        settle_05=float(settle_05),
        settle_10=float(settle_10),
        settle_15=float(settle_15),
        pxx=float(p_mean[0, 0] / denom),
        pyy=float(p_mean[1, 1] / denom),
        pzz=float(p_mean[2, 2] / denom),
        pxy=float(p_mean[0, 1] / denom),
        ttr_mean=ttr_mean,
        trot_mean=trot_mean,
        tmix_mean=tmix_mean,
        ratio_mean=ratio_mean,
    )


def list_case_dirs(base_dir: Path) -> list[Path]:
    cases = []
    for path in sorted(base_dir.iterdir()):
        if path.is_dir() and path.name.startswith("alpha_"):
            cases.append(path)
    return cases


def collect_lammps_summary(base_dir: Path, shared_e: set[float], frac: float) -> list[dict[str, float]]:
    rows = []
    for folder in sorted(base_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("e_"):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        if round(e_val, 2) not in shared_e:
            continue
        shear = np.atleast_2d(np.loadtxt(folder / "shear_stats.dat", comments="#"))
        temp = np.atleast_2d(np.loadtxt(folder / "temperature_stats.dat", comments="#"))
        time = temp[:, 0]
        ttr = temp[:, 1]
        trot = temp[:, 2]
        tmix = (3.0 * ttr + 2.0 * trot) / 5.0
        steady_start, settle_05, settle_10, settle_15 = choose_steady_start(time, tmix)
        rows.append({
            "e": e_val,
            "Pxx*": average_last_fraction(shear[:, 1], frac),
            "Pyy*": average_last_fraction(shear[:, 2], frac),
            "Pzz*": average_last_fraction(shear[:, 3], frac),
            "Pxy*": average_last_fraction(shear[:, 4], frac),
            "Ttr_mean": average_last_fraction(ttr, frac),
            "Trot_mean": average_last_fraction(trot, frac),
            "Tmix_mean": average_last_fraction(tmix, frac),
            "Ttr_over_Trot": average_last_fraction(ttr, frac) / max(average_last_fraction(trot, frac), 1.0e-30),
            "folder": str(folder),
            "steady_start": steady_start,
            "settle_05": settle_05,
            "settle_10": settle_10,
            "settle_15": settle_15,
        })
    rows.sort(key=lambda row: row["e"])
    return rows


def write_summary_csv(rows: list[CaseSummary], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "alpha", "t_end", "steady_start", "settle_05", "settle_10", "settle_15",
            "Pxx_star", "Pyy_star", "Pzz_star", "Pxy_star",
            "Ttr_mean", "Trot_mean", "Tmix_mean", "Ttr_over_Trot",
        ])
        for r in rows:
            writer.writerow([
                f"{r.alpha:.2f}", r.t_end, r.steady_start, r.settle_05, r.settle_10, r.settle_15,
                r.pxx, r.pyy, r.pzz, r.pxy,
                r.ttr_mean, r.trot_mean, r.tmix_mean, r.ratio_mean,
            ])


def write_overlay_csv(dsmc_rows: list[CaseSummary], lammps_rows: list[dict[str, float]], out_path: Path) -> None:
    dsmc_by_e = {round(r.alpha, 2): r for r in dsmc_rows}
    lammps_by_e = {round(r["e"], 2): r for r in lammps_rows}
    shared_e = sorted(set(dsmc_by_e) & set(lammps_by_e))

    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "e",
            "Pxx*_LAMMPS", "Pxx*_DSMC",
            "Pyy*_LAMMPS", "Pyy*_DSMC",
            "Pzz*_LAMMPS", "Pzz*_DSMC",
            "Pxy*_LAMMPS", "Pxy*_DSMC",
            "Ttr_over_Trot_LAMMPS", "Ttr_over_Trot_DSMC",
            "steady_start_LAMMPS", "steady_start_DSMC",
        ])
        for e_val in shared_e:
            lrow = lammps_by_e[e_val]
            drow = dsmc_by_e[e_val]
            writer.writerow([
                f"{e_val:.2f}",
                lrow["Pxx*"], drow.pxx,
                lrow["Pyy*"], drow.pyy,
                lrow["Pzz*"], drow.pzz,
                lrow["Pxy*"], drow.pxy,
                lrow["Ttr_over_Trot"], drow.ratio_mean,
                lrow["steady_start"],
                drow.steady_start,
            ])


def plot_stress(rows: list[CaseSummary], out_path: Path) -> None:
    e = np.array([r.alpha for r in rows])
    comps = ["pxx", "pyy", "pzz", "pxy"]
    titles = [r"$P_{xx}^*$", r"$P_{yy}^*$", r"$P_{zz}^*$", r"$P_{xy}^*$"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharex=True)
    for ax, comp, title in zip(axes.ravel(), comps, titles):
        y = np.array([getattr(r, comp) for r in rows])
        ax.plot(e, y, "s-", lw=1.9, ms=5.5, color="tab:green")
        ax.set_title(title)
        ax.set_ylabel("Reduced Stress")
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Coefficient of Restitution e")
    axes[1, 1].set_xlabel("Coefficient of Restitution e")
    fig.suptitle("Reduced Stress Tensor - DSMC 0D Spherocylinder USF (AR=2.0)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_temperature_diagnostics(rows: list[CaseSummary], rep_case: CaseSummary, out_path: Path) -> None:
    t, tau, ttr, trot, tmix = load_dsmc_results(rep_case.result_file)
    e = np.array([r.alpha for r in rows])
    ratio = np.array([r.ratio_mean for r in rows])

    mask = t >= rep_case.steady_start
    tail_t = t[mask]
    tail_ttr = ttr[mask]
    tail_trot = trot[mask]
    tail_tmix = tmix[mask]

    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.4))

    ax = axes[0]
    ax.plot(e, ratio, "s-", lw=1.9, ms=5.5, color="tab:orange")
    ax.set_xlabel("Coefficient of Restitution e")
    ax.set_ylabel(r"Steady $T_{tr}/T_{rot}$")
    ax.set_title(r"Asymptotic Temperature Ratio - AR=2.0")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axvline(rep_case.steady_start, color="0.2", lw=1.1, alpha=0.8, label="steady start")
    tx, yy = thin_series(t, ttr, 1600)
    ax.plot(tx, yy, lw=1.3, color="tab:blue", label=r"$T_{tr}$")
    tx, yy = thin_series(t, trot, 1600)
    ax.plot(tx, yy, lw=1.3, color="tab:orange", label=r"$T_{rot}$")
    tx, yy = thin_series(t, tmix, 1600)
    ax.plot(tx, yy, "--", lw=1.8, color="tab:green", label=r"$(3T_{tr}+2T_{rot})/5$")
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Temperature")
    ax.set_title(f"Temperature Evolution (e={rep_case.alpha:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.4)

    ax = axes[2]
    offset = tail_t - tail_t[0]
    tx, yy = thin_series(offset, tail_ttr, 900)
    ax.plot(tx, yy, lw=1.5, color="tab:blue", label=r"$T_{tr}$")
    tx, yy = thin_series(offset, tail_trot, 900)
    ax.plot(tx, yy, lw=1.5, color="tab:orange", label=r"$T_{rot}$")
    tx, yy = thin_series(offset, tail_tmix, 900)
    ax.plot(tx, yy, "--", lw=1.8, color="tab:green", label=r"$(3T_{tr}+2T_{rot})/5$")
    ax.axhline(np.mean(tail_ttr), color="tab:blue", lw=1.0, alpha=0.45)
    ax.axhline(np.mean(tail_trot), color="tab:orange", lw=1.0, alpha=0.45)
    ax.axhline(np.mean(tail_tmix), color="tab:green", lw=1.0, alpha=0.45)
    ax.set_xlabel("Physical time offset within steady window")
    ax.set_ylabel("Temperature")
    ax.set_title("Steady Window Used For Averages")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.5)

    fig.suptitle("Temperature Diagnostics - DSMC 0D Spherocylinder USF (AR=2.0)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_stress_overlay(dsmc_rows: list[CaseSummary], lammps_rows: list[dict[str, float]], out_path: Path) -> None:
    e_l = np.array([row["e"] for row in lammps_rows])
    e_d = np.array([row.alpha for row in dsmc_rows])
    comps = ["pxx", "pyy", "pzz", "pxy"]
    titles = [r"$P_{xx}^*$", r"$P_{yy}^*$", r"$P_{zz}^*$", r"$P_{xy}^*$"]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), sharex=True)
    for ax, comp, title in zip(axes.ravel(), comps, titles):
        y_l = np.array([
            row["Pxx*"] if comp == "pxx" else
            row["Pyy*"] if comp == "pyy" else
            row["Pzz*"] if comp == "pzz" else
            row["Pxy*"]
            for row in lammps_rows
        ])
        y_d = np.array([getattr(row, comp) for row in dsmc_rows])
        ax.plot(e_l, y_l, "o-", lw=1.8, ms=5.0, color="tab:orange", label="LAMMPS")
        ax.plot(e_d, y_d, "s-", lw=1.9, ms=5.2, color="tab:green", label="DSMC 0D")
        ax.set_title(title)
        ax.set_ylabel("Reduced Stress")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8.5)
    axes[1, 0].set_xlabel("Coefficient of Restitution e")
    axes[1, 1].set_xlabel("Coefficient of Restitution e")
    fig.suptitle("Reduced Stress Overlay - Spherocylinder USF (AR=2.0)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_lammps_temperatures(case_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temp = np.atleast_2d(np.loadtxt(os.path.join(case_dir, "temperature_stats.dat"), comments="#"))
    time = temp[:, 0]
    ttr = temp[:, 1]
    trot = temp[:, 2]
    tmix = (3.0 * ttr + 2.0 * trot) / 5.0
    return time, ttr, trot, tmix


def plot_temperature_overlay(
    dsmc_rows: list[CaseSummary],
    lammps_rows: list[dict[str, float]],
    rep_case: CaseSummary,
    lammps_case_dir: str,
    out_path: Path,
    lammps_frac: float,
) -> None:
    e_l = np.array([row["e"] for row in lammps_rows])
    e_d = np.array([row.alpha for row in dsmc_rows])
    ratio_l = np.array([row["Ttr_over_Trot"] for row in lammps_rows])
    ratio_d = np.array([row.ratio_mean for row in dsmc_rows])

    t_d, _, ttr_d, trot_d, tmix_d = load_dsmc_results(rep_case.result_file)
    t_l, ttr_l, trot_l, tmix_l = load_lammps_temperatures(lammps_case_dir)
    lammps_ref = next(row for row in lammps_rows if round(row["e"], 2) == round(rep_case.alpha, 2))
    steady_start_l = float(lammps_ref["steady_start"])
    mask_d = t_d >= rep_case.steady_start
    mask_l = t_l >= steady_start_l

    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.6))

    ax = axes[0]
    ax.plot(e_l, ratio_l, "o-", lw=1.8, ms=5.0, color="tab:orange", label="LAMMPS")
    ax.plot(e_d, ratio_d, "s-", lw=1.9, ms=5.2, color="tab:green", label="DSMC 0D")
    ax.set_xlabel("Coefficient of Restitution e")
    ax.set_ylabel(r"Steady $T_{tr}/T_{rot}$")
    ax.set_title(r"Asymptotic Temperature Ratio - AR=2.0")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.8)

    ax = axes[1]
    ax.axvspan(rep_case.steady_start, t_d[-1], color="#ccebc5", alpha=0.7, lw=0.0, label="DSMC steady window")
    tx, yy = thin_series(t_d, ttr_d, 1600)
    ax.plot(tx, yy, lw=1.4, color="tab:blue", label=r"$T_{tr}$")
    tx, yy = thin_series(t_d, trot_d, 1600)
    ax.plot(tx, yy, lw=1.4, color="tab:orange", label=r"$T_{rot}$")
    tx, yy = thin_series(t_d, tmix_d, 1600)
    ax.plot(tx, yy, "--", lw=1.8, color="tab:green", label=r"$(3T_{tr}+2T_{rot})/5$")
    ax.axvline(rep_case.steady_start, color="0.2", lw=1.1, alpha=0.85)
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Temperature")
    ax.set_title(f"DSMC Steady Window (e={rep_case.alpha:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.0)

    ax = axes[2]
    ax.axvspan(steady_start_l, t_l[-1], color="#fde0c5", alpha=0.7, lw=0.0, label="LAMMPS steady window")
    tx, yy = thin_series(t_l, ttr_l, 1600)
    ax.plot(tx, yy, lw=1.4, color="tab:blue", label=r"$T_{tr}$")
    tx, yy = thin_series(t_l, trot_l, 1600)
    ax.plot(tx, yy, lw=1.4, color="tab:orange", label=r"$T_{rot}$")
    tx, yy = thin_series(t_l, tmix_l, 1600)
    ax.plot(tx, yy, "--", lw=1.8, color="tab:green", label=r"$(3T_{tr}+2T_{rot})/5$")
    ax.axvline(steady_start_l, color="0.2", lw=1.1, alpha=0.85)
    ax.set_xlabel("Physical time")
    ax.set_ylabel("Temperature")
    ax.set_title(f"LAMMPS Steady Window (e={rep_case.alpha:.2f})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8.0)

    fig.suptitle(
        "Temperature Overlay - Spherocylinder USF (AR=2.0)\n"
        "Separate actual-time steady windows for DSMC and LAMMPS",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="AR=2 0D DSMC vs LAMMPS USF diagnostics.")
    parser.add_argument("--sweep-dir", default="runs/AR2_usf_sweep")
    parser.add_argument("--lammps-dir", default="/home/muhammed/Documents/LAMMPS/runs/sphcyl_USF_AR2")
    parser.add_argument("--rep-e", type=float, default=0.80)
    parser.add_argument("--lammps-avg-frac", type=float, default=0.50)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    out_dir = sweep_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    dsmc_rows = []
    for case_dir in list_case_dirs(sweep_dir):
        dsmc_rows.append(summarize_case(case_dir))
    dsmc_rows.sort(key=lambda row: row.alpha)

    shared_e = {round(row.alpha, 2) for row in dsmc_rows}
    lammps_rows = collect_lammps_summary(Path(args.lammps_dir).resolve(), shared_e, args.lammps_avg_frac)
    shared_e = sorted(shared_e & {round(row["e"], 2) for row in lammps_rows})
    dsmc_rows = [row for row in dsmc_rows if round(row.alpha, 2) in shared_e]
    lammps_rows = [row for row in lammps_rows if round(row["e"], 2) in shared_e]

    rep_case = min(dsmc_rows, key=lambda row: abs(row.alpha - args.rep_e))
    lammps_case_dir = next(row["folder"] for row in lammps_rows if round(row["e"], 2) == round(rep_case.alpha, 2))

    write_summary_csv(dsmc_rows, out_dir / "dsmc0d_diagnostics_summary.csv")
    write_overlay_csv(dsmc_rows, lammps_rows, out_dir / "dsmc0d_lammps_overlay_summary.csv")
    plot_stress(dsmc_rows, out_dir / "reduced_stress_tensor_ar2_usf_dsmc0d.png")
    plot_temperature_diagnostics(dsmc_rows, rep_case, out_dir / "temperature_diagnostics_ar2_usf_dsmc0d.png")
    plot_stress_overlay(dsmc_rows, lammps_rows, out_dir / "reduced_stress_overlay_dsmc0d_lammps.png")
    plot_temperature_overlay(
        dsmc_rows,
        lammps_rows,
        rep_case,
        lammps_case_dir,
        out_dir / "temperature_overlay_dsmc0d_lammps.png",
        args.lammps_avg_frac,
    )

    print(f"Wrote figures and CSVs to: {out_dir}")


if __name__ == "__main__":
    main()
