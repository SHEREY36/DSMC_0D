#!/usr/bin/env python3
"""Publication-quality USF overlay plots.

Produces three figures:
  1. steady_diagnostics.png  — T_total vs time for all DSMC alpha cases
  2. stress_overlay_2x2.png  — 2×2 panel: Pxx* Pyy* Pzz* Pxy* vs alpha
  3. temperature_ratio.png   — theta* = T_trans/T_rot vs alpha

Usage (from project root):
    python run_usf_final_plots.py
"""
from __future__ import annotations

import argparse
import math
import sys
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import yaml

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.postprocessing.analysis import load_dsmc_results, load_pressure_results
from src.simulation.particle import compute_particle_params


# ---------------------------------------------------------------------------
# Plateau detection (from animate.ipynb, adapted for noisy DSMC data)
# ---------------------------------------------------------------------------

def _smooth(arr, window):
    """Centered moving average with edge-padding."""
    n = len(arr)
    w = min(window, n if n % 2 == 1 else max(1, n - 1))
    if w < 2:
        return arr.copy()
    kernel = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(arr, (pad, pad), mode="edge"), kernel, mode="valid")[:n]


def _detect_plateau(series, threshold=1e-3, smooth_window=51):
    """Find (start_idx, end_idx) of the longest consecutive plateau.

    A plateau is a run of consecutive indices where the point-to-point
    relative change in the smoothed signal is below `threshold`.

    Falls back to the second half of the series if no plateau is found.
    """
    s = _smooth(np.asarray(series, dtype=float), smooth_window)
    # avoid division by zero
    denom = np.where(np.abs(s[:-1]) > 1e-30, np.abs(s[:-1]), 1e-30)
    rel_change = np.abs(np.diff(s)) / denom

    idx = np.where(rel_change < threshold)[0]
    if len(idx) == 0:
        n = len(series)
        return n // 2, n - 1

    groups = [
        list(map(itemgetter(1), g))
        for _, g in groupby(enumerate(idx), lambda ix: ix[0] - ix[1])
    ]
    plateau = max(groups, key=len)
    return int(plateau[0]), int(plateau[-1])


def _stats_mask(series, stats_frac=0.50, threshold=5e-4, smooth_window=51):
    """Return a boolean mask for the last `stats_frac` of the longest plateau.

    E.g. plateau [200, 900], stats_frac=0.5 → mask True for [550, 900].
    Also returns (stats_start_idx, plateau_start_idx) for diagnostic plotting.
    """
    p_start, p_end = _detect_plateau(series, threshold=threshold,
                                     smooth_window=smooth_window)
    plateau_len = p_end - p_start
    stats_start = p_start + int((1.0 - stats_frac) * plateau_len)
    stats_start = max(p_start, stats_start)

    mask = np.zeros(len(series), dtype=bool)
    mask[stats_start: p_end + 1] = True
    return mask, stats_start, p_start


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _thin(x, y, max_pts=1200):
    if len(x) <= max_pts:
        return x, y
    step = max(1, int(math.ceil(len(x) / max_pts)))
    return x[::step], y[::step]


def _axis_limits(values_list, pad=0.07):
    all_vals = [v for vs in values_list for v in vs if np.isfinite(v)]
    if not all_vals:
        return None, None
    lo, hi = min(all_vals), max(all_vals)
    rng = hi - lo if hi > lo else abs(hi) * 0.1 + 0.05
    return lo - pad * rng, hi + pad * rng


# ---------------------------------------------------------------------------
# Kinetic Theory (first Sonine approximation, user-supplied)
# ---------------------------------------------------------------------------

def _beta_0(alpha):
    return ((1 + alpha) / 2) * (1 - (1 - alpha) / 3)


def _zeta_0(alpha):
    return (1 - alpha**2) * (5 / 12)


def _a_steady(beta, zeta):
    return np.sqrt((3 * zeta) / (2 * beta)) * (beta + zeta)


def garzo_dufty_spheres(alpha_arr):
    a = np.asarray(alpha_arr, dtype=float)
    b = _beta_0(a)
    z = _zeta_0(a)
    a_s = _a_steady(b, z)
    Pxx = (b + 3 * z) / (b + z)
    Pyy = b / (b + z)
    Pzz = Pyy.copy()
    Pxy = -(b / (b + z)**2) * a_s
    return dict(alpha=a, Pxx=Pxx, Pyy=Pyy, Pzz=Pzz, Pxy=Pxy)


# ---------------------------------------------------------------------------
# DSMC loader
# ---------------------------------------------------------------------------

def load_dsmc_case(case_dir: Path) -> dict | None:
    cfg_path = case_dir / "config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    alpha = float(cfg["system"]["alpha"])
    phi = float(cfg["system"]["phi"])
    lx, ly, lz = cfg["system"]["domain"]
    volsys = float(lx * ly * lz)

    params = compute_particle_params(cfg)
    Np = math.ceil(phi * volsys / params.volume)
    n_dsmc = Np / volsys

    results_dir = case_dir / "results"
    tag = int(round(alpha * 100))

    seed_pxx, seed_pyy, seed_pzz, seed_pxy, seed_theta = [], [], [], [], []
    # per-seed data for diagnostics
    diag_t, diag_T, diag_theta = [], [], []
    diag_stats_start, diag_plateau_start = [], []

    ar_tag = int(round(params.AR))
    n_seeds = len(cfg.get("simulation", {}).get("seeds", [None] * 4))
    for ri in range(1, n_seeds + 1):
        tp = results_dir / f"AR{ar_tag}_COR{tag}_USF_R{ri}.txt"
        pp = results_dir / f"AR{ar_tag}_COR{tag}_USF_R{ri}_pressure.txt"
        if not (tp.exists() and pp.exists()):
            continue
        try:
            t, _, T_tr, T_rot, T_tot = load_dsmc_results(str(tp))
            pres = load_pressure_results(str(pp))
        except Exception:
            continue

        t_p = pres["t"]
        pij = pres["pij"]   # (N,3,3) total

        # --- plateau-based stats window on temperature ---
        mask_t, stats_idx, plat_idx = _stats_mask(
            T_tot, stats_frac=0.50, threshold=5e-4, smooth_window=51
        )
        stats_t_start = float(t[stats_idx])
        plat_t_start  = float(t[plat_idx])

        diag_t.append(t)
        diag_T.append(T_tot)
        diag_theta.append(T_tr / np.where(T_rot > 0, T_rot, np.nan))
        diag_stats_start.append(stats_t_start)
        diag_plateau_start.append(plat_t_start)

        # apply same time cut-off to pressure grid
        mask_p = t_p >= stats_t_start

        T_tr_ss  = float(np.mean(T_tr[mask_t]))
        T_rot_ss = float(np.mean(T_rot[mask_t]))
        denom = n_dsmc * T_tr_ss

        P_mean = np.mean(pij[mask_p], axis=0)
        seed_pxx.append(P_mean[0, 0] / denom)
        seed_pyy.append(P_mean[1, 1] / denom)
        seed_pzz.append(P_mean[2, 2] / denom)
        seed_pxy.append(P_mean[0, 1] / denom)
        seed_theta.append(T_tr_ss / T_rot_ss if T_rot_ss > 0 else float("nan"))

    if not seed_pxx:
        return None

    return dict(
        alpha=alpha,
        Pxx_mean=float(np.mean(seed_pxx)),  Pxx_std=float(np.std(seed_pxx)),
        Pyy_mean=float(np.mean(seed_pyy)),  Pyy_std=float(np.std(seed_pyy)),
        Pzz_mean=float(np.mean(seed_pzz)),  Pzz_std=float(np.std(seed_pzz)),
        Pxy_mean=float(np.mean(seed_pxy)),  Pxy_std=float(np.std(seed_pxy)),
        theta_mean=float(np.nanmean(seed_theta)),
        theta_std=float(np.nanstd(seed_theta)),
        # diagnostics
        t_series=diag_t,
        T_series=diag_T,
        theta_series=diag_theta,
        stats_starts=diag_stats_start,
        plateau_starts=diag_plateau_start,
        n_seeds=len(seed_pxx),
    )


def load_all_dsmc(sweep_dir: Path) -> list[dict]:
    cases = []
    for d in sorted(sweep_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("alpha_")):
            continue
        c = load_dsmc_case(d)
        if c is not None:
            cases.append(c)
    cases.sort(key=lambda x: x["alpha"])
    return cases


# ---------------------------------------------------------------------------
# LAMMPS loaders — both use last-30 % tail average, matching the reference
# scripts plot_sphcyl_vs_sphere.py (average_last_fraction) and plot_dilute.py
# (tail_mean).  No plateau detection is applied to LAMMPS data.
# ---------------------------------------------------------------------------

def _lammps_tail_mean(arr: np.ndarray, frac: float = 0.30) -> float:
    """Mean of the last `frac` fraction of `arr` (same as plot_dilute.py tail_mean)."""
    n = len(arr)
    i0 = max(0, min(int((1.0 - frac) * n), n - 1))
    return float(np.mean(arr[i0:]))


def load_lammps_sphcyl(base_dir: Path, tail_frac: float = 0.30) -> list[dict]:
    """Load LAMMPS spherocylinder shear + temperature stats.

    Follows plot_sphcyl_vs_sphere.py:
      - shear_stats.dat  cols 1-4 → Pxx*, Pyy*, Pzz*, Pxy*
      - temperature_stats.dat cols 1,2 → Ttr, Trot  → theta = Ttr/Trot
    All averaged over the last `tail_frac` of each file independently.
    """
    rows = []
    for folder in sorted(base_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0

        ss_path = folder / "shear_stats.dat"
        tp_path = folder / "temperature_stats.dat"
        if not (ss_path.exists() and tp_path.exists()):
            continue

        ss  = np.atleast_2d(np.loadtxt(ss_path,  comments="#"))
        tmp = np.atleast_2d(np.loadtxt(tp_path,  comments="#"))

        ttr  = _lammps_tail_mean(tmp[:, 1], tail_frac)
        trot = _lammps_tail_mean(tmp[:, 2], tail_frac)
        rows.append(dict(
            e=e_val,
            Pxx=_lammps_tail_mean(ss[:, 1], tail_frac),
            Pyy=_lammps_tail_mean(ss[:, 2], tail_frac),
            Pzz=_lammps_tail_mean(ss[:, 3], tail_frac),
            Pxy=_lammps_tail_mean(ss[:, 4], tail_frac),
            theta=ttr / (trot + 1e-30),  # T_tr / T_rot, same as plot_sphcyl_vs_sphere.py
        ))
    rows.sort(key=lambda r: r["e"])
    return rows


def load_lammps_spheres(base_dir: Path, tail_frac: float = 0.30,
                        elastic_early_rows: int = 850) -> list[dict]:
    """Load LAMMPS sphere shear stats.

    Follows plot_dilute.py (tail_mean frac=0.3) and plot_sphcyl_vs_sphere.py
    (average_last_fraction frac=0.3).  shear_stats.dat cols 1-4 → Pxx* … Pxy*.

    For the elastic case (alpha=1.0) USF has no steady state — the system
    undergoes a string instability at long times.  The physically meaningful
    window is the early plateau before the instability sets in, so we average
    over the first `elastic_early_rows` rows instead of the tail.
    """
    rows = []
    for folder in sorted(base_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        ss_path = folder / "shear_stats.dat"
        if not ss_path.exists():
            continue
        ss = np.atleast_2d(np.loadtxt(ss_path, comments="#"))

        if abs(e_val - 1.0) < 1e-3:
            # Elastic: use early plateau before string instability
            window = ss[:elastic_early_rows]
            rows.append(dict(
                e=e_val,
                Pxx=float(np.mean(window[:, 1])),
                Pyy=float(np.mean(window[:, 2])),
                Pzz=float(np.mean(window[:, 3])),
                Pxy=float(np.mean(window[:, 4])),
            ))
        else:
            rows.append(dict(
                e=e_val,
                Pxx=_lammps_tail_mean(ss[:, 1], tail_frac),
                Pyy=_lammps_tail_mean(ss[:, 2], tail_frac),
                Pzz=_lammps_tail_mean(ss[:, 3], tail_frac),
                Pxy=_lammps_tail_mean(ss[:, 4], tail_frac),
            ))
    rows.sort(key=lambda r: r["e"])
    return rows


# ---------------------------------------------------------------------------
# LAMMPS nematic order & angular velocity loaders
# ---------------------------------------------------------------------------

def load_lammps_nematic(base_dir: Path, tail_frac: float = 0.30) -> list[dict]:
    """Load LAMMPS nematic order tensor data and compute scalar order parameter S.

    Reads nematic_tensor.dat (cols: TimeStep Qxx Qyy Qzz Qxy Qxz Qyz ...).
    Q is already the traceless tensor Q_ij = <u_i u_j - delta_ij/3>.
    S = largest eigenvalue of Q (per LAMMPS in.usf definition).

    Returns list of dicts: {e, S_mean, S_std} sorted by e.
    """
    rows = []
    for folder in sorted(base_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        nem_path = folder / "nematic_tensor.dat"
        if not nem_path.exists():
            continue

        data = np.atleast_2d(np.loadtxt(nem_path, comments="#"))
        n = len(data)
        i0 = max(0, int((1.0 - tail_frac) * n))
        tail = data[i0:]

        Qxx = tail[:, 1]; Qyy = tail[:, 2]; Qzz = tail[:, 3]
        Qxy = tail[:, 4]; Qxz = tail[:, 5]; Qyz = tail[:, 6]

        S_series = np.empty(len(tail))
        for k in range(len(tail)):
            Q = np.array([
                [Qxx[k], Qxy[k], Qxz[k]],
                [Qxy[k], Qyy[k], Qyz[k]],
                [Qxz[k], Qyz[k], Qzz[k]],
            ])
            S_series[k] = np.linalg.eigvalsh(Q)[-1]

        rows.append(dict(
            e=e_val,
            S_mean=float(np.mean(S_series)),
            S_std=float(np.std(S_series)),
        ))

    rows.sort(key=lambda r: r["e"])
    return rows


def load_lammps_angvel(base_dir: Path, tail_frac: float = 0.30) -> list[dict]:
    """Load LAMMPS angular velocity distribution statistics.

    Reads angular_velocity_stats.dat
    (cols: TimeStep omperp2_avg om2_tot_avg ompecz_avg ompecz2_avg ...).

    Computes:
      ratio     = <omega_pec_z^2> / <omega_perp^2>   (z-fraction of tumbling)
      norm_mean = <omega_pec_z>   / sqrt(<omega_perp^2>)  (normalised mean drift)

    Returns list of dicts: {e, omperp2, ompecz, ompecz2, ratio, norm_mean}.
    """
    rows = []
    for folder in sorted(base_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        av_path = folder / "angular_velocity_stats.dat"
        if not av_path.exists():
            continue

        data = np.atleast_2d(np.loadtxt(av_path, comments="#"))
        omperp2 = _lammps_tail_mean(data[:, 1], tail_frac)
        ompecz  = _lammps_tail_mean(data[:, 3], tail_frac)
        ompecz2 = _lammps_tail_mean(data[:, 4], tail_frac)

        ratio     = ompecz2 / (omperp2 + 1e-30)
        norm_mean = ompecz  / (np.sqrt(omperp2) + 1e-30)

        rows.append(dict(
            e=e_val,
            omperp2=omperp2,
            ompecz=ompecz,
            ompecz2=ompecz2,
            ratio=ratio,
            norm_mean=norm_mean,
        ))

    rows.sort(key=lambda r: r["e"])
    return rows


# ---------------------------------------------------------------------------
# NSP (spatially-resolved) DSMC loader
# ---------------------------------------------------------------------------

def _loadtxt_fortran(filepath):
    """Load Fortran D/d-format scientific notation into a numpy array."""
    from io import StringIO
    with open(filepath, "r") as fh:
        content = fh.read().replace("D", "E").replace("d", "e")
    data = np.loadtxt(StringIO(content))
    return np.atleast_2d(data)


def _parse_nsp_system_input(filepath):
    """Parse system_input.dat → dict with pip, dia, AR, rho, pmass, alpha."""
    with open(filepath, "r") as fh:
        content = fh.read().replace("D", "E").replace("d", "e")
    lines = content.strip().split("\n")
    pip   = int(lines[3].strip())
    vals5 = lines[5].split()
    dia, AR = float(vals5[0]), float(vals5[1])
    rho     = float(lines[6].strip())
    alpha   = float(lines[7].strip())
    lcycl   = dia * (AR - 1.0)
    pvol    = math.pi / 6.0 * dia ** 3 + math.pi * lcycl * (dia / 2.0) ** 2
    pmass   = pvol * rho
    return {"pip": pip, "dia": dia, "AR": AR, "rho": rho, "pmass": pmass, "alpha": alpha}


def load_nsp_dsmc_case(case_dir: Path) -> dict | None:
    """Load one NSP DSMC alpha directory.

    File layout (from output_mod.f90):
      tg.txt   cols: t, tau, T_rot, T_trans, T_total, ...
      Pijk.txt cols: Pxx_k, Pyy_k, Pzz_k, Pxy_k, Pxz_k, Pyz_k
      Pijc.txt raw:  Pxx_c, Pxy_c, Pxz_c, Pyy_c, Pyz_c, Pzz_c  → reordered [xx,yy,zz,xy,xz,yz]

    Pijc has 1 fewer row than Pijk (PREV_TIME==0 on first step).
    Reduced stress: P*_ij = pmass * P_total_ij / (pip * T_trans_mean).
    Pxx↔Pyy swap applied: NSP flow is in y; LAMMPS/KT convention is flow in x.
    Plateau detection on T_total using _stats_mask (same as 0D DSMC).
    """
    sys_path = case_dir / "system_input.dat"
    tg_path  = case_dir / "tg.txt"
    pk_path  = case_dir / "Pijk.txt"
    pc_path  = case_dir / "Pijc.txt"
    if not (sys_path.exists() and tg_path.exists() and pk_path.exists()):
        return None
    try:
        props = _parse_nsp_system_input(sys_path)
        tg    = _loadtxt_fortran(tg_path)
        pijk  = _loadtxt_fortran(pk_path)
    except Exception:
        return None

    tau   = tg[:, 1]
    T_rot = tg[:, 2]
    T_tr  = tg[:, 3]
    T_tot = tg[:, 4]

    # Reorder Pijc columns: file=[xx,xy,xz,yy,yz,zz] → [xx,yy,zz,xy,xz,yz]
    if pc_path.exists() and pc_path.stat().st_size > 0:
        try:
            pc_raw = _loadtxt_fortran(pc_path)
            pijc = np.column_stack([
                pc_raw[:, 0], pc_raw[:, 3], pc_raw[:, 5],
                pc_raw[:, 1], pc_raw[:, 2], pc_raw[:, 4],
            ])
        except Exception:
            pijc = None
    else:
        pijc = None

    # Align: Pijc has 1 fewer row → skip first row of Pijk and tg series
    n_k = pijk.shape[0]
    if pijc is not None:
        offset  = max(0, n_k - pijc.shape[0])
        P_total = pijk[offset:] + pijc   # (n_c, 6) [xx,yy,zz,xy,xz,yz]
    else:
        offset  = 0
        P_total = pijk
    T_tr_al  = T_tr[offset:]
    T_rot_al = T_rot[offset:]
    tau_al   = tau[offset:]
    T_tot_al = T_tot[offset:]

    # Plateau detection on T_total — same machinery as 0D DSMC
    mask_t, stats_idx, plat_idx = _stats_mask(
        T_tot_al, stats_frac=0.50, threshold=1e-3, smooth_window=11
    )
    stats_tau_start = float(tau_al[stats_idx])
    plat_tau_start  = float(tau_al[plat_idx])

    Ttr_mean = float(np.mean(T_tr_al[mask_t]))
    scale    = props["pmass"] / (props["pip"] * Ttr_mean)

    # P_total cols: 0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz
    # Apply Pxx↔Pyy swap: NSP flow=y → DSMC yy maps to plot Pxx (flow direction)
    Pxx = float(scale * np.mean(P_total[mask_t, 1]))
    Pyy = float(scale * np.mean(P_total[mask_t, 0]))
    Pzz = float(scale * np.mean(P_total[mask_t, 2]))
    Pxy = float(scale * np.mean(P_total[mask_t, 3]))

    Trot_mean = float(np.mean(T_rot_al[mask_t]))
    theta     = Ttr_mean / max(Trot_mean, 1e-30)

    return dict(
        alpha=props["alpha"],
        Pxx_mean=Pxx,  Pxx_std=0.0,
        Pyy_mean=Pyy,  Pyy_std=0.0,
        Pzz_mean=Pzz,  Pzz_std=0.0,
        Pxy_mean=Pxy,  Pxy_std=0.0,
        theta_mean=theta, theta_std=0.0,
        t_series=[tau_al],
        T_series=[T_tot_al],
        stats_starts=[stats_tau_start],
        plateau_starts=[plat_tau_start],
        n_seeds=1,
    )


def load_nsp_dsmc_sweep(sweep_dir: Path) -> list[dict]:
    """Load all alpha cases from an NSP DSMC sweep directory.

    Subdirectory naming convention: integer alpha*100 (e.g. 50, 55, ..., 100).
    """
    cases = []
    for d in sweep_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            int(d.name)
        except ValueError:
            continue
        c = load_nsp_dsmc_case(d)
        if c is not None:
            cases.append(c)
    cases.sort(key=lambda x: x["alpha"])
    return cases


# ---------------------------------------------------------------------------
# Sphere NSP DSMC loader (Run_shear_NN format)
# ---------------------------------------------------------------------------

def load_nsphere_dsmc_case(case_dir: Path,
                            elastic_early_rows: int = 850) -> dict | None:
    """Load one sphere NSP DSMC case (Run_shear_NN/NNN format).

    system_input.dat layout (sphere, differs from spherocylinder):
      line 0:  Lx Ly Lz           (domain box lengths)
      line 3:  PIP                 (number of particles)
      line 7:  alpha

    Pijk.txt cols  : Pxx_k  Pyy_k  Pzz_k  Pxy_k  Pxz_k  Pyz_k
    Pijc.txt cols  : Pxx_c  Pxy_c  Pxz_c  Pyy_c  Pyz_c  Pzz_c
    tg.txt   col 2 : GRANULAR = sum_v^2/(3N) = T_trans

    Normalisation: P*_ij = (P_ij_k + P_ij_c) / (N * T_trans)
    No chi factor (dilute); no volume factor (cancels with density).

    Flow direction is y in the NSP sphere code → Pxx↔Pyy swap to match
    the LAMMPS/KT convention where flow is in x.

    For alpha=1 (elastic) there is no USF steady state (string instability).
    The early plateau (rows 0:elastic_early_rows) is used instead of
    plateau detection.
    """
    sys_path = case_dir / "system_input.dat"
    tg_path  = case_dir / "tg.txt"
    pk_path  = case_dir / "Pijk.txt"
    pc_path  = case_dir / "Pijc.txt"
    if not (sys_path.exists() and tg_path.exists() and pk_path.exists()):
        return None

    try:
        with open(sys_path) as fh:
            raw = fh.read().replace("D", "E").replace("d", "e")
        lines = raw.strip().split("\n")
        pip   = int(lines[3].strip())
        alpha = float(lines[7].strip())

        tg   = _loadtxt_fortran(str(tg_path))
        pijk = _loadtxt_fortran(str(pk_path))
    except Exception:
        return None

    if pc_path.exists() and pc_path.stat().st_size > 0:
        try:
            pc_raw = _loadtxt_fortran(str(pc_path))
            # reorder [xx,xy,xz,yy,yz,zz] → [xx,yy,zz,xy,xz,yz]
            pijc = np.column_stack([
                pc_raw[:, 0], pc_raw[:, 3], pc_raw[:, 5],
                pc_raw[:, 1], pc_raw[:, 2], pc_raw[:, 4],
            ])
        except Exception:
            pijc = None
    else:
        pijc = None

    # align: pijk has 1 more row than pijc (PREV_TIME==0 on first step)
    n_k = pijk.shape[0]
    if pijc is not None:
        offset  = max(0, n_k - pijc.shape[0])
        P_total = pijk[offset:] + pijc   # [xx,yy,zz,xy,xz,yz]
    else:
        offset  = 0
        P_total = pijk

    T_trans = tg[offset:, 2]   # GRANULAR = T_trans

    if abs(alpha - 1.0) < 1e-3:
        # Elastic: no steady state — use early plateau before string instability
        n_rows = min(elastic_early_rows, len(T_trans))
        mask_t = np.zeros(len(T_trans), dtype=bool)
        mask_t[:n_rows] = True
    else:
        mask_t, _, _ = _stats_mask(T_trans, stats_frac=0.50,
                                   threshold=5e-4, smooth_window=51)

    T_mean = float(np.mean(T_trans[mask_t]))
    scale  = 1.0 / (pip * T_mean)

    # Pxx↔Pyy swap: NSP flow=y → col 1 (yy) is flow direction = P*_xx
    Pxx = float(scale * np.mean(P_total[mask_t, 1]))
    Pyy = float(scale * np.mean(P_total[mask_t, 0]))
    Pzz = float(scale * np.mean(P_total[mask_t, 2]))
    Pxy = float(scale * np.mean(P_total[mask_t, 3]))

    return dict(
        alpha=alpha,
        Pxx_mean=Pxx, Pxx_std=0.0,
        Pyy_mean=Pyy, Pyy_std=0.0,
        Pzz_mean=Pzz, Pzz_std=0.0,
        Pxy_mean=Pxy, Pxy_std=0.0,
        theta_mean=float("nan"), theta_std=0.0,
    )


def load_nsphere_dsmc_sweep(sweep_dir: Path,
                             elastic_early_rows: int = 850) -> list[dict]:
    """Load all sphere NSP DSMC cases from sweep_dir.

    Subdirectory naming convention: integer alpha*100 (e.g. 60, 70, ..., 100).
    """
    cases = []
    for d in sorted(sweep_dir.iterdir()):
        if not d.is_dir():
            continue
        try:
            int(d.name)
        except ValueError:
            continue
        c = load_nsphere_dsmc_case(d, elastic_early_rows=elastic_early_rows)
        if c is not None:
            cases.append(c)
    cases.sort(key=lambda x: x["alpha"])
    return cases


# ---------------------------------------------------------------------------
# Shared figure-generation helper
# ---------------------------------------------------------------------------

def _generate_all_figures(dsmc_cases, lmp_sphcyl, lmp_spheres, kt,
                           out_dir: Path, time_label: str = "Time",
                           lmp_nematic=None, lmp_angvel=None,
                           lmp_sphcyl_dir=None, dsmc_spheres=None):
    """Write all USF figures into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Plotting steady-state diagnostics...")
    plot_steady_diagnostics(dsmc_cases, out_dir / "steady_diagnostics.png",
                            time_label=time_label)

    print("Plotting DSMC theta diagnostics...")
    plot_dsmc_theta_diagnostics(dsmc_cases, out_dir / "dsmc_theta_diagnostics.png",
                                time_label=time_label)

    if dsmc_spheres and dsmc_spheres[0].get("t_series"):
        print("Plotting sphere steady-state diagnostics...")
        plot_steady_diagnostics(dsmc_spheres,
                                out_dir / "sphere_steady_diagnostics.png",
                                time_label=time_label)

    print("Plotting stress overlay...")
    plot_stress_overlay(dsmc_cases, lmp_sphcyl, lmp_spheres, kt,
                        out_dir / "stress_overlay_2x2.png",
                        dsmc_spheres=dsmc_spheres)

    print("Plotting temperature ratio...")
    plot_temperature_ratio(dsmc_cases, lmp_sphcyl,
                           out_dir / "temperature_ratio.png")

    if lmp_sphcyl_dir is not None:
        print("Plotting LAMMPS steady-state diagnostics...")
        plot_lammps_steady_diagnostics(
            lmp_sphcyl_dir,
            out_dir / "lammps_steady_diagnostics.png"
        )
        print("Plotting LAMMPS theta diagnostics...")
        plot_lammps_theta_diagnostics(
            lmp_sphcyl_dir,
            out_dir / "lammps_theta_diagnostics.png"
        )

    if lmp_nematic is not None:
        print("Plotting nematic order parameter...")
        plot_nematic_order(lmp_nematic, out_dir / "nematic_order.png")

    if lmp_angvel is not None:
        print("Plotting angular velocity distribution...")
        plot_angvel_distribution(lmp_angvel, out_dir / "angvel_distribution.png")

    print(f"  Done. Figures in: {out_dir}")


# ---------------------------------------------------------------------------
# Figure 1: Steady-state diagnostics
# ---------------------------------------------------------------------------

def plot_steady_diagnostics(dsmc_cases: list[dict], out_path: Path,
                             time_label: str = "Time"):
    n = len(dsmc_cases)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).ravel()

    cmap = plt.get_cmap("plasma")
    alpha_vals = [c["alpha"] for c in dsmc_cases]
    amin, amax = min(alpha_vals), max(alpha_vals)

    for idx, case in enumerate(dsmc_cases):
        ax = axes[idx]
        color = cmap(0.1 + 0.8 * (case["alpha"] - amin) / max(amax - amin, 1e-9))

        for t_arr, T_arr in zip(case["t_series"], case["T_series"]):
            tx, Tx = _thin(t_arr, T_arr, 800)
            ax.plot(tx, Tx, color=color, lw=0.8, alpha=0.55)

        # Use first seed for the diagnostic annotations
        if case["t_series"]:
            t0      = case["t_series"][0]
            T0      = case["T_series"][0]
            t_stats = case["stats_starts"][0]
            t_plat  = case["plateau_starts"][0]

            # shade only the stats window (the "very good" part)
            mask_shade = t0 >= t_stats
            if mask_shade.any():
                T_mean = float(np.mean(T0[mask_shade]))
                ax.axvspan(t_stats, float(t0[-1]), color=color, alpha=0.20, lw=0)
                # dashed mean line spanning the stats window only
                ax.hlines(T_mean, t_stats, float(t0[-1]),
                          colors=color, lw=1.3, ls="--", alpha=0.95, zorder=3)

            # dotted line at plateau start (informational — where plateau begins)
            ax.axvline(t_plat,  color="0.5", lw=0.8, ls=":")
            # solid line at stats window start
            ax.axvline(t_stats, color="0.2", lw=1.0, ls="--")

        ax.set_title(rf"$\alpha = {case['alpha']:.2f}$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(time_label, fontsize=8)
        ax.set_ylabel(r"$T_{\rm total}$", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "DSMC USF — steady-state diagnostics\n"
        "shaded = stats window  |  dashed line = stats start  |  dotted = plateau start",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 1b: DSMC theta = T_trans/T_rot diagnostics
# ---------------------------------------------------------------------------

def plot_dsmc_theta_diagnostics(dsmc_cases: list[dict], out_path: Path,
                                 time_label: str = "Time"):
    n = len(dsmc_cases)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).ravel()

    cmap = plt.get_cmap("plasma")
    alpha_vals = [c["alpha"] for c in dsmc_cases]
    amin, amax = min(alpha_vals), max(alpha_vals)

    for idx, case in enumerate(dsmc_cases):
        ax = axes[idx]
        color = cmap(0.1 + 0.8 * (case["alpha"] - amin) / max(amax - amin, 1e-9))

        for t_arr, th_arr in zip(case["t_series"], case["theta_series"]):
            tx, thx = _thin(t_arr, th_arr, 800)
            ax.plot(tx, thx, color=color, lw=0.8, alpha=0.55)

        if case["t_series"]:
            t0       = case["t_series"][0]
            th0      = case["theta_series"][0]
            t_stats  = case["stats_starts"][0]
            t_plat   = case["plateau_starts"][0]

            mask_shade = t0 >= t_stats
            if mask_shade.any():
                th_mean = float(np.nanmean(th0[mask_shade]))
                ax.axvspan(t_stats, float(t0[-1]), color=color, alpha=0.20, lw=0)
                ax.hlines(th_mean, t_stats, float(t0[-1]),
                          colors=color, lw=1.3, ls="--", alpha=0.95, zorder=3)

            ax.axvline(t_plat,  color="0.5", lw=0.8, ls=":")
            ax.axvline(t_stats, color="0.2", lw=1.0, ls="--")

        ax.set_title(rf"$\alpha = {case['alpha']:.2f}$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel(time_label, fontsize=8)
        ax.set_ylabel(r"$\theta = T_{\rm tr}/T_{\rm rot}$", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        r"DSMC USF — temperature ratio $\theta = T_{\rm tr}/T_{\rm rot}$"
        "\nshaded = stats window  |  dashed line = stats start  |  dotted = plateau start",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Stress tensor overlay (2×2)
# ---------------------------------------------------------------------------

def plot_stress_overlay(dsmc_cases: list[dict],
                        lmp_sphcyl: list[dict],
                        lmp_spheres: list[dict],
                        kt: dict,
                        out_path: Path,
                        dsmc_spheres: list[dict] | None = None):

    dsmc_a = np.array([c["alpha"] for c in dsmc_cases])
    lsc_a  = np.array([r["e"]     for r in lmp_sphcyl])
    lsp_a  = np.array([r["e"]     for r in lmp_spheres])

    comps = [
        ("Pxx", r"$P_{xx}^*$"),
        ("Pyy", r"$P_{yy}^*$"),
        ("Pzz", r"$P_{zz}^*$"),
        ("Pxy", r"$P_{xy}^*$"),
    ]

    dsmc_vals = {
        k: (np.array([c[f"{k}_mean"] for c in dsmc_cases]),
            np.array([c[f"{k}_std"]  for c in dsmc_cases]))
        for k in ("Pxx", "Pyy", "Pzz", "Pxy")
    }
    lsc_vals = {k: np.array([r[k] for r in lmp_sphcyl])  for k in ("Pxx","Pyy","Pzz","Pxy")}
    lsp_vals = {k: np.array([r[k] for r in lmp_spheres]) for k in ("Pxx","Pyy","Pzz","Pxy")}
    kt_vals  = {"Pxx": kt["Pxx"], "Pyy": kt["Pyy"], "Pzz": kt["Pzz"], "Pxy": kt["Pxy"]}

    C_DSMC  = "#2166AC"   # steelblue
    C_LSC   = "#1A1A2E"   # near-black navy
    C_LSP   = "#D95F02"   # burnt orange
    C_KT    = "#555555"   # grey
    C_DSPHR = "#2CA02C"   # green — DSMC spheres
    ms = 7.5

    dsp_a    = np.array([c["alpha"]        for c in dsmc_spheres]) if dsmc_spheres else np.array([])
    dsp_vals = {k: np.array([c[f"{k}_mean"] for c in dsmc_spheres])
                for k in ("Pxx","Pyy","Pzz","Pxy")} if dsmc_spheres else {}

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.5))
    axes_flat = axes.ravel()
    legend_handles = []

    for ax_idx, (key, ylabel) in enumerate(comps):
        ax = axes_flat[ax_idx]
        mean_d, std_d = dsmc_vals[key]

        # DSMC AR=2 — open squares, markers only
        h1, = ax.plot(dsmc_a, mean_d, ls="none", marker="s", ms=ms,
                      mec=C_DSMC, mfc="none", mew=1.6, zorder=4,
                      label="DSMC AR=2")
        ax.fill_between(dsmc_a, mean_d - std_d, mean_d + std_d,
                        color=C_DSMC, alpha=0.18, zorder=2)

        # LAMMPS AR=2 — filled squares, markers only
        h2, = ax.plot(lsc_a, lsc_vals[key], ls="none", marker="s",
                      ms=ms - 1, color=C_LSC, zorder=5,
                      label="LAMMPS AR=2")

        # LAMMPS spheres — filled circles, markers only
        h3, = ax.plot(lsp_a, lsp_vals[key], ls="none", marker="o",
                      ms=ms - 1, color=C_LSP, zorder=5,
                      label="LAMMPS spheres")

        # KT theory — line only, no markers
        h4, = ax.plot(kt["alpha"], kt_vals[key], "-", color=C_KT, lw=1.5,
                      zorder=1, label="KT (spheres)")

        # DSMC spheres — open circles
        h5 = None
        if len(dsp_a) > 0:
            h5, = ax.plot(dsp_a, dsp_vals[key], ls="none", marker="o", ms=ms,
                          mec=C_DSPHR, mfc="none", mew=1.6, zorder=4,
                          label="DSMC spheres")

        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(labelsize=9, direction="in", which="both", top=True, right=True)
        ax.grid(True, ls="--", lw=0.4, alpha=0.45)

        all_y = (list(mean_d) + list(lsc_vals[key]) +
                 list(lsp_vals[key]) + list(kt_vals[key]) +
                 (list(dsp_vals[key]) if len(dsp_a) > 0 else []))
        ylo, yhi = _axis_limits([all_y])
        if ylo is not None:
            ax.set_ylim(ylo, yhi)

        all_x = (list(dsmc_a) + list(lsc_a) + list(lsp_a) + list(kt["alpha"]) +
                 (list(dsp_a) if len(dsp_a) > 0 else []))
        xlo, xhi = _axis_limits([all_x], pad=0.03)
        if xlo is not None:
            ax.set_xlim(xlo, xhi)

        if ax_idx >= 2:
            ax.set_xlabel(r"Coefficient of restitution $\alpha$", fontsize=11)

        if ax_idx == 0:
            legend_handles = [h for h in [h1, h2, h3, h4, h5] if h is not None]

    ncol_leg = len(legend_handles)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles],
               loc="lower center", ncol=ncol_leg, fontsize=9.5,
               frameon=True, edgecolor="0.7",
               bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        r"Reduced stress tensor $P_{ij}^* = P_{ij}/nT$  —  AR=2 spherocylinder vs sphere USF",
        fontsize=11, y=1.01
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1.0])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Temperature ratio
# ---------------------------------------------------------------------------

def plot_temperature_ratio(dsmc_cases: list[dict],
                           lmp_sphcyl: list[dict],
                           out_path: Path):

    dsmc_a  = np.array([c["alpha"]      for c in dsmc_cases])
    dsmc_th = np.array([c["theta_mean"] for c in dsmc_cases])
    dsmc_ts = np.array([c["theta_std"]  for c in dsmc_cases])

    lsc_a  = np.array([r["e"] for r in lmp_sphcyl])
    # theta already stored as T_tr/T_rot by load_lammps_sphcyl
    lsc_th = np.array([r["theta"] for r in lmp_sphcyl])

    C_DSMC = "#2166AC"
    C_LSC  = "#1A1A2E"
    ms = 7.5

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.plot(dsmc_a, dsmc_th, ls="none", marker="s", ms=ms,
            mec=C_DSMC, mfc="none", mew=1.6, zorder=4, label="DSMC AR=2")
    ax.fill_between(dsmc_a, dsmc_th - dsmc_ts, dsmc_th + dsmc_ts,
                    color=C_DSMC, alpha=0.18, zorder=2)
    ax.plot(lsc_a, lsc_th, ls="none", marker="s", ms=ms - 1,
            color=C_LSC, zorder=5, label="LAMMPS AR=2")

    ylo, yhi = _axis_limits([list(dsmc_th) + list(lsc_th)])
    if ylo is not None:
        ax.set_ylim(ylo, yhi)

    xlo, xhi = _axis_limits([list(dsmc_a) + list(lsc_a)], pad=0.03)
    if xlo is not None:
        ax.set_xlim(xlo, xhi)

    ax.set_xlabel(r"Coefficient of restitution $\alpha$", fontsize=11)
    ax.set_ylabel(r"Steady-state $\theta^* = T_{\rm tr}/T_{\rm rot}$", fontsize=11)
    ax.tick_params(labelsize=9, direction="in", which="both", top=True, right=True)
    ax.grid(True, ls="--", lw=0.4, alpha=0.45)
    ax.legend(fontsize=9.5, frameon=True, edgecolor="0.7", loc="best")
    ax.set_title(
        r"Translational/rotational temperature ratio — USF steady state (AR=2)",
        fontsize=10
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: LAMMPS steady-state diagnostics
# ---------------------------------------------------------------------------

def plot_lammps_steady_diagnostics(base_dir: Path, out_path: Path,
                                    tail_frac: float = 0.30):
    """Steady-state diagnostic plot for LAMMPS USF spherocylinder data.

    For each e_XXX case loads temperature_stats.dat, computes
    T_total = (3*T_trans + 2*T_rot)/5 normalised by its initial value,
    then applies the same plateau detection as the DSMC diagnostic to
    identify the stats window.

    x-axis: row index (each row = Nfreq=5000 timesteps of production data)
    y-axis: T_total / T_total[0]
    """
    # Collect cases
    cases = []
    for folder in sorted(base_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        tp_path = folder / "temperature_stats.dat"
        if not tp_path.exists():
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        data = np.atleast_2d(np.loadtxt(tp_path, comments="#"))
        T_trans = data[:, 1]
        T_rot   = data[:, 2]
        T_total = (3.0 * T_trans + 2.0 * T_rot) / 5.0
        row_idx = np.arange(len(T_total), dtype=float)
        cases.append(dict(e=e_val, row_idx=row_idx, T_total=T_total))

    if not cases:
        print("  Skipping LAMMPS steady diagnostics — no temperature_stats.dat found.")
        return

    cases.sort(key=lambda c: c["e"])
    n = len(cases)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).ravel()

    cmap = plt.get_cmap("plasma")
    e_vals = [c["e"] for c in cases]
    emin, emax = min(e_vals), max(e_vals)

    for idx, case in enumerate(cases):
        ax = axes[idx]
        color = cmap(0.1 + 0.8 * (case["e"] - emin) / max(emax - emin, 1e-9))

        row_idx = case["row_idx"]
        T_tot   = case["T_total"]

        ax.plot(row_idx, T_tot, color=color, lw=0.8, alpha=0.75)

        # Plateau detection on the raw series
        mask, stats_idx, plat_idx = _stats_mask(
            T_tot, stats_frac=0.50, threshold=5e-4, smooth_window=51
        )

        if mask.any():
            T_mean = float(np.mean(T_tot[mask]))
            stats_start_x = float(row_idx[stats_idx])
            plat_start_x  = float(row_idx[plat_idx])

            ax.axvspan(stats_start_x, float(row_idx[-1]),
                       color=color, alpha=0.20, lw=0)
            ax.hlines(T_mean, stats_start_x, float(row_idx[-1]),
                      colors=color, lw=1.3, ls="--", alpha=0.95, zorder=3)
            ax.axvline(plat_start_x,  color="0.5", lw=0.8, ls=":")
            ax.axvline(stats_start_x, color="0.2", lw=1.0, ls="--")

        ax.set_title(rf"$\alpha = {case['e']:.2f}$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Production output index", fontsize=8)
        ax.set_ylabel(r"$T_{\rm total}$", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        "LAMMPS USF — steady-state diagnostics (AR=2 spherocylinder)\n"
        "shaded = stats window  |  dashed line = stats start  |  dotted = plateau start",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: LAMMPS theta = T_trans/T_rot diagnostic over time
# ---------------------------------------------------------------------------

def plot_lammps_theta_diagnostics(base_dir: Path, out_path: Path):
    """Tile of theta = T_trans/T_rot vs production output index for each LAMMPS case.

    Same layout and style as plot_lammps_steady_diagnostics.
    """
    cases = []
    for folder in sorted(base_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        tp_path = folder / "temperature_stats.dat"
        if not tp_path.exists():
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        data = np.atleast_2d(np.loadtxt(tp_path, comments="#"))
        T_trans = data[:, 1]
        T_rot   = data[:, 2]
        theta   = T_trans / (T_rot + 1e-30)
        row_idx = np.arange(len(theta), dtype=float)
        cases.append(dict(e=e_val, row_idx=row_idx, theta=theta))

    if not cases:
        print("  Skipping LAMMPS theta diagnostics — no temperature_stats.dat found.")
        return

    cases.sort(key=lambda c: c["e"])
    n = len(cases)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.6 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).ravel()

    cmap = plt.get_cmap("plasma")
    e_vals = [c["e"] for c in cases]
    emin, emax = min(e_vals), max(e_vals)

    for idx, case in enumerate(cases):
        ax = axes[idx]
        color = cmap(0.1 + 0.8 * (case["e"] - emin) / max(emax - emin, 1e-9))

        row_idx = case["row_idx"]
        theta   = case["theta"]

        ax.plot(row_idx, theta, color=color, lw=0.8, alpha=0.75)

        mask, stats_idx, plat_idx = _stats_mask(
            theta, stats_frac=0.50, threshold=5e-4, smooth_window=51
        )

        if mask.any():
            theta_mean    = float(np.mean(theta[mask]))
            stats_start_x = float(row_idx[stats_idx])
            plat_start_x  = float(row_idx[plat_idx])

            ax.axvspan(stats_start_x, float(row_idx[-1]),
                       color=color, alpha=0.20, lw=0)
            ax.hlines(theta_mean, stats_start_x, float(row_idx[-1]),
                      colors=color, lw=1.3, ls="--", alpha=0.95, zorder=3)
            ax.axvline(plat_start_x,  color="0.5", lw=0.8, ls=":")
            ax.axvline(stats_start_x, color="0.2", lw=1.0, ls="--")

        ax.set_title(rf"$\alpha = {case['e']:.2f}$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Production output index", fontsize=8)
        ax.set_ylabel(r"$\theta = T_{\rm tr}/T_{\rm rot}$", fontsize=8)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(
        r"LAMMPS USF — temperature ratio $\theta = T_{\rm tr}/T_{\rm rot}$ (AR=2 spherocylinder)"
        "\nshaded = stats window  |  dashed line = stats start  |  dotted = plateau start",
        fontsize=10
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: Nematic order parameter S(alpha)
# ---------------------------------------------------------------------------

def plot_nematic_order(lmp_nematic: list[dict], out_path: Path):
    """Plot scalar nematic order parameter S vs alpha for LAMMPS spherocylinder USF."""
    if not lmp_nematic:
        print("  Skipping nematic_order.png — no data loaded.")
        return

    C_LSC = "#1A1A2E"
    ms = 6.5

    alphas = np.array([r["e"]      for r in lmp_nematic])
    S_mean = np.array([r["S_mean"] for r in lmp_nematic])
    S_std  = np.array([r["S_std"]  for r in lmp_nematic])

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.errorbar(alphas, S_mean, yerr=S_std,
                fmt="s", ms=ms, color=C_LSC, ecolor=C_LSC,
                elinewidth=1.0, capsize=3, zorder=5, label="LAMMPS AR=2")

    ax.axhline(0.0, color="0.55", ls="--", lw=1.0, zorder=2,
               label="Isotropic ($S=0$)")

    ylo, yhi = _axis_limits([list(S_mean - S_std) + list(S_mean + S_std)], pad=0.12)
    ylo = min(ylo, -0.005) if ylo is not None else -0.005
    if ylo is not None:
        ax.set_ylim(ylo, yhi)

    ax.set_xlabel(r"Coefficient of restitution $\alpha$", fontsize=11)
    ax.set_ylabel(r"$S = \lambda_{\rm max}(Q)$", fontsize=11)
    ax.tick_params(labelsize=9, direction="in", which="both", top=True, right=True)
    ax.grid(True, ls="--", lw=0.4, alpha=0.45)
    ax.legend(fontsize=9.5, frameon=True, edgecolor="0.7", loc="best")
    ax.set_title(
        "Nematic order parameter — AR=2 spherocylinder USF",
        fontsize=10
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5: Angular velocity distribution anisotropy
# ---------------------------------------------------------------------------

def plot_angvel_distribution(lmp_angvel: list[dict], out_path: Path):
    """Plot angular velocity anisotropy metrics vs alpha.

    Left panel:  <omega_pec_z^2> / <omega_perp^2>  (z-spin energy fraction)
    Right panel: <omega_pec_z>   / sqrt(<omega_perp^2>)  (normalised mean drift)
    """
    if not lmp_angvel:
        print("  Skipping angvel_distribution.png — no data loaded.")
        return

    C_LSC = "#1A1A2E"
    ms = 6.5

    alphas    = np.array([r["e"]         for r in lmp_angvel])
    ratios    = np.array([r["ratio"]     for r in lmp_angvel])
    norm_mean = np.array([r["norm_mean"] for r in lmp_angvel])

    # Elastic (alpha=1.0) baseline for z-fraction reference
    elastic_rows = [r for r in lmp_angvel if abs(r["e"] - 1.0) < 1e-3]
    ratio_ref = float(elastic_rows[0]["ratio"]) if elastic_rows else None

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))

    # --- Left: z-spin energy fraction ---
    ax = axes[0]
    ax.plot(alphas, ratios, ls="none", marker="s", ms=ms,
            color=C_LSC, zorder=5, label="LAMMPS AR=2")
    if ratio_ref is not None:
        ax.axhline(ratio_ref, color="0.55", ls="--", lw=1.0, zorder=2,
                   label=rf"Elastic ($\alpha=1$) baseline")
    ax.set_xlabel(r"Coefficient of restitution $\alpha$", fontsize=11)
    ax.set_ylabel(
        r"$\langle\omega_{{\rm pec},z}^2\rangle\,/\,\langle\omega_\perp^2\rangle$",
        fontsize=11
    )
    ax.tick_params(labelsize=9, direction="in", which="both", top=True, right=True)
    ax.grid(True, ls="--", lw=0.4, alpha=0.45)
    ax.legend(fontsize=9.5, frameon=True, edgecolor="0.7", loc="best")

    # --- Right: normalised mean z-spin drift ---
    ax = axes[1]
    ax.plot(alphas, norm_mean, ls="none", marker="s", ms=ms,
            color=C_LSC, zorder=5, label="LAMMPS AR=2")
    ax.axhline(0.0, color="0.55", ls="--", lw=1.0, zorder=2,
               label="No net drift")
    ax.set_xlabel(r"Coefficient of restitution $\alpha$", fontsize=11)
    ax.set_ylabel(
        r"$\langle\omega_{{\rm pec},z}\rangle\,/\,\langle\omega_\perp^2\rangle^{1/2}$",
        fontsize=11
    )
    ax.tick_params(labelsize=9, direction="in", which="both", top=True, right=True)
    ax.grid(True, ls="--", lw=0.4, alpha=0.45)
    ax.legend(fontsize=9.5, frameon=True, edgecolor="0.7", loc="best")

    fig.suptitle(
        "Angular velocity distribution — AR=2 USF (streaming-subtracted z-spin)",
        fontsize=10
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="USF final overlay plots")
    parser.add_argument("--sweep-dir",      default="runs/AR2_usf_sweep")
    parser.add_argument("--lammps-sphcyl",  default="LAMMPS_data/USF/sphcyl_USF_AR2")
    parser.add_argument("--lammps-spheres", default="LAMMPS_data/USF/spheres/dilute3a")
    parser.add_argument("--dsmc-spheres",   default="runs/Run_shear_NN",
                        help="Path to sphere DSMC sweep: either 0D format (alpha_NNN "
                             "subdirs) or NSP Fortran format (integer subdirs). "
                             "Pass empty string to skip.")
    parser.add_argument("--out-dir",        default=None)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_dir   = Path(args.out_dir) if args.out_dir else sweep_dir / "figures_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DSMC cases...")
    dsmc_cases = load_all_dsmc(sweep_dir)
    print(f"  {len(dsmc_cases)} cases: alpha = {[c['alpha'] for c in dsmc_cases]}")

    print("Loading LAMMPS spherocylinder data...")
    lmp_sphcyl = load_lammps_sphcyl(Path(args.lammps_sphcyl))
    print(f"  {len(lmp_sphcyl)} cases: e = {[r['e'] for r in lmp_sphcyl]}")

    print("Loading LAMMPS sphere data...")
    lmp_spheres = load_lammps_spheres(Path(args.lammps_spheres))
    print(f"  {len(lmp_spheres)} cases: e = {[r['e'] for r in lmp_spheres]}")

    print("Computing KT prediction...")
    kt = garzo_dufty_spheres(np.linspace(0.50, 1.00, 200))

    print("Loading LAMMPS nematic order data...")
    lmp_nematic = load_lammps_nematic(Path(args.lammps_sphcyl))
    print(f"  {len(lmp_nematic)} cases loaded.")

    print("Loading LAMMPS angular velocity data...")
    lmp_angvel = load_lammps_angvel(Path(args.lammps_sphcyl))
    print(f"  {len(lmp_angvel)} cases loaded.")

    dsmc_spheres = None
    if args.dsmc_spheres:
        dsmc_sph_dir = Path(args.dsmc_spheres)
        if dsmc_sph_dir.exists():
            # Auto-detect format: 0D sweep has alpha_NNN subdirs; NSP has integer subdirs
            has_alpha_dirs = any(
                d.is_dir() and d.name.startswith("alpha_")
                for d in dsmc_sph_dir.iterdir()
            )
            print("Loading DSMC sphere cases...")
            if has_alpha_dirs:
                print("  Detected 0D sweep format (alpha_NNN subdirs).")
                dsmc_spheres = load_all_dsmc(dsmc_sph_dir)
            else:
                print("  Detected NSP Fortran format (integer subdirs).")
                dsmc_spheres = load_nsphere_dsmc_sweep(dsmc_sph_dir)
            print(f"  {len(dsmc_spheres)} cases: alpha = {[c['alpha'] for c in dsmc_spheres]}")
        else:
            print(f"  DSMC spheres dir not found: {dsmc_sph_dir} — skipping.")

    _generate_all_figures(dsmc_cases, lmp_sphcyl, lmp_spheres, kt, out_dir,
                          time_label="Time",
                          lmp_nematic=lmp_nematic, lmp_angvel=lmp_angvel,
                          lmp_sphcyl_dir=Path(args.lammps_sphcyl),
                          dsmc_spheres=dsmc_spheres)
    print(f"\nDone. Figures in: {out_dir}")


def main_nsp():
    """Entry point for NSP (spatially-resolved) DSMC sweep figures.

    Processes one or more NSP sweep directories, producing the same three
    figures as main() but using the NSP DSMC file format.  LAMMPS AR=2 and
    sphere data and the KT prediction are the same shared reference.

    Usage (from project root):
        python run_nsp_final_plots.py [--sweeps <dir> [<dir> ...]]
    """
    parser = argparse.ArgumentParser(
        description="NSP DSMC USF overlay plots (same style as 0D sweep)"
    )
    parser.add_argument(
        "--sweeps", nargs="+",
        default=["runs/Run_sphcyl_shear_NN3", "runs/Run_sphcyl_shear_NN"],
        help="One or more NSP DSMC sweep directories (each produces its own figures_final/).",
    )
    parser.add_argument("--lammps-sphcyl",  default="LAMMPS_data/USF/sphcyl_USF_AR2")
    parser.add_argument("--lammps-spheres", default="LAMMPS_data/USF/spheres/dilute3a")
    args = parser.parse_args()

    print("Loading LAMMPS spherocylinder data...")
    lmp_sphcyl = load_lammps_sphcyl(Path(args.lammps_sphcyl))
    print(f"  {len(lmp_sphcyl)} cases: e = {[r['e'] for r in lmp_sphcyl]}")

    print("Loading LAMMPS sphere data...")
    lmp_spheres = load_lammps_spheres(Path(args.lammps_spheres))
    print(f"  {len(lmp_spheres)} cases: e = {[r['e'] for r in lmp_spheres]}")

    print("Computing KT prediction...")
    kt = garzo_dufty_spheres(np.linspace(0.50, 1.00, 200))

    for sweep_path in args.sweeps:
        sweep_dir = Path(sweep_path)
        out_dir   = sweep_dir / "figures_final"
        print(f"\n=== NSP sweep: {sweep_dir} ===")
        print("Loading NSP DSMC cases...")
        dsmc_cases = load_nsp_dsmc_sweep(sweep_dir)
        print(f"  {len(dsmc_cases)} cases: alpha = {[c['alpha'] for c in dsmc_cases]}")
        if not dsmc_cases:
            print("  No usable cases, skipping.")
            continue
        _generate_all_figures(dsmc_cases, lmp_sphcyl, lmp_spheres, kt, out_dir,
                               time_label=r"Collision time $\tau$")

    print("\nAll done.")


if __name__ == "__main__":
    main()
