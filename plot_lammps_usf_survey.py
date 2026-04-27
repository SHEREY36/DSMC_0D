#!/usr/bin/env python3
"""LAMMPS-only USF survey: stress, theta, nematic order, and angular velocity
across all AR cases in LAMMPS_data/USF/.

Produces:
  lammps_usf_stress.png   — P*_xx and |P*_xy| vs alpha, all ARs overlaid
  lammps_usf_theta.png    — T_tr/T_rot vs alpha, spherocylinders
  lammps_usf_nematic.png  — nematic order S vs alpha, spherocylinders
  lammps_usf_angvel.png   — rotational anisotropy vs alpha, spherocylinders

Usage:
    python plot_lammps_usf_survey.py [--out-dir <dir>]
"""

import argparse
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AR_DIRS = OrderedDict([
    (1.0,  "AR1"),
    (1.1,  "AR11"),
    (1.5,  "AR15"),
    (2.0,  "AR2"),
    (2.5,  "AR25"),
    (3.0,  "AR30"),
])
SPHCYL_ARS = OrderedDict((k, v) for k, v in AR_DIRS.items() if k > 1.0)

LAMMPS_ROOT  = Path("LAMMPS_data/USF")
TAIL_FRAC    = 0.30
EARLY_ROWS   = 850   # early plateau for alpha=1 (string instability avoidance)

# Color palette — one per AR value
COLORS  = {1.0: "#555555", 1.1: "#2166AC", 1.5: "#4DAC26",
           2.0: "#D01C8B", 2.5: "#F1A340", 3.0: "#D73027"}
MARKERS = {1.0: "o", 1.1: "s", 1.5: "^", 2.0: "D", 2.5: "v", 3.0: "P"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tail_mean(arr: np.ndarray, frac: float = TAIL_FRAC) -> float:
    n = len(arr)
    i0 = max(0, int((1.0 - frac) * n))
    return float(np.mean(arr[i0:]))


def early_mean(arr: np.ndarray, n_rows: int = EARLY_ROWS) -> float:
    return float(np.mean(arr[:n_rows]))


def is_elastic(e_val: float) -> bool:
    return abs(e_val - 1.0) < 1e-3


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_stress(ar_dir: Path) -> list[dict]:
    """Load stress (P*_xx, P*_xy, P*_yy, P*_zz) and theta for one AR."""
    rows = []
    for folder in sorted(ar_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        ss_path = folder / "shear_stats.dat"
        tp_path = folder / "temperature_stats.dat"
        if not ss_path.exists():
            continue

        ss = np.atleast_2d(np.loadtxt(ss_path, comments="#"))

        if is_elastic(e_val):
            pxx = early_mean(ss[:, 1])
            pyy = early_mean(ss[:, 2])
            pzz = early_mean(ss[:, 3])
            pxy = early_mean(ss[:, 4])
        else:
            pxx = tail_mean(ss[:, 1])
            pyy = tail_mean(ss[:, 2])
            pzz = tail_mean(ss[:, 3])
            pxy = tail_mean(ss[:, 4])

        theta = float("nan")
        if tp_path.exists():
            tmp = np.atleast_2d(np.loadtxt(tp_path, comments="#"))
            if is_elastic(e_val):
                ttr  = early_mean(tmp[:, 1])
                trot = early_mean(tmp[:, 2])
            else:
                ttr  = tail_mean(tmp[:, 1])
                trot = tail_mean(tmp[:, 2])
            theta = ttr / (trot + 1e-30)

        rows.append(dict(e=e_val, Pxx=pxx, Pyy=pyy, Pzz=pzz, Pxy=pxy, theta=theta))
    rows.sort(key=lambda r: r["e"])
    return rows


def load_nematic(ar_dir: Path) -> list[dict]:
    """Load nematic order parameter S = largest eigenvalue of Q."""
    rows = []
    for folder in sorted(ar_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        nem_path = folder / "nematic_tensor.dat"
        if not nem_path.exists():
            continue

        data = np.atleast_2d(np.loadtxt(nem_path, comments="#"))
        if is_elastic(e_val):
            window = data[:EARLY_ROWS]
        else:
            n = len(data)
            i0 = max(0, int((1.0 - TAIL_FRAC) * n))
            window = data[i0:]

        Qxx = window[:, 1]; Qyy = window[:, 2]; Qzz = window[:, 3]
        Qxy = window[:, 4]; Qxz = window[:, 5]; Qyz = window[:, 6]

        S_arr = np.empty(len(window))
        for k in range(len(window)):
            Q = np.array([
                [Qxx[k], Qxy[k], Qxz[k]],
                [Qxy[k], Qyy[k], Qyz[k]],
                [Qxz[k], Qyz[k], Qzz[k]],
            ])
            S_arr[k] = np.linalg.eigvalsh(Q)[-1]

        rows.append(dict(e=e_val, S=float(np.mean(S_arr)), S_std=float(np.std(S_arr))))
    rows.sort(key=lambda r: r["e"])
    return rows


def load_angvel(ar_dir: Path) -> list[dict]:
    """Load angular velocity stats.

    Columns: TimeStep omperp2 om2_tot ompecz ompecz2 ...
      omperp2:  <ω_perp²>  — tumbling (perp to particle axis)
      ompecz:   <ω_pec_z>  — mean peculiar z-angular velocity (~0 at SS)
      ompecz2:  <ω_pec_z²> — variance of peculiar z-angular velocity

    Follows the same convention as plot_usf_final.plot_angvel_distribution:
      ratio     = ompecz2 / omperp2
      norm_mean = ompecz  / sqrt(omperp2)
    """
    rows = []
    for folder in sorted(ar_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        av_path = folder / "angular_velocity_stats.dat"
        if not av_path.exists():
            continue

        data = np.atleast_2d(np.loadtxt(av_path, comments="#"))
        if is_elastic(e_val):
            window = data[:EARLY_ROWS]
        else:
            n = len(data)
            i0 = max(0, int((1.0 - TAIL_FRAC) * n))
            window = data[i0:]

        omperp2   = float(np.mean(window[:, 1]))
        ompecz    = float(np.mean(window[:, 3]))
        ompecz2   = float(np.mean(window[:, 4]))
        ratio     = ompecz2 / (omperp2 + 1e-30)
        norm_mean = ompecz  / (np.sqrt(omperp2) + 1e-30)

        rows.append(dict(e=e_val, omperp2=omperp2, ompecz=ompecz,
                         ompecz2=ompecz2, ratio=ratio, norm_mean=norm_mean))
    rows.sort(key=lambda r: r["e"])
    return rows


def _sigma_c(ar: float) -> float:
    """Mean collision cross-section (d=1).
    Sphere (AR=1): π·d²=π.  Spherocylinder: empirical fit from particle.py.
    """
    if abs(ar - 1.0) < 1e-3:
        return math.pi
    return (0.32 * ar**2 + 0.694 * ar - 0.0213) * math.pi


def load_astar(ar_dir: Path, ar_val: float) -> list[dict]:
    """Load steady-state reduced shear rate a* for one AR.

    a* = γ̇ / (n · σ_c · √T_trans)

    For α=1 (elastic) uses early-window T_trans; inelastic uses tail mean.
    Returns list of dicts {e, a_star, a_star_sq} sorted by e.
    """
    sig_c = _sigma_c(ar_val)
    rows = []
    for folder in sorted(ar_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        tp_path = folder / "temperature_stats.dat"
        if not tp_path.exists():
            continue

        # Read gdot, N, L
        gdot, N, L = None, None, None
        in_usf = folder / "in.usf"
        if in_usf.exists():
            with open(in_usf) as fh:
                for line in fh:
                    if "variable" in line and "gdot" in line and "equal" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "equal" and i + 1 < len(parts):
                                try:
                                    gdot = float(parts[i + 1])
                                except ValueError:
                                    pass
                                break
        N, L = _read_N_L(folder)
        if gdot is None or N is None or L is None:
            continue

        tmp = np.atleast_2d(np.loadtxt(tp_path, comments="#"))
        if is_elastic(e_val):
            T_tr = early_mean(tmp[:, 1])
        else:
            T_tr = tail_mean(tmp[:, 1])

        if T_tr <= 0:
            continue

        n = N / L**3
        a_s = gdot / (n * sig_c * math.sqrt(T_tr))
        rows.append(dict(e=e_val, a_star=a_s, a_star_sq=a_s**2))
    rows.sort(key=lambda r: r["e"])
    return rows


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Bagnold-scaling helpers and loader
# ---------------------------------------------------------------------------

def _particle_volume(ar: float) -> float:
    """Volume of one particle (d=1, m=1).
    Sphere: π/6.  Spherocylinder: π(3·AR−1)/12.
    """
    if abs(ar - 1.0) < 1e-3:
        return math.pi / 6.0
    return math.pi * (3.0 * ar - 1.0) / 12.0


def _equiv_diam(ar: float) -> float:
    """Equivalent-volume sphere diameter d_v = ((3·AR−1)/2)^(1/3).
    Matches Guo et al. convention: d_v is held fixed across shapes.
    AR=1→1.0, AR=1.1→1.0477, AR=1.5→1.2051, AR=2→1.3572, AR=2.5→1.4812, AR=3→1.5874.
    """
    return ((3.0 * ar - 1.0) / 2.0) ** (1.0 / 3.0)


def _read_N_L(folder: Path):
    """Return (N, L) from the *.data file in folder. Returns (None, None) on failure."""
    for fpath in folder.glob("*.data"):
        N, L = None, None
        with open(fpath) as fh:
            for line in fh:
                s = line.strip()
                if "atoms" in s and not s.startswith("#"):
                    try:
                        N = int(s.split()[0])
                    except (ValueError, IndexError):
                        pass
                elif "xlo xhi" in s:
                    try:
                        L = float(s.split()[1]) - float(s.split()[0])
                    except (ValueError, IndexError):
                        pass
                if N is not None and L is not None:
                    return N, L
    return None, None


def _read_dt(folder: Path):
    """Read resolved timestep from log.lammps. Returns None on failure."""
    log = folder / "log.lammps"
    dt = None
    if log.exists():
        with open(log) as fh:
            for line in fh:
                if line.startswith("timestep") and "${" not in line:
                    try:
                        dt = float(line.split()[1])
                    except (ValueError, IndexError):
                        pass
    return dt


def load_astar_nu(ar_dir: Path, ar_val: float) -> list[dict]:
    """a*_ν = γ̇ / ν, where ν is measured directly from collision_count.dat.

    ν (per particle) = 2 × Σ(new_events in window) / (N × time_window)
    Factor 2: each pair event involves 2 particles.
    α=1 excluded (elastic USF has no steady-state ν).
    """
    rows = []
    for folder in sorted(ar_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val = float(folder.name.split("_")[1]) / 100.0
        if is_elastic(e_val):
            continue
        cc_path = folder / "collision_count.dat"
        tp_path = folder / "temperature_stats.dat"
        if not (cc_path.exists() and tp_path.exists()):
            continue

        dt = _read_dt(folder)
        N, L = _read_N_L(folder)
        if dt is None or N is None:
            continue

        gdot = None
        in_usf = folder / "in.usf"
        if in_usf.exists():
            with open(in_usf) as fh:
                for line in fh:
                    if "variable" in line and "gdot" in line and "equal" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "equal" and i + 1 < len(parts):
                                try:
                                    gdot = float(parts[i + 1])
                                except ValueError:
                                    pass
                                break
        if gdot is None:
            gdot = 1.8

        cc  = np.atleast_2d(np.loadtxt(cc_path, comments="#"))
        tmp = np.atleast_2d(np.loadtxt(tp_path, comments="#"))

        prod_start = int(tmp[0, 0])
        cc_prod = cc[cc[:, 0] >= prod_start]
        n_rows  = len(cc_prod)
        i0 = max(0, int((1.0 - TAIL_FRAC) * n_rows))
        cc_window = cc_prod[i0:]

        if len(cc_window) < 2:
            continue

        N_events  = float(cc_window[:, 1].sum())
        step_span = float(cc_window[-1, 0] - cc_window[0, 0]) + 1000.0
        time_span = step_span * dt
        nu        = 2.0 * N_events / (N * time_span)

        a_s = gdot / nu
        rows.append(dict(e=e_val, a_star=a_s, a_star_sq=a_s**2))
    rows.sort(key=lambda r: r["e"])
    return rows


def load_bagnold_stress(ar_dir: Path, ar_val: float) -> list[dict]:
    """Load Bagnold-scaled stress: σ^N_ij = P̃_ij · φ · T_trans / (d_v² · γ²)

    Uses equivalent-volume diameter d_v (Guo et al. convention), not d=1.
    P̃_ij = P/(n·T_trans) from LAMMPS, m_p=1, so:
      σ^N = P̃ · φ · T_trans / (m_p · d_v² · γ²)
    """
    V_part = _particle_volume(ar_val)
    d_v    = _equiv_diam(ar_val)
    GDOT   = 1.8   # fixed for all cases

    rows = []
    for folder in sorted(ar_dir.iterdir()):
        if not (folder.is_dir() and folder.name.startswith("e_")):
            continue
        e_val   = float(folder.name.split("_")[1]) / 100.0
        # Elastic USF has no steady state — T_trans grows without bound.
        # Bagnold scale ∝ T_trans makes α=1 ~1000× larger; exclude it.
        if is_elastic(e_val):
            continue
        ss_path = folder / "shear_stats.dat"
        tp_path = folder / "temperature_stats.dat"
        if not (ss_path.exists() and tp_path.exists()):
            continue

        ss  = np.atleast_2d(np.loadtxt(ss_path,  comments="#"))
        tmp = np.atleast_2d(np.loadtxt(tp_path,  comments="#"))

        pxx = tail_mean(ss[:, 1]); pyy = tail_mean(ss[:, 2])
        pzz = tail_mean(ss[:, 3]); pxy = tail_mean(ss[:, 4])
        T_tr = tail_mean(tmp[:, 1])

        N, L = _read_N_L(folder)
        if N is None or L is None:
            continue

        phi   = (N / L**3) * V_part
        scale = phi * T_tr / (d_v**2 * GDOT**2)

        rows.append(dict(e=e_val,
                         Pxx=pxx * scale, Pyy=pyy * scale,
                         Pzz=pzz * scale, Pxy=pxy * scale))
    rows.sort(key=lambda r: r["e"])
    return rows


# Plotting helpers
# ---------------------------------------------------------------------------

def _alphas(rows): return np.array([r["e"] for r in rows])

def _add_legend(ax, fontsize=8):
    ax.legend(fontsize=fontsize, framealpha=0.7)

def _style_ax(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.25, lw=0.5)


# ---------------------------------------------------------------------------
# Figure 1: Stress overlay
# ---------------------------------------------------------------------------

def plot_stress(data_by_ar: dict, out_path: Path):
    """2×2 grid: Pxx, Pyy, Pzz, Pxy vs alpha — all ARs overlaid.
    Legend placed below the figure so it does not obstruct the panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_pxx, ax_pyy, ax_pzz, ax_pxy = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    handles = []
    for ar, rows in data_by_ar.items():
        a   = _alphas(rows)
        pxx = np.array([r["Pxx"] for r in rows])
        pyy = np.array([r["Pyy"] for r in rows])
        pzz = np.array([r["Pzz"] for r in rows])
        pxy = np.array([r["Pxy"] for r in rows])
        c = COLORS[ar]; m = MARKERS[ar]
        label = f"AR={ar:.1f}"
        kw = dict(marker=m, color=c, lw=1.4, ms=5)
        ln, = ax_pxx.plot(a, pxx, label=label, **kw)
        ax_pyy.plot(a, pyy, **kw)
        ax_pzz.plot(a, pzz, **kw)
        ax_pxy.plot(a, pxy, **kw)
        handles.append(ln)

    _style_ax(ax_pxx, r"$\alpha$", r"$P^*_{xx}$", r"$P^*_{xx}$")
    _style_ax(ax_pyy, r"$\alpha$", r"$P^*_{yy}$", r"$P^*_{yy}$")
    _style_ax(ax_pzz, r"$\alpha$", r"$P^*_{zz}$", r"$P^*_{zz}$")
    _style_ax(ax_pxy, r"$\alpha$", r"$P^*_{xy}$", r"$P^*_{xy}$")

    labels = [f"AR={ar:.1f}" for ar in data_by_ar]
    fig.legend(handles, labels, loc="lower center", ncol=len(data_by_ar),
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("LAMMPS USF — stress tensor components (all AR)", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Theta (T_tr / T_rot)
# ---------------------------------------------------------------------------

def plot_theta(data_by_ar: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for ar, rows in data_by_ar.items():
        if ar == 1.0:
            continue  # sphere has no rotational DOF in this LAMMPS setup
        a   = _alphas(rows)
        th  = np.array([r["theta"] for r in rows])
        c = COLORS[ar]; m = MARKERS[ar]
        ax.plot(a, th, marker=m, color=c, label=f"AR={ar:.1f}", lw=1.4, ms=5)

    ax.axhline(1.0, color="k", lw=1, ls="--", label=r"$\theta=1$ (equipartition)")
    _style_ax(ax, r"$\alpha$", r"$\theta = T_{\rm tr}/T_{\rm rot}$",
              r"Temperature ratio $\theta$ vs $\alpha$")
    _add_legend(ax)
    fig.suptitle("LAMMPS USF — translational/rotational temperature ratio", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3: Nematic order
# ---------------------------------------------------------------------------

def plot_nematic(nem_by_ar: dict, out_path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for ar, rows in nem_by_ar.items():
        a = _alphas(rows)
        S = np.array([r["S"] for r in rows])
        S_std = np.array([r["S_std"] for r in rows])
        c = COLORS[ar]; m = MARKERS[ar]
        ax.plot(a, S, marker=m, color=c, label=f"AR={ar:.1f}", lw=1.4, ms=5)
        ax.fill_between(a, S - S_std, S + S_std, color=c, alpha=0.15)

    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    _style_ax(ax, r"$\alpha$", r"$S$ (nematic order)",
              r"Nematic order parameter $S$ vs $\alpha$")
    _add_legend(ax)
    fig.suptitle("LAMMPS USF — nematic order (spherocylinders)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4: Angular velocity anisotropy
# ---------------------------------------------------------------------------

def plot_angvel(av_by_ar: dict, out_path: Path):
    """Angular velocity anisotropy — same convention as plot_usf_final.

    Left:  <ω_pec_z²> / <ω_perp²>  vs alpha
    Right: <ω_pec_z>  / sqrt(<ω_perp²>)  vs alpha

    Each AR is one color; its own alpha=1 elastic baseline is a dashed
    line in the same color.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ms = 6

    for ar, rows in av_by_ar.items():
        a         = _alphas(rows)
        ratio     = np.array([r["ratio"]     for r in rows])
        norm_mean = np.array([r["norm_mean"] for r in rows])
        c = COLORS[ar]; m = MARKERS[ar]
        label = f"AR={ar:.1f}"

        # Elastic baseline for this AR
        elastic = [r for r in rows if is_elastic(r["e"])]
        ratio_ref = elastic[0]["ratio"] if elastic else None

        axes[0].plot(a, ratio, ls="none", marker=m, ms=ms, color=c, label=label, zorder=5)
        if ratio_ref is not None:
            axes[0].axhline(ratio_ref, color=c, ls="--", lw=1.0, zorder=2)

        axes[1].plot(a, norm_mean, ls="none", marker=m, ms=ms, color=c, label=label, zorder=5)
        axes[1].axhline(0.0, color="0.55", ls="--", lw=0.8, zorder=1)

    _style_ax(axes[0],
              r"Coefficient of restitution $\alpha$",
              r"$\langle\omega_{{\rm pec},z}^2\rangle\,/\,\langle\omega_\perp^2\rangle$",
              r"z-spin energy fraction (dashed = elastic baseline per AR)")
    _style_ax(axes[1],
              r"Coefficient of restitution $\alpha$",
              r"$\langle\omega_{{\rm pec},z}\rangle\,/\,\langle\omega_\perp^2\rangle^{1/2}$",
              r"Normalised mean z-spin drift (dashed = zero)")
    for ax in axes:
        ax.tick_params(direction="in", which="both", top=True, right=True)
    _add_legend(axes[0]); _add_legend(axes[1])
    fig.suptitle("LAMMPS USF — angular velocity distribution (spherocylinders)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Bagnold stress plot
# ---------------------------------------------------------------------------

def plot_bagnold_stress(bagnold_by_ar: dict, out_path: Path):
    """2×2 grid of Bagnold-scaled stress components vs alpha — all ARs overlaid.

    σ^N_ij = P_ij / (ρ_mat · d² · γ²) = P̃_ij · φ · T_trans / γ²
    Legend placed below the figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    ax_pxx, ax_pyy, ax_pzz, ax_pxy = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    handles = []
    for ar, rows in bagnold_by_ar.items():
        if not rows:
            continue
        a   = _alphas(rows)
        pxx = np.array([r["Pxx"] for r in rows])
        pyy = np.array([r["Pyy"] for r in rows])
        pzz = np.array([r["Pzz"] for r in rows])
        pxy = np.array([r["Pxy"] for r in rows])
        c = COLORS[ar]; m = MARKERS[ar]
        kw = dict(marker=m, color=c, lw=1.4, ms=5)
        ln, = ax_pxx.plot(a, pxx, label=f"AR={ar:.1f}", **kw)
        ax_pyy.plot(a, pyy, **kw)
        ax_pzz.plot(a, pzz, **kw)
        ax_pxy.plot(a, pxy, **kw)
        handles.append(ln)

    ylabel = r"$\sigma^N_{ij} = P_{ij}\,/\,(\rho_{\rm mat}\,d_v^2\,\dot{\gamma}^2)$"
    _style_ax(ax_pxx, r"$\alpha$", ylabel, r"$\sigma^N_{xx}$")
    _style_ax(ax_pyy, r"$\alpha$", ylabel, r"$\sigma^N_{yy}$")
    _style_ax(ax_pzz, r"$\alpha$", ylabel, r"$\sigma^N_{zz}$")
    _style_ax(ax_pxy, r"$\alpha$", ylabel, r"$\sigma^N_{xy}$")

    labels = [f"AR={ar:.1f}" for ar in bagnold_by_ar if bagnold_by_ar[ar]]
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        r"LAMMPS USF — Bagnold-scaled stress  "
        r"$\sigma^N = P\,/\,(\rho_{\rm mat}\,d_v^2\,\dot{\gamma}^2)$  (all AR)",
        fontsize=11, y=1.01
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6: a*² vs alpha
# ---------------------------------------------------------------------------

def plot_astar_sq(astar_by_ar: dict, out_path: Path):
    """Plot (a*)² vs coefficient of restitution α — all ARs overlaid.

    Analogous to Brey et al. Fig. 4 (steady reduced shear rate squared).
    α=1 (elastic) excluded: no steady state, a* diverges.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for ar, rows in astar_by_ar.items():
        # exclude elastic point (a* → ∞)
        rows_inel = [r for r in rows if not is_elastic(r["e"])]
        if not rows_inel:
            continue
        a   = np.array([r["e"]        for r in rows_inel])
        asq = np.array([r["a_star_sq"] for r in rows_inel])
        c = COLORS[ar]; m = MARKERS[ar]
        ax.plot(a, asq, marker=m, color=c, label=f"AR={ar:.1f}", lw=1.4, ms=5)

    _style_ax(ax,
              r"Coefficient of restitution $\alpha$",
              r"$(a^*)^2 = \dot{\gamma}^2\,/\,(n\,\sigma_c\,\sqrt{T_{\rm tr}})^2$",
              r"Steady reduced shear rate $(a^*)^2$ vs $\alpha$")
    _add_legend(ax)
    fig.suptitle("LAMMPS USF — $(a^*)^2$ vs $\\alpha$ (all AR)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 7: a*²_ν vs alpha (collision-frequency based — σ_c validation)
# ---------------------------------------------------------------------------

def plot_astar_sq_nu(astar_nu_by_ar: dict, out_path: Path):
    """Plot (a*_ν)² vs α — a* from actual collision frequency ν.

    a*_ν = γ̇ / ν where ν is read from collision_count.dat.
    Compare with lammps_usf_astar_sq.png (σ_c-based) to validate σ_c(AR).
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for ar, rows in astar_nu_by_ar.items():
        if not rows:
            continue
        a   = np.array([r["e"]        for r in rows])
        asq = np.array([r["a_star_sq"] for r in rows])
        c = COLORS[ar]; m = MARKERS[ar]
        ax.plot(a, asq, marker=m, color=c, label=f"AR={ar:.1f}", lw=1.4, ms=5)

    _style_ax(ax,
              r"Coefficient of restitution $\alpha$",
              r"$(a^*_\nu)^2 = \dot{\gamma}^2\,/\,\nu^2$",
              r"Reduced shear rate from $\nu$ — $(a^*_\nu)^2$ vs $\alpha$")
    _add_legend(ax)
    fig.suptitle(r"LAMMPS USF — $(a^*_\nu)^2$ from collision frequency $\nu$ (all AR)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figures 8 & 9: Rheological quantities vs a*
# ---------------------------------------------------------------------------

_RHEOL_CMAP = cm.plasma_r
_ALPHA_VALS = np.array([0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
_RHEOL_NORM = plt.Normalize(vmin=_ALPHA_VALS.min(), vmax=_ALPHA_VALS.max())


def _merge_rheol(stress_rows: list, astar_rows: list) -> list:
    """Merge stress and a* data by α; compute η*, N1*, N2*. Excludes α=1."""
    astar_map = {r["e"]: r["a_star"] for r in astar_rows}
    merged = []
    for r in stress_rows:
        e = r["e"]
        if is_elastic(e):
            continue
        a_s = astar_map.get(e)
        if a_s is None or not np.isfinite(a_s) or a_s <= 0:
            continue
        merged.append(dict(
            e=e,
            a_star=a_s,
            eta= -r["Pxy"] / a_s,
            N1 =  r["Pxx"] - r["Pyy"],
            N2 =  r["Pyy"] - r["Pzz"],
        ))
    return sorted(merged, key=lambda x: x["e"])


def plot_rheology(data_by_ar: dict, astar_by_ar: dict, out_path: Path):
    """Three-panel figure: η*_s, N1*_s, N2*_s vs a*_s.

    Marker shape = AR.  Point color = α value (plasma_r colorbar).
    All ARs overlaid in each panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    ax_eta, ax_N1, ax_N2 = axes

    # Legend handles: one dummy scatter per AR (marker shape only, black)
    legend_handles = []

    for ar in AR_DIRS:
        if ar not in data_by_ar or ar not in astar_by_ar:
            continue
        rows = _merge_rheol(data_by_ar[ar], astar_by_ar[ar])
        if not rows:
            continue

        a_s  = np.array([r["a_star"] for r in rows])
        eta  = np.array([r["eta"]    for r in rows])
        N1   = np.array([r["N1"]     for r in rows])
        N2   = np.array([r["N2"]     for r in rows])
        alphas = np.array([r["e"]    for r in rows])
        colors = _RHEOL_CMAP(_RHEOL_NORM(alphas))

        m = MARKERS[ar]
        kw = dict(s=45, marker=m, c=colors, edgecolors="none", zorder=5)
        ax_eta.scatter(a_s, eta, **kw)
        ax_N1.scatter(a_s, N1,  **kw)
        ax_N2.scatter(a_s, N2,  **kw)

        # legend: black marker, no fill
        h = ax_eta.scatter([], [], s=40, marker=m, color="k",
                           label=f"AR={ar:.1f}", zorder=1)
        legend_handles.append(h)

    # Reference lines
    ax_N1.axhline(0, color="0.6", lw=0.8, ls="--")
    ax_N2.axhline(0, color="0.6", lw=0.8, ls="--")

    _style_ax(ax_eta, r"$a^*_s$", r"$\eta^*_s = -P^*_{xy,s}\,/\,a^*_s$",
              r"Shear viscosity $\eta^*_s$")
    _style_ax(ax_N1,  r"$a^*_s$", r"$N^*_{1,s} = P^*_{xx,s} - P^*_{yy,s}$",
              r"1st normal stress diff. $N^*_{1,s}$")
    _style_ax(ax_N2,  r"$a^*_s$", r"$N^*_{2,s} = P^*_{yy,s} - P^*_{zz,s}$",
              r"2nd normal stress diff. $N^*_{2,s}$")

    # Colorbar for α
    sm = plt.cm.ScalarMappable(cmap=_RHEOL_CMAP, norm=_RHEOL_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label(r"Coefficient of restitution $\alpha$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # AR legend inside first panel
    ax_eta.legend(handles=legend_handles, fontsize=7, loc="upper right",
                  framealpha=0.7, markerscale=1.0)

    fig.suptitle("LAMMPS USF — rheological quantities vs $a^*_s$ (all AR)",
                 fontsize=11)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=".", help="Output directory for figures")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load stress + theta (all ARs including sphere)
    print("Loading LAMMPS stress & temperature data...")
    stress_by_ar = {}
    for ar, dirname in AR_DIRS.items():
        ar_dir = LAMMPS_ROOT / dirname
        if not ar_dir.exists():
            print(f"  Skipping {dirname} (not found)")
            continue
        stress_by_ar[ar] = load_stress(ar_dir)
        print(f"  AR={ar:.1f}: {len(stress_by_ar[ar])} alpha values")

    # Load nematic (spherocylinders only)
    print("Loading nematic order data...")
    nem_by_ar = {}
    for ar, dirname in SPHCYL_ARS.items():
        ar_dir = LAMMPS_ROOT / dirname
        if not ar_dir.exists():
            continue
        rows = load_nematic(ar_dir)
        if rows:
            nem_by_ar[ar] = rows
            print(f"  AR={ar:.1f}: {len(rows)} alpha values")

    # Load angular velocity (spherocylinders only)
    print("Loading angular velocity data...")
    av_by_ar = {}
    for ar, dirname in SPHCYL_ARS.items():
        ar_dir = LAMMPS_ROOT / dirname
        if not ar_dir.exists():
            continue
        rows = load_angvel(ar_dir)
        if rows:
            av_by_ar[ar] = rows
            print(f"  AR={ar:.1f}: {len(rows)} alpha values")

    # Load Bagnold-scaled stress (all ARs)
    print("Loading Bagnold-scaled stress data...")
    bagnold_by_ar = {}
    for ar, dirname in AR_DIRS.items():
        ar_dir = LAMMPS_ROOT / dirname
        if not ar_dir.exists():
            continue
        bagnold_by_ar[ar] = load_bagnold_stress(ar_dir, ar)
        print(f"  AR={ar:.1f}: {len(bagnold_by_ar[ar])} alpha values")

    # Load reduced shear rate a* via σ_c (all ARs)
    print("Loading a* (σ_c-based) data...")
    astar_by_ar = {}
    for ar, dirname in AR_DIRS.items():
        ar_dir = LAMMPS_ROOT / dirname
        if not ar_dir.exists():
            continue
        astar_by_ar[ar] = load_astar(ar_dir, ar)
        print(f"  AR={ar:.1f}: {len(astar_by_ar[ar])} alpha values")

    # Load reduced shear rate a*_ν via collision frequency (all ARs)
    print("Loading a*_ν (collision-frequency) data...")
    astar_nu_by_ar = {}
    for ar, dirname in AR_DIRS.items():
        ar_dir = LAMMPS_ROOT / dirname
        if not ar_dir.exists():
            continue
        astar_nu_by_ar[ar] = load_astar_nu(ar_dir, ar)
        print(f"  AR={ar:.1f}: {len(astar_nu_by_ar[ar])} alpha values")

    # Generate figures
    plot_stress(stress_by_ar,  out_dir / "lammps_usf_stress.png")
    plot_theta(stress_by_ar,   out_dir / "lammps_usf_theta.png")
    if nem_by_ar:
        plot_nematic(nem_by_ar, out_dir / "lammps_usf_nematic.png")
    if av_by_ar:
        plot_angvel(av_by_ar,   out_dir / "lammps_usf_angvel.png")
    if bagnold_by_ar:
        plot_bagnold_stress(bagnold_by_ar, out_dir / "lammps_usf_bagnold_stress.png")
    if astar_by_ar:
        plot_astar_sq(astar_by_ar, out_dir / "lammps_usf_astar_sq.png")
    if astar_nu_by_ar:
        plot_astar_sq_nu(astar_nu_by_ar, out_dir / "lammps_usf_astar_sq_nu.png")
    if stress_by_ar and astar_by_ar:
        plot_rheology(stress_by_ar, astar_by_ar,
                      out_dir / "lammps_usf_rheology.png")

    print("Done.")


if __name__ == "__main__":
    main()
