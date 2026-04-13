#!/usr/bin/env python3
"""Compare calibrated DSMC vs LAMMPS (calibrate_cpp, 2003p, always homogeneous).

Two panels for all alpha values (AR=2):
  1. Cumulative collisions per particle (tau) vs physical time
       DSMC tau   — as-is (counts each collision from both particles)
       LAMMPS cpp — multiplied by 2 to match DSMC convention
  2. Normalised total temperature T(t)/T(0) vs physical time (semilog)
       Both normalised at start of inelastic dynamics

Time alignment:
  DSMC   — elastic equilibration (t < 2.0) stripped; tau shifted accordingly
  LAMMPS — t=0 already post-equilibration (reset_timestep after nodiss run)
"""
import glob
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAMMPS_ROOT     = "LAMMPS_data/Equal/calibrate_cpp/modeB_e_sweep2"
CALIB_DIR       = "runs/calib_C_alpha"
C_TABLE_PATH    = "models/C_alpha_table_AR20.json"
FIGURES_DIR     = "figures"
LAMMPS_N        = 2003
LAMMPS_T_STRIDE = 500           # ~2.5M rows → ~5000 pts
LAMMPS_C_STRIDE = 400           # ~500k  rows → ~1250 pts
DSMC_EQUIL_T    = 2.0           # DSMC elastic equilibration end (physical time)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _read_dt_from_log(folder):
    log_path = os.path.join(folder, "log.lammps")
    with open(log_path) as f:
        for line in f:
            m = re.search(r"dt=([0-9eE+\-\.]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"Could not find 'dt=...' in {log_path}")


def _load_lammps(alpha):
    alpha_int = int(round(alpha * 100))
    folder = os.path.join(LAMMPS_ROOT, f"e_{alpha_int:03d}")

    T_candidates = glob.glob(os.path.join(folder, "hcs_temperatures_B_e*.dat"))
    if not T_candidates:
        return None, None
    T_raw = np.loadtxt(T_candidates[0], comments='#')[::LAMMPS_T_STRIDE]
    lmp_T = {
        "t":       T_raw[:, 0],
        "T_total": T_raw[:, 4],
    }

    coll_file = os.path.join(folder, "collision_events.dat")
    if not os.path.exists(coll_file):
        return lmp_T, None

    dt    = _read_dt_from_log(folder)
    C_raw = np.loadtxt(coll_file, comments='#')[::LAMMPS_C_STRIDE]
    lmp_C = {
        "t":   C_raw[:, 0] * dt,
        "cpp": C_raw[:, 2] / LAMMPS_N,
    }
    return lmp_T, lmp_C


def _load_dsmc(path):
    data = np.loadtxt(path)
    return {
        "t":       data[:, 0],
        "tau":     data[:, 1],
        "T_trans": data[:, 2],
        "T_rot":   data[:, 3],
        "T_total": data[:, 4],
    }


def _alpha_color(i, n, cmap_name="plasma"):
    cmap = matplotlib.colormaps[cmap_name]
    return cmap(0.1 + 0.8 * i / max(n - 1, 1))


def _format_ax(ax):
    ax.tick_params(axis='both', direction='in', which='both', right=True, top=True)
    ax.grid(True, linestyle='--', alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(C_TABLE_PATH) as f:
        c_table = json.load(f)

    PLOT_ALPHAS = {0.5, 0.6, 0.7, 0.8, 0.9}
    alpha_values = sorted(
        float(k.strip('()').split(',')[0].strip())
        for k in c_table.keys()
        if round(float(k.strip('()').split(',')[0].strip()), 2) in PLOT_ALPHAS
    )
    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, (ax_cpp, ax_T) = plt.subplots(1, 2, figsize=(14, 5))

    dsmc_kw = dict(linestyle='-',  linewidth=1.5, alpha=0.9)
    lmp_kw  = dict(linestyle='--', linewidth=1.2, alpha=0.85)

    xmax_dsmc = 0.0

    for i, alpha in enumerate(alpha_values):
        alpha_int = int(round(alpha * 100))
        C_opt  = c_table[f"({alpha:.3f}, 2.0)"]
        color  = _alpha_color(i, len(alpha_values))
        label  = f"e={alpha:.2f}"

        # ---- DSMC ----
        fname = f"calib_C{C_opt:.5f}_a{alpha_int:03d}_s42.txt"
        dpath = os.path.join(CALIB_DIR, fname)
        if not os.path.exists(dpath):
            print(f"  DSMC missing: {dpath}")
            continue
        dsmc = _load_dsmc(dpath)

        # Strip elastic equilibration; shift t and tau to start at (0, 0)
        i_eq = int(np.searchsorted(dsmc["t"], DSMC_EQUIL_T))
        i_eq = min(i_eq, len(dsmc["t"]) - 1)
        t_dsmc     = dsmc["t"][i_eq:]   - dsmc["t"][i_eq]
        tau_dsmc   = dsmc["tau"][i_eq:] - dsmc["tau"][i_eq]
        T0_dsmc    = dsmc["T_total"][i_eq]
        Tnorm_dsmc = dsmc["T_total"][i_eq:] / T0_dsmc
        xmax_dsmc  = max(xmax_dsmc, t_dsmc[-1])

        # ---- LAMMPS (calibrate_cpp, 2003p) ----
        lmp_T, lmp_C = _load_lammps(alpha)

        # ---- Panel 1: tau vs physical time ----
        ax_cpp.plot(t_dsmc, tau_dsmc, color=color, label=label, **dsmc_kw)
        if lmp_C is not None:
            ax_cpp.plot(lmp_C["t"], lmp_C["cpp"], color=color, **lmp_kw)

        # ---- Panel 2: T(t)/T(0) vs physical time ----
        ax_T.plot(t_dsmc, Tnorm_dsmc, color=color, label=label, **dsmc_kw)
        if lmp_T is not None:
            T0_lmp = lmp_T["T_total"][0]
            ax_T.plot(lmp_T["t"], lmp_T["T_total"] / T0_lmp, color=color, **lmp_kw)

    # Clip x-axis to DSMC run length
    ax_cpp.set_xlim(0, xmax_dsmc)
    ax_T.set_xlim(0, xmax_dsmc)

    # ---- Formatting ----
    ax_cpp.set_xlabel("Physical time", fontsize=11)
    ax_cpp.set_ylabel(r"Cumulative collisions per particle $\tau$", fontsize=11)
    ax_cpp.set_title("Collision rate", fontsize=13)
    _format_ax(ax_cpp)

    ax_T.set_xlabel("Physical time", fontsize=11)
    ax_T.set_ylabel(r"$T(t)\,/\,T(0)$", fontsize=12)
    ax_T.set_title("Cooling rate", fontsize=13)
    ax_T.set_yscale('log')
    _format_ax(ax_T)

    # ---- Legends ----
    style_handles = [
        Line2D([0], [0], color='k', linestyle='-',  linewidth=1.5, label='DSMC'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.2, label='LAMMPS (2003p, homogeneous)'),
    ]
    handles_alpha, labels_alpha = ax_cpp.get_legend_handles_labels()
    ax_cpp.legend(handles=handles_alpha, labels=labels_alpha,
                  fontsize=7.5, ncol=2, loc='upper left')
    ax_T.legend(handles=style_handles, fontsize=9, loc='upper right')

    fig.suptitle(
        "DSMC (—) vs LAMMPS calibrate_cpp 2003p (--), AR=2.0\n"
        r"Aligned at start of inelastic dynamics; LAMMPS cpp (no $\times 2$)",
        fontsize=12
    )
    plt.tight_layout()

    out = os.path.join(FIGURES_DIR, "compare_dsmc_lammps_AR2.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
