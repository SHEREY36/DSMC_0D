#!/usr/bin/env python3
"""Validation plots comparing calibrated DSMC against LAMMPS HCS data.

Three panels for all alpha values (AR=2):
  1. Collision frequency: tau vs physical time (DSMC) and cpp vs LAMMPS time
     plotted with independent y-axes (DSMC left, LAMMPS right) since the two
     time units differ.
  2. Temperature ratio theta = T_trans / T_rot vs tau (cumulative cpp)
  3. Normalised total temperature T(t)/T(0) vs tau (cumulative cpp)

Using tau as the x-axis for panels 2 and 3 eliminates the time-unit
discrepancy: both systems obey the same per-collision physics.

DSMC data: optimal-C files from runs/calib_C_alpha/
LAMMPS data: LAMMPS_data/modeB_e_sweep2a/modeB_e_sweep/
"""
import glob
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAMMPS_ROOT   = "LAMMPS_data/Equal/Calibrate_r/modeB_e_sweep2"
CALIB_DIR     = "runs/calib_C_alpha"
C_TABLE_PATH  = "models/C_alpha_table_AR20.json"
FIGURES_DIR   = "figures"
LAMMPS_N      = 40000          # particles in LAMMPS
LAMMPS_STRIDE = 900            # downsample temperature file (906k rows → ~1000 pts)
COLL_STRIDE   = 180            # downsample collision file   (181k rows → ~1000 pts)
DSMC_T_MAX    = 300.0          # DSMC run length



# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_dsmc(path):
    """Load DSMC output. Columns: t, tau (NColl/Np), T_trans, T_rot, T_total."""
    data = np.loadtxt(path)
    return {
        "t":       data[:, 0],
        "tau":     data[:, 1],
        "T_trans": data[:, 2],
        "T_rot":   data[:, 3],
        "T_total": data[:, 4],
    }


def _load_lammps_temperatures(alpha):
    """Load LAMMPS HCS temperature file (downsampled).

    Columns: time T_trans T_rot T_tr/T_rot T_total
    """
    alpha_int = int(round(alpha * 100))
    folder = os.path.join(LAMMPS_ROOT, f"e_{alpha_int:03d}")
    candidates = glob.glob(os.path.join(folder, "hcs_temperatures_B_e*.dat"))
    if not candidates:
        return None
    raw = np.loadtxt(candidates[0], comments='#')[::LAMMPS_STRIDE]
    return {
        "t":       raw[:, 0],
        "T_trans": raw[:, 1],
        "T_rot":   raw[:, 2],
        "theta":   raw[:, 3],
        "T_total": raw[:, 4],
    }


def _read_dt_from_log(folder):
    """Read timestep dt from log.lammps (searches for 'dt=...' pattern)."""
    log_path = os.path.join(folder, "log.lammps")
    with open(log_path, "r") as f:
        for line in f:
            m = re.search(r"dt=([0-9eE+\-\.]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"Could not find 'dt=...' in {log_path}")


def _load_lammps_collisions(alpha):
    """Load LAMMPS collision events, convert to (time, cpp).

    cpp = cumulative collisions per particle = global_cumulative_events / N_part
    """
    alpha_int = int(round(alpha * 100))
    folder = os.path.join(LAMMPS_ROOT, f"e_{alpha_int:03d}")
    coll_file = os.path.join(folder, "collision_events.dat")
    if not os.path.exists(coll_file):
        return None
    dt = _read_dt_from_log(folder)
    raw = np.loadtxt(coll_file, comments='#')[::COLL_STRIDE]
    t   = raw[:, 0] * dt
    cpp = raw[:, 2] / LAMMPS_N
    return {"t": t, "cpp": cpp}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _alpha_color(alpha, alpha_values, cmap_name="plasma"):
    cmap = matplotlib.colormaps[cmap_name]
    idx  = list(alpha_values).index(alpha)
    return cmap(0.1 + 0.8 * idx / max(len(alpha_values) - 1, 1))


def _format_ax(ax):
    ax.tick_params(axis='both', direction='in', which='both', right=True, top=True)
    ax.grid(True, linestyle='--', alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(C_TABLE_PATH) as f:
        c_table = json.load(f)

    # Collect and sort alpha values
    alpha_values = sorted(
        float(k.strip('()').split(',')[0].strip()) for k in c_table.keys()
    )

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax_coll, ax_theta, ax_T = axes

    # Twin axis for collision panel: DSMC tau (left) and LAMMPS cpp (right)
    ax_coll_lmp = ax_coll.twinx()

    _lammps_kw = dict(linestyle='--', linewidth=1.2, alpha=0.85)
    _dsmc_kw   = dict(linestyle='-',  linewidth=1.5, alpha=0.9)

    for alpha in alpha_values:
        alpha_int = int(round(alpha * 100))
        key   = f"({alpha:.3f}, 2.0)"
        C_opt = c_table[key]
        color = _alpha_color(alpha, alpha_values)

        # ---- DSMC ----
        fname = f"calib_C{C_opt:.5f}_a{alpha_int:03d}_s42.txt"
        dpath = os.path.join(CALIB_DIR, fname)
        if not os.path.exists(dpath):
            print(f"  Warning: DSMC file not found: {dpath}")
            continue
        dsmc = _load_dsmc(dpath)

        # ---- LAMMPS temperatures ----
        lmp_T = _load_lammps_temperatures(alpha)

        # ---- LAMMPS collisions ----
        lmp_C = _load_lammps_collisions(alpha)

        # Compute LAMMPS cpp aligned to temperature time axis
        # (interpolate cpp at each temperature sample time so we can use
        #  cpp as x-axis for panels 2 and 3)
        lmp_cpp_at_T = None
        if lmp_T is not None and lmp_C is not None:
            lmp_cpp_at_T = np.interp(lmp_T["t"], lmp_C["t"], lmp_C["cpp"])

        label = f"e={alpha:.2f}"

        # ---- Panel 1: tau vs physical time ----
        # DSMC: left axis; LAMMPS: right axis (different time units)
        ax_coll.plot(dsmc["t"], dsmc["tau"],
                     color=color, label=label, **_dsmc_kw)
        if lmp_C is not None:
            ax_coll_lmp.plot(lmp_C["t"], lmp_C["cpp"],
                             color=color, **_lammps_kw)

        # ---- Panel 2: theta vs tau (cumulative cpp) ----
        theta_dsmc = dsmc["T_trans"] / dsmc["T_rot"]
        ax_theta.plot(dsmc["tau"], theta_dsmc,
                      color=color, label=label, **_dsmc_kw)
        if lmp_T is not None and lmp_cpp_at_T is not None:
            ax_theta.plot(lmp_cpp_at_T, lmp_T["theta"],
                          color=color, **_lammps_kw)

        # ---- Panel 3: T/T0 vs tau (cumulative cpp) ----
        T0_dsmc = dsmc["T_total"][0]
        ax_T.plot(dsmc["tau"], dsmc["T_total"] / T0_dsmc,
                  color=color, label=label, **_dsmc_kw)
        if lmp_T is not None and lmp_cpp_at_T is not None:
            T0_lmp = lmp_T["T_total"][0]
            ax_T.plot(lmp_cpp_at_T, lmp_T["T_total"] / T0_lmp,
                      color=color, **_lammps_kw)

    # ---- Panel 1 formatting ----
    ax_coll.set_xlabel("DSMC time", fontsize=11)
    ax_coll.set_ylabel(r"DSMC $\tau$ (NColl/Np)", fontsize=11, color='k')
    ax_coll_lmp.set_ylabel("LAMMPS cpp", fontsize=11, color='dimgray')
    ax_coll_lmp.tick_params(axis='y', labelcolor='dimgray')
    ax_coll.set_title("Collision frequency", fontsize=13)
    ax_coll.set_xlabel("Time", fontsize=11)
    _format_ax(ax_coll)

    # ---- Panel 2 formatting ----
    ax_theta.set_xlabel(r"Cumulative collisions per particle $\tau$", fontsize=11)
    ax_theta.set_ylabel(r"$\theta = T_\mathrm{tr} / T_\mathrm{rot}$", fontsize=12)
    ax_theta.set_title("Temperature ratio", fontsize=13)
    _format_ax(ax_theta)

    # ---- Panel 3 formatting ----
    ax_T.set_xlabel(r"Cumulative collisions per particle $\tau$", fontsize=11)
    ax_T.set_ylabel(r"$T(t) / T(0)$", fontsize=12)
    ax_T.set_title("Cooling rate", fontsize=13)
    ax_T.set_yscale('log')
    _format_ax(ax_T)

    # ---- Legends ----
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='k', linestyle='-',  linewidth=1.5, label='DSMC'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=1.2, label='LAMMPS'),
    ]
    handles_alpha, labels_alpha = ax_theta.get_legend_handles_labels()
    ax_theta.legend(handles=handles_alpha, labels=labels_alpha,
                    fontsize=7.5, ncol=2, loc='lower right')
    ax_coll.legend(handles=style_handles, fontsize=9, loc='upper left')
    ax_T.legend(handles=style_handles, fontsize=9, loc='upper right')

    fig.suptitle(
        "Validation: DSMC (solid) vs LAMMPS (dashed), AR=2.0\n"
        r"Panels 2–3: $x$-axis = $\tau$ (cpp), panels share the same per-collision physics",
        fontsize=12
    )
    plt.tight_layout()

    out_path = os.path.join(FIGURES_DIR, "validation_C_alpha_AR2.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
