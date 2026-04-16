#!/usr/bin/env python3
"""Diagnostic: HCS cooling rate and collision rate — LAMMPS vs calibrated DSMC, AR=2.

Two separate figures (each saved individually):
  Fig 1 — T(t)/T(0) vs physical time (semilog y)
  Fig 2 — Cumulative collisions per particle (tau) vs physical time

LAMMPS = solid line, DSMC = dashed line.  One colour per alpha.
x-axis clipped to T_MAX=300 (extent of DSMC calibration runs).

Loading strategy for LAMMPS tau is identical to the "LAMMPS (new)" series in
compare_hcs_cpp_methods.py:
  tau = 2.0 * collision_count[:, 2] / N_part
  t   = collision_count[:, 0] * dt     (dt and N_part read from log.lammps)
  Clips to T_MAX, then prepends (0, 0) if the first recorded step > 0.

Aspect convention: both axes are rendered as square boxes using the same
set_aspect_one() function as compare_hcs_cpp_methods.py.

Usage
-----
  python plot_hcs_cooling_diagnostic.py           # plot only
  python plot_hcs_cooling_diagnostic.py --measure # also print correction-factor table
"""
import argparse
import glob
import json
import os
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'legend.fontsize': 8.5,
    'lines.linewidth': 1.6,
})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAMMPS_HCS_ROOT  = "LAMMPS_data/HCS/calibrate_cpp/modeB_e_sweep2"
SWEEP_DIR        = "runs/AR2_eta_sweep"   # preferred: new corrected runs
CALIB_DIR        = "runs/calib_C_alpha"   # fallback: old calibration runs
C_TABLE_PATH     = "models/C_alpha_table_AR20.json"
FIGURES_DIR      = "figures"

FALLBACK_N_PART  = 2003            # used only if log parse fails
LAMMPS_T_STRIDE  = 500             # row stride for temperature files (~1.8 M rows)
DSMC_EQUIL_T     = 2.0             # elastic equilibration time in DSMC runs
T_MAX            = 300.0           # x-axis clip [same physical-time units]


# ---------------------------------------------------------------------------
# Helpers (identical to compare_hcs_cpp_methods.py)
# ---------------------------------------------------------------------------

def set_aspect_one(ax):
    """Make the axes box square using the current data limits."""
    xr = ax.get_xlim()[1] - ax.get_xlim()[0]
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    if yr > 0:
        ax.set_aspect(xr / yr, adjustable="box")


def _prepend_origin_if_needed(x, y):
    if len(x) == 0:
        return x, y
    if x[0] <= 0.0:
        return x, y
    return np.insert(x, 0, 0.0), np.insert(y, 0, 0.0)


def _clip_time_window(x, y, t_max):
    mask = np.asarray(x) <= t_max
    return np.asarray(x)[mask], np.asarray(y)[mask]


def _format_ax(ax):
    ax.tick_params(axis='both', direction='in', which='both', right=True, top=True)
    ax.grid(True, linestyle='--', alpha=0.3)


# ---------------------------------------------------------------------------
# Log readers (identical to compare_hcs_cpp_methods.py)
# ---------------------------------------------------------------------------

def _read_dt_from_log(folder):
    log_path = os.path.join(folder, "log.lammps")
    with open(log_path) as fh:
        for line in fh:
            m = re.search(r"dt=([0-9eE+\-\.]+)", line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"Could not find 'dt=...' in {log_path}")


def _read_npart_from_log(folder):
    log_path = os.path.join(folder, "log.lammps")
    with open(log_path) as fh:
        for line in fh:
            m = re.search(r"\bN=(\d+),\s*phi=", line)
            if m:
                return int(m.group(1))
    return FALLBACK_N_PART


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_lammps_tau(alpha):
    """Load LAMMPS collision_count.dat — exact same logic as compare_hcs_cpp_methods.py
    '_load_lammps_counter(..., "collision_count.dat", T_MAX)'."""
    alpha_int = int(round(alpha * 100))
    folder    = os.path.join(LAMMPS_HCS_ROOT, f"e_{alpha_int:03d}")
    path      = os.path.join(folder, "collision_count.dat")
    if not os.path.exists(path):
        print(f"  collision_count.dat missing for alpha={alpha:.2f}")
        return None, None

    raw    = np.loadtxt(path, comments="#")          # no stride — all rows
    dt     = _read_dt_from_log(folder)
    n_part = _read_npart_from_log(folder)

    t   = raw[:, 0] * dt
    tau = 2.0 * raw[:, 2] / float(n_part)

    t, tau = _clip_time_window(t, tau, T_MAX)
    t, tau = _prepend_origin_if_needed(t, tau)
    return t, tau


def load_lammps_temperature(alpha):
    """Load LAMMPS hcs_temperatures file (strided — ~1.8 M rows)."""
    alpha_int  = int(round(alpha * 100))
    folder     = os.path.join(LAMMPS_HCS_ROOT, f"e_{alpha_int:03d}")
    candidates = glob.glob(os.path.join(folder, "hcs_temperatures_B_e*.dat"))
    if not candidates:
        print(f"  LAMMPS temperature file missing for alpha={alpha:.2f}")
        return None, None

    raw = np.loadtxt(candidates[0], comments='#')[::LAMMPS_T_STRIDE]
    t       = raw[:, 0]
    T_total = raw[:, 4]
    t, T_total = _clip_time_window(t, T_total, T_MAX)
    return t, T_total


def find_dsmc_file(alpha, c_table):
    """Return (path, C_target) for this alpha.

    Preference order:
      1. runs/AR2_eta_sweep/alpha_NNN/results/AR2_CORNNN_R1.txt  (new corrected runs)
      2. runs/calib_C_alpha/calib_C*_aNNN_s42.txt                (old calibration runs)
    """
    alpha_int = int(round(alpha * 100))
    key = f"({alpha:.3f}, 2.0)"
    if key not in c_table:
        return None, None
    C_target = float(c_table[key])

    # 1. New sweep output
    sweep_path = os.path.join(
        SWEEP_DIR, f"alpha_{alpha_int:03d}", "results", f"AR2_COR{alpha_int}_R1.txt"
    )
    if os.path.exists(sweep_path):
        return sweep_path, C_target

    # 2. Fallback: old calibration directory
    candidates = glob.glob(
        os.path.join(CALIB_DIR, f"calib_C*_a{alpha_int:03d}_s42.txt")
    )
    if not candidates:
        return None, C_target
    best = min(
        candidates,
        key=lambda p: abs(
            float(re.search(r"calib_C([0-9.]+)_a", os.path.basename(p)).group(1))
            - C_target
        ),
    )
    return best, C_target


def load_dsmc(path):
    data = np.loadtxt(path)
    t, tau, T_total = data[:, 0], data[:, 1], data[:, 4]

    i_eq  = min(int(np.searchsorted(t, DSMC_EQUIL_T)), len(t) - 1)
    t     = t[i_eq:]   - t[i_eq]
    tau   = tau[i_eq:] - tau[i_eq]
    T_total = T_total[i_eq:]

    t_T, T_total = _clip_time_window(t, T_total, T_MAX)
    t_tau, tau   = _clip_time_window(t, tau,     T_MAX)
    t_T,   T_total = _prepend_origin_if_needed(t_T,   T_total)
    t_tau, tau     = _prepend_origin_if_needed(t_tau, tau)
    return t_T, T_total, t_tau, tau


# ---------------------------------------------------------------------------
# Correction-factor measurement
# ---------------------------------------------------------------------------

def _fit_slope(t, y, t_lo=50.0, t_hi=250.0):
    """Linear least-squares slope of y vs t in the window [t_lo, t_hi]."""
    mask = (t >= t_lo) & (t <= t_hi)
    if mask.sum() < 5:
        return np.nan
    return float(np.polyfit(t[mask], y[mask], 1)[0])


def measure_corrections(c_table, alpha_values):
    """Print a table of per-alpha correction factors.

    gamma_max_scale(α) = |d(logT)/dt|_LAMMPS / |d(logT)/dt|_DSMC
      → >1 means DSMC cools too slowly; need to increase gamma_max.
      Computed for α = 0.50–0.95 (all inelastic; LAMMPS T(t)/T(0) reliable for 0.50).

    sigma_c_scale(α)   = (dτ/dt)_LAMMPS / (dτ/dt)_DSMC
      → >1 means DSMC collision rate is too low; need to increase sigma_c.
      Computed for α = 0.55–0.95 (α=0.50 excluded: LAMMPS clustering corrupts τ).
    """
    print("\n" + "=" * 72)
    print(f"  {'alpha':>6}  {'rate_LAMMPS':>12}  {'rate_DSMC':>12}  "
          f"{'gmax_scale':>11}  {'tau_LAMMPS':>11}  {'tau_DSMC':>11}  "
          f"{'sc_scale':>9}")
    print("=" * 72)

    gmax_ratios = []
    sc_ratios   = []

    for alpha in alpha_values:
        alpha_int = int(round(alpha * 100))

        # ---- LAMMPS temperature ----
        t_lT, T_lT = load_lammps_temperature(alpha)
        lT_rate = np.nan
        if t_lT is not None and len(t_lT) > 10:
            logT = np.log(np.maximum(T_lT / T_lT[0], 1e-15))
            lT_rate = _fit_slope(t_lT, logT)

        # ---- LAMMPS tau ----
        t_lC, tau_lC = load_lammps_tau(alpha)
        lC_slope = np.nan
        if t_lC is not None:
            lC_slope = _fit_slope(t_lC, tau_lC)

        # ---- DSMC ----
        dpath, _ = find_dsmc_file(alpha, c_table)
        dT_rate = np.nan
        dC_slope = np.nan
        if dpath is not None:
            t_dT, T_dT, t_dtau, tau_d = load_dsmc(dpath)
            if len(t_dT) > 10:
                logT_d = np.log(np.maximum(T_dT / T_dT[0], 1e-15))
                dT_rate = _fit_slope(t_dT, logT_d)
            dC_slope = _fit_slope(t_dtau, tau_d)

        # ---- ratios ----
        gmax_r = (lT_rate / dT_rate) if (np.isfinite(lT_rate) and np.isfinite(dT_rate)
                                          and dT_rate != 0) else np.nan
        sc_r   = (lC_slope / dC_slope) if (np.isfinite(lC_slope) and np.isfinite(dC_slope)
                                            and dC_slope != 0) else np.nan

        # accumulate (gamma_max: all α; sigma_c: skip α=0.50)
        if np.isfinite(gmax_r):
            gmax_ratios.append(gmax_r)
        if alpha > 0.50 and np.isfinite(sc_r):
            sc_ratios.append(sc_r)

        print(f"  {alpha:>6.2f}  {lT_rate:>12.5f}  {dT_rate:>12.5f}  "
              f"{gmax_r:>11.4f}  {lC_slope:>11.5f}  {dC_slope:>11.5f}  "
              f"{sc_r:>9.4f}")

    mean_gmax = float(np.nanmean(gmax_ratios))
    mean_sc   = float(np.nanmean(sc_ratios))
    print("=" * 72)
    print(f"  Mean gamma_max_scale (α 0.50–0.95) : {mean_gmax:.4f}")
    print(f"  Mean sigma_c_scale   (α 0.55–0.95) : {mean_sc:.4f}")
    print("=" * 72 + "\n")
    return mean_gmax, mean_sc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--measure', action='store_true',
                        help='Print gamma_max_scale and sigma_c_scale correction table')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    with open(C_TABLE_PATH) as fh:
        c_table = json.load(fh)

    alpha_values = sorted([
        float(k.strip('()').split(',')[0].strip())
        for k in c_table
        if float(k.strip('()').split(',')[0].strip()) < 1.0
    ])
    n_alpha = len(alpha_values)
    print(f"Alpha values: {alpha_values}")

    if args.measure:
        measure_corrections(c_table, alpha_values)
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    cmap   = matplotlib.colormaps['tab10']
    colors = [cmap(i) for i in range(n_alpha)]

    lmp_kw  = dict(linestyle='-',  linewidth=1.6, alpha=0.92)
    dsmc_kw = dict(linestyle='--', linewidth=1.5, alpha=0.92)

    fig1, ax_T   = plt.subplots(figsize=(7, 7))
    fig2, ax_tau = plt.subplots(figsize=(7, 7))

    for color, alpha in zip(colors, alpha_values):
        label = rf"$\alpha={alpha:.2f}$"

        # ---- LAMMPS temperature ----
        t_lT, T_lT = load_lammps_temperature(alpha)
        if t_lT is not None:
            T0_lmp = T_lT[0]
            ax_T.plot(t_lT, T_lT / T0_lmp, color=color, label=label, **lmp_kw)

        # ---- LAMMPS tau (exact compare_hcs_cpp_methods strategy) ----
        t_lC, tau_lC = load_lammps_tau(alpha)
        if t_lC is not None:
            ax_tau.plot(t_lC, tau_lC, color=color, label=label, **lmp_kw)

        # ---- DSMC ----
        dpath, C_target = find_dsmc_file(alpha, c_table)
        if dpath is None:
            print(f"  DSMC missing for alpha={alpha:.2f}")
            continue

        t_dT, T_dT, t_dtau, tau_d = load_dsmc(dpath)
        T0_d = T_dT[0]
        ax_T.plot(  t_dT,   T_dT / T0_d, color=color, **dsmc_kw)
        ax_tau.plot(t_dtau, tau_d,        color=color, **dsmc_kw)

        print(f"  alpha={alpha:.2f}  C_target={C_target:.5f}  "
              f"file={os.path.basename(dpath)}")

    # ---- Figure 1: Cooling rate ----
    ax_T.set_xlabel("Physical time $t$")
    ax_T.set_ylabel(r"$T(t)\,/\,T(0)$")
    ax_T.set_yscale('log')
    ax_T.set_xlim(0, T_MAX)
    ax_T.set_title(
        r"HCS cooling rate — AR=2, all $\alpha$"
        "\nLAMMPS (solid)   vs   DSMC calibrated (dashed)",
        fontsize=11,
    )
    _format_ax(ax_T)

    handles_alpha, labels_alpha = ax_T.get_legend_handles_labels()
    leg1 = ax_T.legend(handles=handles_alpha, labels=labels_alpha,
                       fontsize=8, ncol=2, loc='upper right',
                       title=r'$\alpha$', title_fontsize=9, framealpha=0.85)
    ax_T.add_artist(leg1)
    ax_T.legend(handles=[
        Line2D([0], [0], color='k', ls='-',  lw=1.6, label='LAMMPS'),
        Line2D([0], [0], color='k', ls='--', lw=1.5, label='DSMC'),
    ], loc='lower left', framealpha=0.85)

    set_aspect_one(ax_T)
    fig1.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "hcs_cooling_diagnostic_AR2.png")
    fig1.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out1}")

    # ---- Figure 2: Collision rate ----
    ax_tau.set_xlabel("Physical time $t$")
    ax_tau.set_ylabel(r"Cumulative collisions per particle $\tau$")
    ax_tau.set_xlim(0, T_MAX)
    ax_tau.set_title(
        r"HCS collision rate — AR=2, all $\alpha$"
        "\nLAMMPS (solid)   vs   DSMC calibrated (dashed)",
        fontsize=11,
    )
    _format_ax(ax_tau)

    handles_alpha2, labels_alpha2 = ax_tau.get_legend_handles_labels()
    leg2 = ax_tau.legend(handles=handles_alpha2, labels=labels_alpha2,
                         fontsize=8, ncol=2, loc='upper left',
                         title=r'$\alpha$', title_fontsize=9, framealpha=0.85)
    ax_tau.add_artist(leg2)
    ax_tau.legend(handles=[
        Line2D([0], [0], color='k', ls='-',  lw=1.6, label='LAMMPS'),
        Line2D([0], [0], color='k', ls='--', lw=1.5, label='DSMC'),
    ], loc='lower right', framealpha=0.85)

    set_aspect_one(ax_tau)
    fig2.tight_layout()
    out2 = os.path.join(FIGURES_DIR, "hcs_collision_rate_diagnostic_AR2.png")
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"Saved: {out2}")

    plt.show()


if __name__ == "__main__":
    main()
