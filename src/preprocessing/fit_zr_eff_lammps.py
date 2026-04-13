"""Extract Z_R_eff and theta* from LAMMPS HCS data for AR=2.0.

Reads modeB HCS temperature and collision data, converts to collisions/particle
time axis, then fits the two-temperature model to extract Z_R_eff(alpha) and
theta*(alpha). Saves results to models/zr_eff_table_AR20.json and generates
validation figures.

Run from project root:
    python src/preprocessing/fit_zr_eff_lammps.py
"""

import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root and LAMMPS_data are importable regardless of working directory
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.normpath(os.path.join(_here, '..', '..'))
_lammps_data_dir = os.path.join(_project_root, 'LAMMPS_data')
for _p in [_project_root, _lammps_data_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fit_zr_eff import (  # noqa: E402
    extract_theta_star, extract_gamma, extract_lambda_theta,
    compute_model_parameters, forward_integrate,
)
from src.preprocessing.zr_eff_table import save_zr_eff_table  # noqa: E402

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
N_PART = 40_000
AR = 2.0
ALPHA_VALUES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DATA_ROOT = os.path.join(_project_root, 'LAMMPS_data', 'modeB_e_sweep2a', 'modeB_e_sweep')
STRIDE = 100   # subsample temperature rows for speed


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def _alpha_dir(alpha):
    return f"e_{int(round(alpha * 100)):03d}"


def _alpha_str(alpha):
    """Format alpha as LAMMPS uses in filenames: 0.5, 0.55, ..., 0.95."""
    s = f"{alpha:g}"
    return s


def _read_dt_from_log(case_dir):
    log_path = os.path.join(case_dir, 'log.lammps')
    with open(log_path, 'r') as f:
        for line in f:
            m = re.search(r'dt=([0-9eE+\-\.]+)', line)
            if m:
                return float(m.group(1))
    raise RuntimeError(f"Could not find 'dt=...' in {log_path}")


def load_lammps_case(data_root, alpha):
    """Load temperature and collision data for one alpha value.

    Returns (tau, T_tr, T_rot) arrays aligned on the same time grid,
    subsampled by STRIDE from the temperature file.
    """
    case_dir = os.path.join(data_root, _alpha_dir(alpha))

    # 1. Timestep from log
    dt = _read_dt_from_log(case_dir)

    # 2. Collision events: step → tau = cumulative / N_PART
    coll_file = os.path.join(case_dir, 'collision_events.dat')
    coll_data = np.loadtxt(coll_file, comments='#')
    coll_step = coll_data[:, 0]
    coll_cumulative = coll_data[:, 2]
    coll_time = coll_step * dt
    coll_tau = coll_cumulative / N_PART

    # 3. Temperature file (subsampled)
    temp_fname = f"hcs_temperatures_B_e{_alpha_str(alpha)}.dat"
    temp_file = os.path.join(case_dir, temp_fname)
    temp_data = np.loadtxt(temp_file, comments='#')[::STRIDE]
    temp_time = temp_data[:, 0]
    T_tr = temp_data[:, 1]
    T_rot = temp_data[:, 2]

    # 4. Interpolate tau at each temperature time point
    tau = np.interp(temp_time, coll_time, coll_tau)

    return tau, T_tr, T_rot


# ------------------------------------------------------------------
# Main extraction loop
# ------------------------------------------------------------------

def run_extraction(data_root, alpha_values):
    results = {}
    for alpha in alpha_values:
        print(f"\n{'='*55}")
        print(f"alpha = {alpha:.2f}")
        print(f"{'='*55}")

        tau, T_tr, T_rot = load_lammps_case(data_root, alpha)
        print(f"  Loaded {len(tau)} time points; tau range [{tau[0]:.2f}, {tau[-1]:.2f}]")

        theta_star, _ = extract_theta_star(tau, T_tr, T_rot)
        gamma_val, _, T_g0 = extract_gamma(tau, T_tr, T_rot)
        lambda_theta, _, _ = extract_lambda_theta(tau, T_tr, T_rot, theta_star)

        params = compute_model_parameters(theta_star, gamma_val, lambda_theta)

        # Forward integration for validation
        tau_model, T_tr_model, T_rot_model = forward_integrate(
            params, T_tr[0], T_rot[0], [tau[0], tau[-1]]
        )

        results[alpha] = {
            'params': params,
            'data': (tau, T_tr, T_rot),
            'model': (tau_model, T_tr_model, T_rot_model),
        }

    return results


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------

def plot_validation(results, out_path):
    """2×5 grid of theta(tau): data scatter + model + theta* dashed."""
    alphas = sorted(results.keys())
    n = len(alphas)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows),
                             sharex=False, sharey=False)
    axes = np.array(axes).flatten()

    for i, alpha in enumerate(alphas):
        ax = axes[i]
        tau, T_tr, T_rot = results[alpha]['data']
        tau_m, T_tr_m, T_rot_m = results[alpha]['model']
        params = results[alpha]['params']
        theta_star = params['theta_star']
        Z_R_eff = params['Z_R_eff']

        theta_data = T_tr / T_rot
        theta_model = T_tr_m / T_rot_m

        ax.scatter(tau, theta_data, s=0.5, alpha=0.15, color='steelblue', rasterized=True)
        ax.plot(tau_m, theta_model, 'r-', linewidth=1.5, label='Model')
        ax.axhline(theta_star, color='k', linestyle='--', linewidth=1.0,
                   label=rf'$\theta^*={theta_star:.3f}$')

        ax.set_title(rf'$\alpha={alpha:.2f}$, $Z_R^{{eff}}={Z_R_eff:.2f}$', fontsize=9)
        ax.set_xlabel(r'$\tau$ (coll./particle)', fontsize=8)
        ax.set_ylabel(r'$\theta = T_{tr}/T_{rot}$', fontsize=8)
        ax.legend(fontsize=7, markerscale=3)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r'$Z_R^{eff}$ extraction validation — AR=2.0', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out_path}")


def plot_params_vs_alpha(results, out_path):
    """theta*(alpha) and Z_R_eff(alpha) side by side."""
    alphas = sorted(results.keys())
    theta_stars = [results[a]['params']['theta_star'] for a in alphas]
    zr_effs = [results[a]['params']['Z_R_eff'] for a in alphas]
    gammas = [results[a]['params']['gamma'] for a in alphas]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].plot(alphas, theta_stars, 'o-', color='steelblue')
    axes[0].set_xlabel(r'$\alpha$ (coefficient of restitution)')
    axes[0].set_ylabel(r'$\theta^* = T_{tr}^*/T_{rot}^*$')
    axes[0].set_title(r'Asymptotic temperature ratio $\theta^*(\alpha)$')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas, zr_effs, 's-', color='tomato')
    axes[1].set_xlabel(r'$\alpha$ (coefficient of restitution)')
    axes[1].set_ylabel(r'$Z_R^{eff}$')
    axes[1].set_title(r'Effective rotational collision number $Z_R^{eff}(\alpha)$')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(alphas, gammas, '^-', color='seagreen')
    axes[2].set_xlabel(r'$\alpha$ (coefficient of restitution)')
    axes[2].set_ylabel(r'$\gamma$ (cooling rate per collision)')
    axes[2].set_title(r'Total cooling rate $\gamma(\alpha)$')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    models_dir = os.path.join(_project_root, 'models')
    figures_dir = os.path.join(_project_root, 'figures')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    results = run_extraction(DATA_ROOT, ALPHA_VALUES)

    # Build table for JSON
    table = {}
    for alpha, res in results.items():
        p = res['params']
        key = f"({alpha:.3f}, {AR:.1f})"
        table[key] = {
            'theta_star': p['theta_star'],
            'Z_R_eff': p['Z_R_eff'],
            'gamma': p['gamma'],
            'lambda_theta': p['lambda_theta'],
            'K_eff': p['K_eff'],
            'Psi_tr': p['Psi_tr'],
            'Psi_rot': p['Psi_rot'],
        }

    table_path = os.path.join(models_dir, 'zr_eff_table_AR20.json')
    save_zr_eff_table(table, table_path)
    print(f"\nSaved Z_R_eff table: {table_path}")

    plot_validation(results, os.path.join(figures_dir, 'zr_eff_validation.png'))
    plot_params_vs_alpha(results, os.path.join(figures_dir, 'zr_eff_params_vs_alpha.png'))

    # Summary table
    print(f"\n{'alpha':>8} {'theta*':>10} {'Z_R_eff':>10} {'gamma':>10}")
    print('-' * 42)
    for alpha in sorted(results.keys()):
        p = results[alpha]['params']
        print(f"{alpha:8.2f} {p['theta_star']:10.4f} {p['Z_R_eff']:10.3f} {p['gamma']:10.5f}")


if __name__ == '__main__':
    main()
