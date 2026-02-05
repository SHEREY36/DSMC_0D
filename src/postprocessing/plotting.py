import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from .analysis import (
    load_dsmc_results, load_dem_results,
    compute_normalized_temperature, compute_temperature_ratio
)


def setup_axes(ax, aspect=1.0):
    """Apply consistent formatting to an axis."""
    formatter = FuncFormatter(lambda x, _: '0' if x == 0 else f'{x:.1f}')
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', direction='in', which='both',
                   right=True, top=True)


def plot_temperature_evolution(dsmc_files, labels, output_path,
                               dem_files=None, dem_labels=None):
    """Plot T_total/T0 vs tau for multiple DSMC realizations.

    Optionally overlays DEM reference data.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    for file_path, label in zip(dsmc_files, labels):
        t, tau, T_trans, T_rot, T_total = load_dsmc_results(file_path)
        T_norm = compute_normalized_temperature(T_total)
        ax.plot(tau, T_norm, label=f"DSMC {label}")

    if dem_files:
        dem_labels = dem_labels or [f"DEM {i}" for i in range(len(dem_files))]
        for file_path, label in zip(dem_files, dem_labels):
            t, tau, T_trans, T_rot = load_dem_results(file_path)
            T_total_dem = (3 * T_trans + 2 * T_rot) / 5.0
            T_norm = compute_normalized_temperature(T_total_dem)
            ax.plot(tau, T_norm, '--', label=label)

    ax.set_xlabel(r"$\tau$ (collisions per particle)")
    ax.set_ylabel(r"$T / T_0$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Temperature evolution plot saved to {output_path}")


def plot_temperature_components(dsmc_files, labels, output_path):
    """Plot T_trans and T_rot separately vs tau."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for file_path, label in zip(dsmc_files, labels):
        t, tau, T_trans, T_rot, T_total = load_dsmc_results(file_path)
        T_trans_norm = compute_normalized_temperature(T_trans)
        T_rot_norm = compute_normalized_temperature(T_rot)
        ax1.plot(tau, T_trans_norm, label=label)
        ax2.plot(tau, T_rot_norm, label=label)

    ax1.set_xlabel(r"$\tau$")
    ax1.set_ylabel(r"$T_{tr} / T_{tr,0}$")
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel(r"$\tau$")
    ax2.set_ylabel(r"$T_{rot} / T_{rot,0}$")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Temperature components plot saved to {output_path}")


def plot_temperature_ratio_evolution(dsmc_files, labels, output_path):
    """Plot theta = T_trans/T_rot vs tau."""
    fig, ax = plt.subplots(figsize=(6, 4))

    for file_path, label in zip(dsmc_files, labels):
        t, tau, T_trans, T_rot, T_total = load_dsmc_results(file_path)
        theta = compute_temperature_ratio(T_trans, T_rot)
        ax.plot(tau, theta, label=label)

    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\theta = T_{tr} / T_{rot}$")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Temperature ratio plot saved to {output_path}")
