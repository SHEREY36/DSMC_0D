import numpy as np


def load_dsmc_results(file_path):
    """Load DSMC simulation output.

    Expected columns: t, tau (NColl/Np), T_trans, T_rot, T_total

    Returns (t, tau, T_trans, T_rot, T_total).
    """
    data = np.loadtxt(file_path)
    t = data[:, 0]
    tau = data[:, 1]
    T_trans = data[:, 2]
    T_rot = data[:, 3]
    T_total = data[:, 4]
    return t, tau, T_trans, T_rot, T_total


def load_dem_results(file_path):
    """Load DEM reference data.

    Expected columns: t, tau, T_trans, T_rot
    Returns (t, tau, T_trans, T_rot).
    """
    data = np.loadtxt(file_path)
    t = data[:, 0]
    tau = data[:, 1]
    T_trans = data[:, 2]
    T_rot = data[:, 3]
    return t, tau, T_trans, T_rot


def compute_normalized_temperature(T, T0=None):
    """Normalize temperature by initial value."""
    if T0 is None:
        T0 = T[0]
    return T / T0


def compute_temperature_ratio(T_trans, T_rot):
    """Compute theta = T_trans / T_rot."""
    return T_trans / T_rot
