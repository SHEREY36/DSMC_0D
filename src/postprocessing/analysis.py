import numpy as np
import os


def load_dsmc_results(file_path):
    """Load DSMC simulation output.

    Expected columns: t, tau (NColl/Np), T_trans, T_rot, T_total

    Returns (t, tau, T_trans, T_rot, T_total).
    """
    if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
        raise ValueError(f"Invalid or empty DSMC result file: {file_path}")
    data = np.loadtxt(file_path)
    data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] < 5:
        raise ValueError(
            f"Invalid or empty DSMC result file: {file_path}"
        )
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
    if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
        raise ValueError(f"Invalid or empty DEM result file: {file_path}")
    data = np.loadtxt(file_path)
    data = np.atleast_2d(data)
    if data.size == 0 or data.shape[1] < 4:
        raise ValueError(
            f"Invalid or empty DEM result file: {file_path}"
        )
    t = data[:, 0]
    tau = data[:, 1]
    T_trans = data[:, 2]
    T_rot = data[:, 3]
    return t, tau, T_trans, T_rot


def load_pressure_results(file_path):
    """Load pressure tensor output file.

    Columns: t  tau
             Pxx_k Pxy_k Pxz_k Pyy_k Pyz_k Pzz_k   (kinetic, upper triangle)
             Pxx_c Pxy_c Pxz_c Pyy_c Pyz_c Pzz_c   (collisional, upper triangle)

    Returns dict with keys:
        t, tau      : 1-D arrays
        pij_k       : (N,3,3) kinetic pressure tensor
        pij_c       : (N,3,3) collisional pressure tensor
        pij         : (N,3,3) total pressure tensor (kinetic + collisional)
    """
    if (not os.path.exists(file_path)) or os.path.getsize(file_path) == 0:
        raise ValueError(f"Invalid or empty pressure result file: {file_path}")
    data = np.loadtxt(file_path)
    data = np.atleast_2d(data)
    if data.shape[1] < 14:
        raise ValueError(
            f"Pressure file has {data.shape[1]} columns, expected 14: {file_path}"
        )

    t   = data[:, 0]
    tau = data[:, 1]

    def _to_tensor(cols):
        n = len(t)
        P = np.zeros((n, 3, 3))
        P[:, 0, 0] = cols[:, 0]                   # Pxx
        P[:, 0, 1] = P[:, 1, 0] = cols[:, 1]      # Pxy
        P[:, 0, 2] = P[:, 2, 0] = cols[:, 2]      # Pxz
        P[:, 1, 1] = cols[:, 3]                    # Pyy
        P[:, 1, 2] = P[:, 2, 1] = cols[:, 4]      # Pyz
        P[:, 2, 2] = cols[:, 5]                    # Pzz
        return P

    pij_k = _to_tensor(data[:, 2:8])
    pij_c = _to_tensor(data[:, 8:14])
    return dict(t=t, tau=tau, pij_k=pij_k, pij_c=pij_c, pij=pij_k + pij_c)


def compute_normalized_temperature(T, T0=None):
    """Normalize temperature by initial value."""
    if T0 is None:
        T0 = T[0]
    return T / T0


def compute_temperature_ratio(T_trans, T_rot):
    """Compute theta = T_trans / T_rot."""
    return T_trans / T_rot
