"""
Conditional chi model: p(chi | mu, alpha, AR) as a Beta distribution.

    chi/pi ~ Beta(a(mu, alpha, AR), b(mu, alpha, AR))

Parameters a and b are fitted as polynomials in (AR, phi(alpha), mu) in log
space, where phi(alpha) = 1 - alpha^beta_exp.  Log-space fitting guarantees
a, b > 0 everywhere.

Data source: Coll_Models/results/alpha_*_r1.00_AR*/chi.txt
  col 1 (index 1): chi in radians
  col 3 (index 3): mu = |n_contact · ghat| (pre-collision cosine)
"""

import os
import re
import glob

import numpy as np
from scipy.stats import beta as _scipy_beta


_EPS_CHI = 1.0e-4   # guard: clip chi/pi away from boundary for stable MLE
_LOG_A_BOUNDS = (-3.0, 7.0)   # a in [exp(-3)≈0.05, exp(7)≈1097]
_LOG_B_BOUNDS = (-3.0, 7.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _phi(alpha, beta_exp):
    return 1.0 - float(alpha) ** float(beta_exp)


def _equal_count_edges(mu, n_bins):
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(mu, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)
    return edges


def _build_design_matrix(mu_arr, phi_arr, ar_arr, M, N, J):
    """Design matrix with basis AR^m * phi^n * mu^j (row-major flattening)."""
    R = len(mu_arr)
    ncols = (M + 1) * (N + 1) * (J + 1)
    X = np.zeros((R, ncols))
    for r in range(R):
        col = 0
        for m in range(M + 1):
            for n in range(N + 1):
                for j in range(J + 1):
                    X[r, col] = (ar_arr[r] ** m) * (phi_arr[r] ** n) * (mu_arr[r] ** j)
                    col += 1
    return X


def _fit_beta_bin(chi_norm_bin):
    """MLE fit of Beta(a, b) on [0,1] to chi_norm values in one mu-bin.

    Returns (log_a, log_b) or None on failure.
    """
    data = np.clip(chi_norm_bin, _EPS_CHI, 1.0 - _EPS_CHI)
    try:
        a_fit, b_fit, _, _ = _scipy_beta.fit(data, floc=0.0, fscale=1.0)
        if not (np.isfinite(a_fit) and np.isfinite(b_fit) and
                a_fit > 0.0 and b_fit > 0.0):
            return None
        return float(np.log(a_fit)), float(np.log(b_fit))
    except Exception:
        return None


def _eval_log_param(mu, phi, ar, c, M, N, J):
    """Evaluate a polynomial coefficient array at a single (mu, phi, AR) point."""
    val = 0.0
    for m in range(M + 1):
        for n in range(N + 1):
            for j in range(J + 1):
                val += c[m, n, j] * (ar ** m) * (phi ** n) * (mu ** j)
    return val


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_mu_chi_model(source_root, n_mu_bins=20, M=2, N=2, J=3, beta_exp=0.5,
                     min_bin_samples=30, verbose=True):
    """Fit the conditional Beta chi model from CTC data at r=1.00.

    Scans *source_root* for directories matching alpha_*_r1.00_AR*, loads
    chi.txt from each, bins by mu, fits Beta(a,b) per bin, then solves a
    global polynomial regression for log(a) and log(b).

    Parameters
    ----------
    source_root : str
        Directory containing alpha_*_r1.00_AR* sub-folders.
    n_mu_bins : int
        Number of equal-count mu bins per (alpha, AR) case.
    M, N, J : int
        Polynomial degrees in AR, phi(alpha), mu.
    beta_exp : float
        Exponent in phi(alpha) = 1 - alpha^beta_exp.
    min_bin_samples : int
        Skip a mu-bin if it has fewer than this many samples.

    Returns
    -------
    (c_a, c_b, M, N, J, beta_exp)
        c_a, c_b : ndarray, shape (M+1, N+1, J+1)
            Log-space polynomial coefficients for a and b.
    """
    pattern = os.path.join(source_root, "alpha_*_r1.00_AR*")
    cases = []
    for case_dir in sorted(glob.glob(pattern)):
        m = re.search(r"alpha_([0-9.]+)_r1\.00_AR([0-9.]+)$", case_dir)
        if m is None:
            continue
        chi_path = os.path.join(case_dir, "chi.txt")
        if os.path.exists(chi_path):
            cases.append((float(m.group(1)), float(m.group(2)), chi_path))

    if not cases:
        raise FileNotFoundError(
            f"No alpha_*_r1.00_AR* cases with chi.txt found under {source_root}"
        )

    if verbose:
        print(f"  fit_mu_chi_model: {len(cases)} cases from {source_root}")

    mu_centers, phi_vals, ar_vals, log_a_vals, log_b_vals = [], [], [], [], []

    for alpha_val, ar_val, chi_path in cases:
        raw = np.loadtxt(chi_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        if raw.shape[1] < 4:
            if verbose:
                print(f"    skip {chi_path}: only {raw.shape[1]} columns (mu missing)")
            continue

        mu = raw[:, 3]
        chi_norm = raw[:, 1] / np.pi

        valid = (
            np.isfinite(mu) & np.isfinite(chi_norm) &
            (mu >= 0.0) & (mu <= 1.0) &
            (chi_norm > 0.0) & (chi_norm < 1.0)
        )
        mu = mu[valid]
        chi_norm = chi_norm[valid]
        if mu.size < n_mu_bins * min_bin_samples:
            if verbose:
                print(f"    skip {chi_path}: too few valid rows ({mu.size})")
            continue

        edges = _equal_count_edges(mu, n_mu_bins)
        phi_v = _phi(alpha_val, beta_exp)
        n_ok = 0

        for i in range(n_mu_bins):
            lo, hi = edges[i], edges[i + 1]
            mu_c = 0.5 * (lo + hi)
            mask = (mu >= lo) & (mu <= hi) if i == n_mu_bins - 1 else \
                   (mu >= lo) & (mu < hi)
            if mask.sum() < min_bin_samples:
                continue

            result = _fit_beta_bin(chi_norm[mask])
            if result is None:
                continue

            log_a, log_b = result
            mu_centers.append(mu_c)
            phi_vals.append(phi_v)
            ar_vals.append(ar_val)
            log_a_vals.append(log_a)
            log_b_vals.append(log_b)
            n_ok += 1

        if verbose:
            print(f"    alpha={alpha_val:.2f} AR={ar_val:.1f}: {n_ok}/{n_mu_bins} bins fitted")

    if len(mu_centers) == 0:
        raise RuntimeError("No valid mu-bins were fitted. Check data path and format.")

    mu_arr = np.array(mu_centers)
    phi_arr = np.array(phi_vals)
    ar_arr = np.array(ar_vals)
    log_a_arr = np.array(log_a_vals)
    log_b_arr = np.array(log_b_vals)

    X = _build_design_matrix(mu_arr, phi_arr, ar_arr, M, N, J)
    shape = (M + 1, N + 1, J + 1)

    c_a_vec, res_a, _, _ = np.linalg.lstsq(X, log_a_arr, rcond=None)
    c_b_vec, res_b, _, _ = np.linalg.lstsq(X, log_b_arr, rcond=None)

    c_a = c_a_vec.reshape(shape, order='C')
    c_b = c_b_vec.reshape(shape, order='C')

    if verbose:
        log_a_pred = X @ c_a_vec
        log_b_pred = X @ c_b_vec
        rmse_a = float(np.sqrt(np.mean((log_a_pred - log_a_arr) ** 2)))
        rmse_b = float(np.sqrt(np.mean((log_b_pred - log_b_arr) ** 2)))
        print(f"  Polynomial fit RMSE (log-space): log_a={rmse_a:.4f}, log_b={rmse_b:.4f}")
        print(f"  Total fit points: {len(mu_centers)},  "
              f"coefficients per param: {shape[0]*shape[1]*shape[2]}")

    return c_a, c_b, M, N, J, beta_exp


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_mu_chi_model(path, c_a, c_b, M, N, J, beta_exp):
    """Save the conditional Beta chi model to a compressed npz file."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        path,
        c_a=c_a.astype(np.float64),
        c_b=c_b.astype(np.float64),
        M=np.array([int(M)], dtype=np.int32),
        N=np.array([int(N)], dtype=np.int32),
        J=np.array([int(J)], dtype=np.int32),
        beta_exp=np.array([float(beta_exp)]),
    )


def load_mu_chi_model(path):
    """Load the conditional Beta chi model from an npz file.

    Returns (c_a, c_b, M, N, J, beta_exp).
    """
    d = np.load(path)
    return (
        np.array(d['c_a'], dtype=np.float64),
        np.array(d['c_b'], dtype=np.float64),
        int(d['M'][0]),
        int(d['N'][0]),
        int(d['J'][0]),
        float(d['beta_exp'][0]),
    )


# ---------------------------------------------------------------------------
# Evaluation and sampling
# ---------------------------------------------------------------------------

def eval_beta_params(mu, alpha, AR, c_a, c_b, M, N, J, beta_exp=0.5):
    """Evaluate Beta distribution parameters (a, b) at (mu, alpha, AR).

    Both returned values are positive floats.
    """
    phi = _phi(alpha, beta_exp)
    mu = float(np.clip(mu, 0.0, 1.0))
    ar = float(AR)
    log_a = _eval_log_param(mu, phi, ar, c_a, M, N, J)
    log_b = _eval_log_param(mu, phi, ar, c_b, M, N, J)
    log_a = float(np.clip(log_a, _LOG_A_BOUNDS[0], _LOG_A_BOUNDS[1]))
    log_b = float(np.clip(log_b, _LOG_B_BOUNDS[0], _LOG_B_BOUNDS[1]))
    return float(np.exp(log_a)), float(np.exp(log_b))


def sample_chi_given_mu(mu, alpha, AR, c_a, c_b, M, N, J,
                         beta_exp=0.5, rng=np.random):
    """Sample chi (radians) from p(chi | mu, alpha, AR).

    Parameters
    ----------
    mu : float
        |eij · ghat| for the current NTC-accepted collision, in [0, 1].
    alpha : float
        Coefficient of restitution.
    AR : float
        Aspect ratio.
    c_a, c_b : ndarray, shape (M+1, N+1, J+1)
        Log-space polynomial coefficients (from load_mu_chi_model).

    Returns
    -------
    float
        chi in radians, in [0, pi].
    """
    a, b = eval_beta_params(mu, alpha, AR, c_a, c_b, M, N, J, beta_exp)
    chi_norm = rng.beta(a, b)
    return float(np.pi * chi_norm)
