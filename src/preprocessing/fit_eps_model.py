"""
Azimuthal scattering model: p(eps | mu, alpha, AR) as a von Mises distribution.

    eps ~ vonMises(0, kappa(mu, alpha, AR))

eps is the azimuthal angle of ghat_post around ghat_pre, measured relative to
the in-plane (-n_perp_hat) direction.  eps=0 is the hard-sphere / in-plane
result; eps uniform (kappa=0) means azimuthally isotropic scattering.

kappa(mu, alpha, AR) is fitted as a polynomial in (AR, phi(alpha), mu) in
log-space (log(1+kappa), to allow kappa=0), matching the chi-model framework.

Data source: 10-column chi.txt (new format with eij and ghat_post):
  col 0 (index 0): b_impact
  col 1 (index 1): chi (scattering angle, radians)
  col 2 (index 2): psi
  col 3 (index 3): mu_in = |r12_fc_hat · ghat_pre|
  col 4 (index 4): eij_x  (contact normal, first-contact only)
  col 5 (index 5): eij_y
  col 6 (index 6): eij_z
  col 7 (index 7): ghat_post_x
  col 8 (index 8): ghat_post_y
  col 9 (index 9): ghat_post_z

Usage:
    python run_eps_fit.py --config config/default.yaml
"""

import os
import re
import glob

import numpy as np
from scipy.special import iv as _bessel_i   # modified Bessel functions


# ---------------------------------------------------------------------------
# Internal helpers (matching mu_chi_model framework)
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
    """Polynomial basis: AR^m * phi^n * mu^j."""
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


def _von_mises_kappa_from_R(R_bar):
    """MLE estimate of von Mises concentration kappa from mean resultant length R_bar.

    Uses the approximation from Mardia & Jupp (2000):
      R_bar < 0.53:  kappa ≈ 2*R + R^3 + 5/6*R^5
      0.53 <= R < 0.85: kappa ≈ -0.4 + 1.39*R + 0.43/(1-R)
      R >= 0.85:       kappa ≈ 1/(R^3 - 4*R^2 + 3*R)
    """
    R_bar = float(np.clip(R_bar, 0.0, 1.0 - 1.0e-9))
    if R_bar < 0.53:
        return 2.0 * R_bar + R_bar**3 + (5.0 / 6.0) * R_bar**5
    elif R_bar < 0.85:
        return -0.4 + 1.39 * R_bar + 0.43 / (1.0 - R_bar)
    else:
        denom = R_bar**3 - 4.0 * R_bar**2 + 3.0 * R_bar
        return 1.0 / max(denom, 1.0e-10)


def _compute_eps_angles(eij, ghat_post):
    """Compute eij-relative azimuthal eps for each record.

    eps is the azimuthal angle of ghat_post around ghat_pre (= CTC convention
    (-1,0,0)), measured relative to the (ghat_pre, eij) plane.  eps=0 is the
    in-plane direction (same as the hard-sphere / mu_plane_post_relative limit).

    The frame matches mu_plane_post_relative_with_eps in mu_joint.py:
      n_perp_hat = (eij - mu*ghat_pre) / |...|   [e1, eps=0 reference]
      n2         = ghat_pre x n_perp_hat           [e2, out-of-plane]

    Returns eps array in (-pi, pi].
    """
    ghat_pre_vec = np.array([-1.0, 0.0, 0.0])   # CTC standardises VREL0
    N = eij.shape[0]
    eps_out = np.zeros(N)

    for i in range(N):
        eij_i = eij[i] / np.linalg.norm(eij[i])
        gq    = ghat_post[i]

        # n_perp_hat: component of eij perpendicular to ghat_pre
        mu_i   = float(np.clip(np.dot(eij_i, ghat_pre_vec), -1.0, 1.0))
        n_perp = eij_i - mu_i * ghat_pre_vec
        n_perp_norm = np.linalg.norm(n_perp)
        if n_perp_norm < 1.0e-10:
            eps_out[i] = 0.0
            continue
        n_perp_hat = n_perp / n_perp_norm          # e1 — eps=0 direction
        n2 = np.cross(ghat_pre_vec, n_perp_hat)    # e2 — out-of-plane

        # Project ghat_post onto the (n_perp_hat, n2) plane
        gq_perp = gq - np.dot(gq, ghat_pre_vec) * ghat_pre_vec
        norm_perp = np.linalg.norm(gq_perp)
        if norm_perp < 1.0e-10:
            eps_out[i] = 0.0
            continue
        gq_perp_hat = gq_perp / norm_perp
        eps_out[i] = np.arctan2(np.dot(gq_perp_hat, n2),
                                np.dot(gq_perp_hat, n_perp_hat))

    return eps_out


def _load_case_eps(chi_path, min_chi_rad=0.05):
    """Load a 10-column chi.txt and return (mu, eps_rad) arrays.

    Columns 4–6 must be eij (contact normal), columns 7–9 must be ghat_post.
    Skips records where chi < min_chi_rad (near-forward scatter: eij nearly
    parallel to ghat_pre, so the in-plane frame is degenerate).
    """
    try:
        data = np.loadtxt(chi_path)
    except Exception:
        return None, None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 10:
        return None, None   # old 4-column format

    mu        = data[:, 3]
    eij       = data[:, 4:7]
    ghat_post = data[:, 7:10]
    chi_vals  = data[:, 1]

    eij_norms  = np.linalg.norm(eij,       axis=1, keepdims=True)
    post_norms = np.linalg.norm(ghat_post, axis=1, keepdims=True)
    valid = ((eij_norms  > 0.5).ravel()
             & (post_norms > 0.5).ravel()
             & (chi_vals   > min_chi_rad)
             & np.isfinite(mu)
             & (mu >= 0.0) & (mu <= 1.0))
    if valid.sum() < 10:
        return None, None

    eij       = eij[valid]       / eij_norms[valid]
    ghat_post = ghat_post[valid] / post_norms[valid]
    mu        = mu[valid]

    eps = _compute_eps_angles(eij, ghat_post)
    return mu, eps


# ---------------------------------------------------------------------------
# Main fitting routine
# ---------------------------------------------------------------------------

def fit_eps_model(source_root, n_mu_bins=20, M=2, N=2, J=3,
                  beta_exp=0.5, min_count=30):
    """Fit von Mises kappa(mu, alpha, AR) from 10-column CTC chi.txt files.

    Returns (c_kappa, M, N, J, beta_exp, diagnostics) where c_kappa has shape
    (M+1, N+1, J+1) and diagnostics is a list of dicts.
    """
    pattern = os.path.join(source_root, "alpha_*_r1.00_AR*")
    _alpha_re = re.compile(r"alpha_([0-9.]+)_r1\.00_AR([0-9.]+)")

    records = []   # list of (alpha, AR, mu, eps)
    diagnostics = []

    for case_dir in sorted(glob.glob(pattern)):
        m = _alpha_re.search(case_dir)
        if not m:
            continue
        alpha = float(m.group(1))
        AR    = float(m.group(2))
        chi_path = os.path.join(case_dir, "chi.txt")
        if not os.path.exists(chi_path):
            continue

        mu_arr, eps_arr = _load_case_eps(chi_path)
        if mu_arr is None:
            print(f"  Skipping {case_dir}: old chi.txt format or too few records")
            continue

        print(f"  alpha={alpha:.2f} AR={AR:.1f}: {len(mu_arr)} records")
        records.append((alpha, AR, mu_arr, eps_arr))

        # Per-case isotropy diagnostic
        R_bar_all = float(np.abs(np.mean(np.exp(1j * eps_arr))))
        kappa_all = _von_mises_kappa_from_R(R_bar_all)
        diagnostics.append({
            "alpha": alpha, "AR": AR,
            "n": len(mu_arr), "R_bar": R_bar_all, "kappa": kappa_all,
        })

    if not records:
        raise FileNotFoundError(
            f"No 10-column chi.txt found in {source_root}. "
            "Re-run CTC with updated measure_dem.f90 first."
        )

    # Build per-bin dataset for polynomial fit
    mu_pts, phi_pts, ar_pts, lkappa_pts = [], [], [], []

    for alpha, AR, mu_arr, eps_arr in records:
        phi_val = _phi(alpha, beta_exp)
        edges   = _equal_count_edges(mu_arr, n_mu_bins)

        for b in range(n_mu_bins):
            lo = edges[b]; hi = edges[b + 1]
            if b < n_mu_bins - 1:
                mask = (mu_arr >= lo) & (mu_arr < hi)
            else:
                mask = (mu_arr >= lo) & (mu_arr <= hi)
            if mask.sum() < min_count:
                continue

            eps_bin = eps_arr[mask]
            R_bar   = float(np.abs(np.mean(np.exp(1j * eps_bin))))
            kappa   = _von_mises_kappa_from_R(R_bar)
            mu_mid  = float(np.mean(mu_arr[mask]))

            mu_pts.append(mu_mid)
            phi_pts.append(phi_val)
            ar_pts.append(float(AR))
            lkappa_pts.append(np.log1p(kappa))   # log(1+kappa): stable for kappa near 0

    mu_pts    = np.array(mu_pts)
    phi_pts   = np.array(phi_pts)
    ar_pts    = np.array(ar_pts)
    lkappa_pts = np.array(lkappa_pts)

    # Polynomial regression in log(1+kappa)
    X = _build_design_matrix(mu_pts, phi_pts, ar_pts, M, N, J)
    c_flat, _, _, _ = np.linalg.lstsq(X, lkappa_pts, rcond=None)
    c_kappa = c_flat.reshape((M + 1, N + 1, J + 1))

    return c_kappa, M, N, J, float(beta_exp), diagnostics


def save_eps_model(output_path, c_kappa, M, N, J, beta_exp):
    """Save von Mises azimuthal model coefficients to .npz."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(
        output_path,
        c_kappa=c_kappa,
        M=np.array([int(M)]),
        N=np.array([int(N)]),
        J=np.array([int(J)]),
        beta_exp=np.array([float(beta_exp)]),
    )


def load_eps_model(path):
    """Load eps model. Returns (c_kappa, M, N, J, beta_exp) or None."""
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return (
        np.array(d["c_kappa"]),
        int(d["M"][0]), int(d["N"][0]), int(d["J"][0]),
        float(d["beta_exp"][0]),
    )


def eval_kappa(mu, alpha, AR, c_kappa, M, N, J, beta_exp):
    """Evaluate von Mises kappa(mu, alpha, AR) from polynomial coefficients."""
    phi_val = _phi(alpha, beta_exp)
    lkappa = 0.0
    col = 0
    for m in range(M + 1):
        for n in range(N + 1):
            for j in range(J + 1):
                lkappa += c_kappa.ravel()[col] * (AR ** m) * (phi_val ** n) * (mu ** j)
                col += 1
    return max(np.expm1(lkappa), 0.0)   # kappa = exp(log(1+kappa)) - 1


def sample_eps_given_mu(mu, alpha, AR, c_kappa, M, N, J, beta_exp, rng=np.random):
    """Sample azimuthal angle eps ~ vonMises(0, kappa(mu, alpha, AR)).

    Returns eps in (-pi, pi].  If kappa < 0.1 (effectively isotropic),
    returns a uniform sample to avoid numerical issues.
    """
    kappa = eval_kappa(float(mu), float(alpha), float(AR), c_kappa, M, N, J, beta_exp)
    if kappa < 0.1:
        return rng.uniform(-np.pi, np.pi)
    # von Mises sampling via Best-Fisher algorithm
    a = 1.0 + np.sqrt(1.0 + 4.0 * kappa ** 2)
    b = (a - np.sqrt(2.0 * a)) / (2.0 * kappa)
    r = (1.0 + b ** 2) / (2.0 * b)
    while True:
        U1 = rng.uniform(0.0, 1.0)
        z = np.cos(np.pi * U1)
        f = (1.0 + r * z) / (r + z)
        c_val = kappa * (r - f)
        U2 = rng.uniform(0.0, 1.0)
        if c_val * (2.0 - c_val) > U2:
            U3 = rng.uniform(0.0, 1.0)
            return np.sign(U3 - 0.5) * np.arccos(f)
        if np.log(c_val / U2) + 1.0 - c_val >= 0.0:
            U3 = rng.uniform(0.0, 1.0)
            return np.sign(U3 - 0.5) * np.arccos(f)
