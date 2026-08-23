#!/usr/bin/env python3
"""Validate the eps azimuthal scattering model against CTC data.

Tests the vonMises eps model p(eps|mu,alpha,AR) using purely MC evaluation
of the scattering kernel — no full DSMC needed.  ghat_post direction is
independent of the energy model (GMM, dissipation, Z_r) by construction,
so this test is a complete and direct validation of the direction field.

Tests
-----
A  Per-case kappa scan   — empirical R_bar / kappa vs polynomial fit, all 66 cases
B  Per-mu-bin detail     — kappa(mu) trend for selected --alpha / --ar
C  ghat_post direction   — CTC events: ctc_direct vs eps_zero / eps_model / chicond variants
D  Anisotropy kernel     — synthetic anisotropic g_pre: relax_gxy per chi+eps combination

Usage
-----
    python diagnose_eps_model.py [--alpha 1.0] [--ar 2.0] [--n-samples 50000]
                                  [--ctc-source ../Coll_Models/results]
                                  [--eps-coeffs models/eps_azimuth_coeffs.npz]
                                  [--mu-chi-coeffs models/mu_chi_beta_coeffs.npz]
                                  [--covariance 1.70,0.70,0.80,-0.55]
                                  [--output-dir runs/eps_diagnostics]
                                  [--n-mu-bins 10]
"""
import argparse
import csv
import json
import os
import re
import glob
from pathlib import Path

import numpy as np

from src.preprocessing.fit_eps_model import (
    _load_case_eps, _von_mises_kappa_from_R,
    eval_kappa, sample_eps_given_mu, load_eps_model,
)
from src.preprocessing.mu_chi_model import load_mu_chi_model, sample_chi_given_mu
from src.simulation.mu_joint import mu_plane_post_relative_with_eps


# ---------------------------------------------------------------------------
# Hard-sphere chi formula
# ---------------------------------------------------------------------------

def _chi_hs(mu, alpha):
    """Hard-sphere scattering angle chi given mu=|eij.ghat| and alpha."""
    mu = float(np.clip(mu, 0.0, 1.0))
    denom = np.sqrt(max(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cos_chi = (1.0 - (1.0 + alpha) * mu * mu) / denom
    return float(np.arccos(np.clip(cos_chi, -1.0, 1.0)))


def _chi_hs_vec(mu_arr, alpha):
    """Vectorized hard-sphere chi."""
    mu_arr = np.clip(mu_arr, 0.0, 1.0)
    denom = np.sqrt(np.maximum(1.0 - (1.0 - alpha * alpha) * mu_arr ** 2, 1.0e-30))
    cos_chi = (1.0 - (1.0 + alpha) * mu_arr ** 2) / denom
    return np.arccos(np.clip(cos_chi, -1.0, 1.0))


# ---------------------------------------------------------------------------
# NTC acceptance sampler  (inline from diagnose_collision_kernel.py)
# ---------------------------------------------------------------------------

def _sample_accepted_eij(g, rng, chunk_size=50000):
    """Sample eij ~ |eij.ghat| (NTC acceptance). Returns (eij, mu_abs)."""
    n_samples = g.shape[0]
    eij = np.empty_like(g)
    mu = np.empty(n_samples, dtype=float)
    pending = np.arange(n_samples)
    while pending.size:
        n_try = min(chunk_size, pending.size)
        idx = pending[:n_try]
        cands = rng.normal(size=(n_try, 3))
        cands /= np.linalg.norm(cands, axis=1)[:, None]
        g_chunk = g[idx]
        gmag = np.linalg.norm(g_chunk, axis=1)
        ghat = g_chunk / gmag[:, None]
        signed_mu = np.einsum("ij,ij->i", cands, ghat)
        abs_mu = np.abs(signed_mu)
        accept = rng.random(n_try) <= abs_mu
        acc_idx = idx[accept]; rej_idx = idx[~accept]
        if acc_idx.size:
            vals = cands[accept].copy()
            flip = signed_mu[accept] < 0.0
            vals[flip] *= -1.0
            eij[acc_idx] = vals
            mu[acc_idx] = abs_mu[accept]
        pending = np.concatenate([pending[n_try:], rej_idx])
    return eij, mu


# ---------------------------------------------------------------------------
# k_hat_frac metric  (inline from diagnose_collision_kernel.py)
# ---------------------------------------------------------------------------

def _k_hat_frac(g_post, g_pre, eij):
    """mean |g_post . k_hat| / |g_post|; 0=in-plane, ~0.48=random."""
    ghat = g_pre / np.maximum(np.linalg.norm(g_pre, axis=1, keepdims=True), 1e-30)
    mu_vec = np.einsum("ij,ij->i", eij, ghat)[:, None]
    n_perp = eij - mu_vec * ghat
    n_perp_norm = np.linalg.norm(n_perp, axis=1, keepdims=True)
    valid = n_perp_norm[:, 0] > 1e-10
    if not np.any(valid):
        return np.nan
    n_perp_hat = np.where(valid[:, None], n_perp / np.maximum(n_perp_norm, 1e-30), 0.0)
    k_hat = np.cross(ghat, n_perp_hat)
    gpost_norm = np.linalg.norm(g_post, axis=1)
    k_dot = np.abs(np.einsum("ij,ij->i", g_post, k_hat))
    fracs = k_dot[valid] / np.maximum(gpost_norm[valid], 1e-30)
    return float(np.mean(fracs))


def _cov_components(g):
    return {
        "gxx": float(np.mean(g[:, 0] ** 2)),
        "gyy": float(np.mean(g[:, 1] ** 2)),
        "gzz": float(np.mean(g[:, 2] ** 2)),
        "gxy": float(np.mean(g[:, 0] * g[:, 1])),
        "gyz": float(np.mean(g[:, 1] * g[:, 2])),
    }


# ---------------------------------------------------------------------------
# CTC case discovery
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"alpha_([0-9.]+)_r1\.00_AR([0-9.]+)")


def _iter_ctc_cases(source_root):
    """Yield (alpha, AR, chi_path) for all valid CTC cases."""
    for case_dir in sorted(glob.glob(os.path.join(source_root, "alpha_*_r1.00_AR*"))):
        m = _ALPHA_RE.search(case_dir)
        if not m:
            continue
        try:
            alpha = float(m.group(1))
            AR = float(m.group(2))
        except ValueError:
            continue
        chi_path = os.path.join(case_dir, "chi.txt")
        if os.path.exists(chi_path):
            yield alpha, AR, chi_path


# ---------------------------------------------------------------------------
# TEST A — per-case kappa summary
# ---------------------------------------------------------------------------

def test_A_kappa_scan(source_root, eps_model, n_mu_bins=20):
    """Print per-case empirical R_bar / kappa_emp vs polynomial kappa_fit."""
    rows = []
    for alpha, AR, chi_path in _iter_ctc_cases(source_root):
        mu_arr, eps_arr = _load_case_eps(chi_path)
        if mu_arr is None or len(mu_arr) < 50:
            continue
        R_bar = float(np.abs(np.mean(np.exp(1j * eps_arr))))
        kappa_emp = _von_mises_kappa_from_R(R_bar)
        mean_mu = float(np.mean(mu_arr))
        kappa_fit = np.nan
        if eps_model is not None:
            c_kappa, M, N, J, beta_exp = eps_model
            kappa_fit = eval_kappa(mean_mu, alpha, AR, c_kappa, M, N, J, beta_exp)
        rows.append({
            "alpha": alpha, "AR": AR, "n": len(mu_arr),
            "R_bar_emp": R_bar,
            "kappa_emp": kappa_emp,
            "kappa_fit": kappa_fit,
            "residual": abs(kappa_emp - kappa_fit) if np.isfinite(kappa_fit) else np.nan,
        })
    return rows


# ---------------------------------------------------------------------------
# TEST B — per-mu-bin kappa comparison
# ---------------------------------------------------------------------------

def test_B_mu_bins(source_root, alpha, AR, eps_model, n_mu_bins=10):
    """Per-mu-bin: empirical kappa_emp vs fitted kappa_fit."""
    pattern = os.path.join(source_root, f"alpha_{alpha:.2f}_r1.00_AR{AR:.1f}")
    candidates = glob.glob(pattern)
    if not candidates:
        pattern2 = os.path.join(source_root, f"alpha_{alpha}_r1.00_AR{AR}")
        candidates = glob.glob(pattern2)
    if not candidates:
        return []
    mu_arr, eps_arr = _load_case_eps(os.path.join(candidates[0], "chi.txt"))
    if mu_arr is None:
        return []

    edges = np.linspace(0.0, 1.0, n_mu_bins + 1)
    rows = []
    for b in range(n_mu_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (mu_arr >= lo) & (mu_arr < hi) if b < n_mu_bins - 1 else (mu_arr >= lo) & (mu_arr <= hi)
        if mask.sum() < 30:
            continue
        eps_bin = eps_arr[mask]
        R_bar = float(np.abs(np.mean(np.exp(1j * eps_bin))))
        kappa_emp = _von_mises_kappa_from_R(R_bar)
        mu_mid = float(np.mean(mu_arr[mask]))
        kappa_fit = np.nan
        if eps_model is not None:
            c_kappa, M, N, J, beta_exp = eps_model
            kappa_fit = eval_kappa(mu_mid, alpha, AR, c_kappa, M, N, J, beta_exp)
        rows.append({
            "mu_mid": mu_mid, "n": int(mask.sum()),
            "R_bar_emp": R_bar,
            "kappa_emp": kappa_emp,
            "kappa_fit": kappa_fit,
        })
    return rows


# ---------------------------------------------------------------------------
# TEST C — ghat_post direction validation using CTC events
# ---------------------------------------------------------------------------

def test_C_ghat_post(source_root, alpha, AR, eps_model, mu_chi_model, rng):
    """Compare ghat_post direction distributions: ctc_direct vs model variants.

    chi for model variants: chi_hs (hard-sphere) or chi_cond (conditional Beta).
    eps for model variants: 0, vonMises model, or uniform.
    ctc_direct: actual ghat_post from CTC data (flipped to DSMC frame).
    """
    pattern = os.path.join(source_root, f"alpha_{alpha:.2f}_r1.00_AR{AR:.1f}")
    candidates = glob.glob(pattern)
    if not candidates:
        pattern2 = os.path.join(source_root, f"alpha_{alpha}_r1.00_AR{AR}")
        candidates = glob.glob(pattern2)
    if not candidates:
        print(f"  [Test C] CTC case not found for alpha={alpha}, AR={AR}")
        return []

    chi_path = os.path.join(candidates[0], "chi.txt")
    try:
        data = np.loadtxt(chi_path)
    except Exception as e:
        print(f"  [Test C] Cannot load {chi_path}: {e}")
        return []
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 10:
        print(f"  [Test C] chi.txt has only {data.shape[1]} columns — need 10")
        return []

    mu_ctc = data[:, 3]
    eij_ctc = data[:, 4:7]
    ghat_post_ctc = data[:, 7:10]
    chi_ctc = data[:, 1]

    # Normalize
    eij_norms = np.linalg.norm(eij_ctc, axis=1, keepdims=True)
    post_norms = np.linalg.norm(ghat_post_ctc, axis=1, keepdims=True)
    valid = ((eij_norms > 0.5).ravel() & (post_norms > 0.5).ravel()
             & np.isfinite(mu_ctc) & (mu_ctc >= 0) & (mu_ctc <= 1)
             & (chi_ctc > 0.05))
    if valid.sum() < 100:
        print(f"  [Test C] Too few valid rows ({valid.sum()})")
        return []
    eij_ctc = (eij_ctc[valid] / eij_norms[valid])
    ghat_post_ctc = (ghat_post_ctc[valid] / post_norms[valid])
    mu_ctc = mu_ctc[valid]
    N = len(mu_ctc)

    # ghat_pre in CTC frame is (-1,0,0); in DSMC frame it's (+1,0,0)
    # ghat_post_CTC = -ghat_post_DSMC  →  ghat_post_DSMC = -ghat_post_CTC
    gp_ctc_dsmc = -ghat_post_ctc      # DSMC-convention direction (unit vectors)

    # Helper: compute g_post for a variant given chi array and eps array
    g_pre_dsmc = np.array([1.0, 0.0, 0.0])   # CTC standardised ghat_pre (DSMC frame)

    def _batch_scatter(mu_arr, chi_arr, eps_arr):
        g_post = np.empty((len(mu_arr), 3))
        for i in range(len(mu_arr)):
            g_post[i] = mu_plane_post_relative_with_eps(
                g_pre_dsmc, eij_ctc[i], chi_arr[i], 1.0, eps_arr[i]
            )
        return g_post

    rows = []

    # ctc_direct
    g_ctc = gp_ctc_dsmc
    rows.append({
        "variant": "ctc_direct",
        **_cov_components(g_ctc),
        "k_hat_frac": _k_hat_frac(g_ctc, np.tile(g_pre_dsmc, (N, 1)), eij_ctc),
    })

    # chi_hs + eps_zero
    chi_hs_arr = _chi_hs_vec(mu_ctc, alpha)
    eps_zero = np.zeros(N)
    g_hs_0 = _batch_scatter(mu_ctc, chi_hs_arr, eps_zero)
    rows.append({
        "variant": "eps_zero_chihs",
        **_cov_components(g_hs_0),
        "k_hat_frac": _k_hat_frac(g_hs_0, np.tile(g_pre_dsmc, (N, 1)), eij_ctc),
    })

    # chi_hs + eps_model
    if eps_model is not None:
        c_kappa, M_e, N_e, J_e, beta_exp_e = eps_model
        eps_mod = np.array([
            sample_eps_given_mu(mu_ctc[i], alpha, AR, c_kappa, M_e, N_e, J_e, beta_exp_e, rng=rng)
            for i in range(N)
        ])
        g_hs_m = _batch_scatter(mu_ctc, chi_hs_arr, eps_mod)
        rows.append({
            "variant": "eps_model_chihs",
            **_cov_components(g_hs_m),
            "k_hat_frac": _k_hat_frac(g_hs_m, np.tile(g_pre_dsmc, (N, 1)), eij_ctc),
        })
    else:
        print("  [Test C] eps model not available — skipping eps_model_chihs variant")

    # chi_cond + eps_zero
    if mu_chi_model is not None:
        c_a, c_b, Mc, Nc, Jc, be_c = mu_chi_model
        chi_cond_arr = np.array([
            sample_chi_given_mu(mu_ctc[i], alpha, AR, c_a, c_b, Mc, Nc, Jc, be_c, rng=rng)
            for i in range(N)
        ])
        g_cond_0 = _batch_scatter(mu_ctc, chi_cond_arr, eps_zero)
        rows.append({
            "variant": "eps_zero_chicond",
            **_cov_components(g_cond_0),
            "k_hat_frac": _k_hat_frac(g_cond_0, np.tile(g_pre_dsmc, (N, 1)), eij_ctc),
        })

        # chi_cond + eps_model
        if eps_model is not None:
            c_kappa, M_e, N_e, J_e, beta_exp_e = eps_model
            eps_mod2 = np.array([
                sample_eps_given_mu(mu_ctc[i], alpha, AR, c_kappa, M_e, N_e, J_e, beta_exp_e, rng=rng)
                for i in range(N)
            ])
            g_cond_m = _batch_scatter(mu_ctc, chi_cond_arr, eps_mod2)
            rows.append({
                "variant": "eps_model_chicond",
                **_cov_components(g_cond_m),
                "k_hat_frac": _k_hat_frac(g_cond_m, np.tile(g_pre_dsmc, (N, 1)), eij_ctc),
            })
    else:
        print("  [Test C] mu-chi model not available — skipping chi_cond variants")

    # chi_hs + eps_isotropic
    eps_iso = rng.uniform(-np.pi, np.pi, N)
    g_hs_iso = _batch_scatter(mu_ctc, chi_hs_arr, eps_iso)
    rows.append({
        "variant": "eps_isotropic_chihs",
        **_cov_components(g_hs_iso),
        "k_hat_frac": _k_hat_frac(g_hs_iso, np.tile(g_pre_dsmc, (N, 1)), eij_ctc),
    })

    return rows


# ---------------------------------------------------------------------------
# TEST D — anisotropy kernel test
# ---------------------------------------------------------------------------

def test_D_anisotropy(alpha, AR, n_samples, covariance, eps_model,
                      mu_chi_model, rng):
    """Anisotropic g_pre → compare relax_gxy per chi+eps variant."""
    g_pre = rng.multivariate_normal(np.zeros(3), covariance, size=n_samples)
    eij, mu_arr = _sample_accepted_eij(g_pre, rng)
    pre = _cov_components(g_pre)
    pre_gxy = pre["gxy"]

    variants = {}
    chi_hs_arr = _chi_hs_vec(mu_arr, alpha)

    if mu_chi_model is not None:
        c_a, c_b, Mc, Nc, Jc, be_c = mu_chi_model
        chi_cond_arr = np.array([
            sample_chi_given_mu(mu_arr[i], alpha, AR, c_a, c_b, Mc, Nc, Jc, be_c, rng=rng)
            for i in range(n_samples)
        ])
    else:
        chi_cond_arr = None

    if eps_model is not None:
        c_kappa, M_e, N_e, J_e, beta_exp_e = eps_model
        eps_mod_arr = np.array([
            sample_eps_given_mu(mu_arr[i], alpha, AR, c_kappa, M_e, N_e, J_e, beta_exp_e, rng=rng)
            for i in range(n_samples)
        ])
    else:
        eps_mod_arr = None

    def _scatter_all(chi_arr, eps_arr):
        g_post = np.empty_like(g_pre)
        gmags = np.linalg.norm(g_pre, axis=1)
        for i in range(n_samples):
            g_post[i] = mu_plane_post_relative_with_eps(
                g_pre[i], eij[i], chi_arr[i], gmags[i], eps_arr[i]
            )
        return g_post

    eps_zero = np.zeros(n_samples)
    eps_iso  = rng.uniform(-np.pi, np.pi, n_samples)

    variants["eps_zero_chihs"]     = (chi_hs_arr,   eps_zero)
    if eps_mod_arr is not None:
        variants["eps_model_chihs"] = (chi_hs_arr,   eps_mod_arr)
    variants["eps_isotropic_chihs"] = (chi_hs_arr,   eps_iso)
    if chi_cond_arr is not None:
        variants["eps_zero_chicond"] = (chi_cond_arr, eps_zero)
        if eps_mod_arr is not None:
            variants["eps_model_chicond"] = (chi_cond_arr, eps_mod_arr)

    rows = []
    for name, (chi_arr, eps_arr) in variants.items():
        g_post = _scatter_all(chi_arr, eps_arr)
        post = _cov_components(g_post)
        relax_gxy = post["gxy"] / pre_gxy if abs(pre_gxy) > 1e-14 else np.nan
        rows.append({
            "variant": name,
            "pre_gxy": pre_gxy,
            "post_gxy": post["gxy"],
            "relax_gxy": relax_gxy,
            "post_gxx": post["gxx"],
            "post_gyy": post["gyy"],
            "post_gzz": post["gzz"],
            "k_hat_frac": _k_hat_frac(g_post, g_pre, eij),
        })
    return rows


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_table_A(rows):
    if not rows:
        return
    has_fit = any(np.isfinite(r.get("kappa_fit", np.nan)) for r in rows)
    hdr = f"{'alpha':>6} {'AR':>5} {'n':>7} {'R_bar':>7} {'kap_emp':>8}"
    if has_fit:
        hdr += f" {'kap_fit':>8} {'resid':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['alpha']:6.2f} {r['AR']:5.1f} {r['n']:7d} "
                f"{r['R_bar_emp']:7.4f} {r['kappa_emp']:8.2f}")
        if has_fit:
            kf = r['kappa_fit']
            rd = r['residual']
            kf_s = f"{kf:8.2f}" if np.isfinite(kf) else "     n/a"
            rd_s = f"{rd:7.2f}" if np.isfinite(rd) else "    n/a"
            line += f" {kf_s} {rd_s}"
        print(line)


def _print_table_B(rows):
    if not rows:
        return
    has_fit = any(np.isfinite(r.get("kappa_fit", np.nan)) for r in rows)
    hdr = f"{'mu_mid':>7} {'n':>6} {'R_bar':>7} {'kap_emp':>8}"
    if has_fit:
        hdr += f" {'kap_fit':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = (f"{r['mu_mid']:7.3f} {r['n']:6d} "
                f"{r['R_bar_emp']:7.4f} {r['kappa_emp']:8.2f}")
        if has_fit:
            kf = r['kappa_fit']
            line += f" {kf:8.2f}" if np.isfinite(kf) else "      n/a"
        print(line)


def _print_table_CD(rows, extra_cols=None):
    if not rows:
        return
    cols = ["variant", "gxx", "gyy", "gzz", "gyz", "k_hat_frac"]
    if extra_cols:
        cols += extra_cols
    w_var = max(len(r.get("variant", "")) for r in rows) + 2
    hdr = f"{'variant':{w_var}s}"
    for c in cols[1:]:
        hdr += f" {c:>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r.get('variant', ''):>{w_var}s}"
        for c in cols[1:]:
            v = r.get(c, np.nan)
            line += f" {v:10.5g}" if isinstance(v, float) and np.isfinite(v) else f" {'n/a':>10}"
        print(line)


def _print_table_D(rows):
    if not rows:
        return
    cols = ["variant", "pre_gxy", "post_gxy", "relax_gxy", "k_hat_frac"]
    w_var = max(len(r.get("variant", "")) for r in rows) + 2
    hdr = f"{'variant':{w_var}s}"
    for c in cols[1:]:
        hdr += f" {c:>11}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r.get('variant', ''):>{w_var}s}"
        for c in cols[1:]:
            v = r.get(c, np.nan)
            line += f" {v:11.5g}" if isinstance(v, float) and np.isfinite(v) else f" {'n/a':>11}"
        print(line)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_covariance(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) == 4:
        xx, yy, zz, xy = vals
        cov = np.array([[xx, xy, 0.0], [xy, yy, 0.0], [0.0, 0.0, zz]])
    elif len(vals) == 6:
        xx, yy, zz, xy, xz, yz = vals
        cov = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    else:
        raise ValueError("--covariance must have 4 or 6 comma-separated values")
    eig = np.linalg.eigvalsh(cov)
    if np.min(eig) <= 0.0:
        raise ValueError(f"Covariance not positive definite: eigenvalues={eig}")
    return cov


def parse_args():
    p = argparse.ArgumentParser(description="Validate eps azimuthal scattering model")
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--ar",           type=float, default=2.0)
    p.add_argument("--n-samples",    type=int,   default=50000)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--ctc-source",   default="../Coll_Models/results")
    p.add_argument("--eps-coeffs",   default="models/eps_azimuth_coeffs.npz")
    p.add_argument("--mu-chi-coeffs",default="models/mu_chi_beta_coeffs.npz")
    p.add_argument("--covariance",   default="1.70,0.70,0.80,-0.55")
    p.add_argument("--output-dir",   default="runs/eps_diagnostics")
    p.add_argument("--n-mu-bins",    type=int,   default=10)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_results(out_dir, rows_A, rows_B, rows_C, rows_D, metadata):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    def _safe(v):
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v

    def _clean(rows):
        return [{k: _safe(v) for k, v in r.items()} for r in rows]

    json_path = os.path.join(out_dir, "eps_diagnostics.json")
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata,
                   "test_A": _clean(rows_A), "test_B": _clean(rows_B),
                   "test_C": _clean(rows_C), "test_D": _clean(rows_D)}, f, indent=2)

    for tag, rows in [("A_kappa_scan", rows_A), ("B_mu_bins", rows_B),
                      ("C_ghat_post", rows_C), ("D_anisotropy", rows_D)]:
        if not rows:
            continue
        csv_path = os.path.join(out_dir, f"eps_{tag}.csv")
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(_clean(rows))

    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    covariance = _parse_covariance(args.covariance)

    # Load models (graceful: None if absent)
    eps_model = load_eps_model(args.eps_coeffs)
    if eps_model is not None:
        print(f"  Loaded eps model: {args.eps_coeffs}")
    else:
        print(f"  eps model not found at {args.eps_coeffs} — fit-quality columns will be empty")

    mu_chi_model = None
    if os.path.exists(args.mu_chi_coeffs):
        mu_chi_model = load_mu_chi_model(args.mu_chi_coeffs)
        print(f"  Loaded chi model: {args.mu_chi_coeffs}")
    else:
        print(f"  chi model not found at {args.mu_chi_coeffs} — chi_cond variants skipped")

    print(f"\n=== Test A: per-case kappa summary (all CTC cases) ===")
    rows_A = test_A_kappa_scan(args.ctc_source, eps_model, n_mu_bins=args.n_mu_bins)
    if rows_A:
        _print_table_A(rows_A)
    else:
        print("  No CTC data found — check --ctc-source")

    print(f"\n=== Test B: per-mu-bin kappa  (alpha={args.alpha:.2f}, AR={args.ar:.1f}) ===")
    rows_B = test_B_mu_bins(args.ctc_source, args.alpha, args.ar, eps_model,
                            n_mu_bins=args.n_mu_bins)
    if rows_B:
        _print_table_B(rows_B)
    else:
        print("  Case not found or no valid data")

    print(f"\n=== Test C: ghat_post direction  (alpha={args.alpha:.2f}, AR={args.ar:.1f}) ===")
    print("  Using CTC events directly; chi_hs or chi_cond; only eps varies")
    rows_C = test_C_ghat_post(args.ctc_source, args.alpha, args.ar,
                               eps_model, mu_chi_model, rng)
    if rows_C:
        _print_table_CD(rows_C)
    else:
        print("  No results (case not found or too few data)")

    print(f"\n=== Test D: anisotropy kernel  (N={args.n_samples}, alpha={args.alpha:.2f}, AR={args.ar:.1f}) ===")
    print(f"  Pre-collision covariance: {args.covariance}")
    rows_D = test_D_anisotropy(args.alpha, args.ar, args.n_samples, covariance,
                                eps_model, mu_chi_model, rng)
    if rows_D:
        _print_table_D(rows_D)
    else:
        print("  No results")

    metadata = {
        "alpha": args.alpha, "ar": args.ar,
        "n_samples": args.n_samples, "seed": args.seed,
        "ctc_source": args.ctc_source,
        "eps_coeffs": args.eps_coeffs,
        "covariance": covariance.tolist(),
    }
    json_path = _save_results(args.output_dir, rows_A, rows_B, rows_C, rows_D, metadata)
    print(f"\nSaved diagnostics to {json_path}")
    print()
    print("Key: k_hat_frac≈0 = in-plane  |  k_hat_frac≈0.48 = random azimuth")
    print("     relax_gxy≈0 = anisotropy preserved  |  relax_gxy≈1 = randomised")


if __name__ == "__main__":
    main()
