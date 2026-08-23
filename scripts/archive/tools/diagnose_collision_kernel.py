#!/usr/bin/env python3
"""Collision-kernel diagnostics for scattering geometry and chi-mu conditioning.

Samples NTC-accepted binary collisions and compares how different kernel
variants transform an anisotropic pre-collision relative-velocity covariance.

Key kernels
-----------
sphere                  : exact hard-sphere (ground truth for AR=1)
sphcyl_ar1              : marginal chi + eps_from_eij (current code — known broken)
sphcyl_ar1_mu_beta_mpr  : conditional chi(mu) + mu_plane_post_relative (NEW correct model)
sphcyl_ar2_mu_beta_mpr  : same for AR=2

Key metrics
-----------
k_hat_frac   : mean |g_post · k_hat| / |g_post|  — 0 = in-plane, ~0.48 = random azimuth
relax_gxy    : post_gxy / pre_gxy  — the stress-tensor shear-relaxation diagnostic
chi_mu_binXX : mean chi in 5 mu-bins — shows whether chi is conditioned on mu
chi_hs_rmse  : RMSE of sampled chi vs hard-sphere chi_hs(mu) — residual from sphere

CTC empirical reference rows (no g_post needed) show the data the models were fitted on.
"""
import argparse
import csv
import json
import os
import re
import glob
from pathlib import Path

import numpy as np

from src.preprocessing.scattering_angle import p_chi_AR_alpha
from src.preprocessing.mu_chi_model import load_mu_chi_model, sample_chi_given_mu
from src.simulation.collision import eps_from_eij, update_velocities
from src.simulation.mu_joint import MuJointAR2Model, mu_plane_post_relative


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def chi_hs_formula(mu, alpha):
    """Hard-sphere scattering angle chi given mu = |eij . ghat| and alpha."""
    mu = np.asarray(mu, dtype=float)
    denom = np.sqrt(np.maximum(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cos_chi = (1.0 - (1.0 + alpha) * mu * mu) / denom
    return np.arccos(np.clip(cos_chi, -1.0, 1.0))


def compute_k_hat_frac(g_post, g_pre, eij):
    """Compute mean |g_post . k_hat| / |g_post| over all samples.

    k_hat = ghat x n_perp_hat is the out-of-plane direction.
    Value 0 means in-plane scattering; ~0.48 means fully random azimuth.
    """
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


def compute_chi_mu_bins(mu, chi_rad, n_bins=5):
    """Return mean chi in equal-width mu bins [0,0.2], [0.2,0.4], ..., [0.8,1.0]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = {}
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (mu >= lo) & (mu < hi) if i < n_bins - 1 else (mu >= lo) & (mu <= hi)
        key = f"chi_mu_{int(lo*10):02d}_{int(hi*10):02d}"
        result[key] = float(np.mean(chi_rad[mask])) if mask.sum() > 0 else np.nan
    return result


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def parse_covariance(text):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) == 4:
        xx, yy, zz, xy = vals
        cov = np.array([[xx, xy, 0.0], [xy, yy, 0.0], [0.0, 0.0, zz]], dtype=float)
    elif len(vals) == 6:
        xx, yy, zz, xy, xz, yz = vals
        cov = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)
    elif len(vals) == 9:
        cov = np.array(vals, dtype=float).reshape(3, 3)
    else:
        raise ValueError("--covariance must have 4, 6, or 9 comma-separated values")
    eig = np.linalg.eigvalsh(cov)
    if np.min(eig) <= 0.0:
        raise ValueError(f"--covariance must be positive definite; eigenvalues={eig}")
    return cov


def sample_relative_velocities(n_samples, covariance, rng):
    return rng.multivariate_normal(np.zeros(3), covariance, size=n_samples)


def sample_accepted_eij(g, rng, chunk_size=50000):
    """Sample eij ∝ |eij . ghat| (NTC acceptance kernel)."""
    n_samples = g.shape[0]
    eij = np.empty_like(g)
    mu = np.empty(n_samples, dtype=float)
    pending = np.arange(n_samples)

    while pending.size:
        n_try = min(chunk_size, pending.size)
        idx = pending[:n_try]
        candidates = rng.normal(size=(n_try, 3))
        candidates /= np.linalg.norm(candidates, axis=1)[:, None]

        g_chunk = g[idx]
        gmag = np.linalg.norm(g_chunk, axis=1)
        ghat = g_chunk / gmag[:, None]
        signed_mu = np.einsum("ij,ij->i", candidates, ghat)
        abs_mu = np.abs(signed_mu)
        accept = rng.random(n_try) <= abs_mu
        accepted_idx = idx[accept]
        rejected_idx = idx[~accept]

        if accepted_idx.size:
            vals = candidates[accept].copy()
            flip = signed_mu[accept] < 0.0
            vals[flip] *= -1.0
            eij[accepted_idx] = vals
            mu[accepted_idx] = abs_mu[accept]
        pending = np.concatenate([pending[n_try:], rejected_idx])

    return eij, mu


class ScatteringPDF:
    """Marginal p(chi | AR, alpha) polynomial model."""

    def __init__(self, coeff_path):
        coeffs = np.load(coeff_path)
        self.a_elastic = coeffs["a_elastic"]
        self.a_inelastic = coeffs["a_inelastic"]
        self.M = int(coeffs["M"])
        self.N = int(coeffs["N"])
        self.K = int(coeffs["K"])
        self.beta = float(coeffs["beta"])

    def pdf(self, chi, AR, alpha):
        return np.maximum(
            p_chi_AR_alpha(chi, AR, alpha, self.a_elastic, self.a_inelastic,
                           self.M, self.N, self.K, self.beta), 0.0
        )

    def sample(self, AR, alpha, size, rng):
        grid = np.linspace(0.0, 1.0, 2000)
        p_max = float(np.max(self.pdf(grid, AR, alpha)) * 1.05)
        if not np.isfinite(p_max) or p_max <= 0.0:
            raise ValueError(f"Invalid p_chi maximum for AR={AR:g}, alpha={alpha:g}")
        out = np.empty(size, dtype=float)
        filled = 0
        while filled < size:
            n_try = max(4096, 2 * (size - filled))
            chi = rng.uniform(0.0, 1.0, n_try)
            accept = rng.uniform(0.0, p_max, n_try) <= self.pdf(chi, AR, alpha)
            vals = chi[accept]
            n_take = min(vals.size, size - filled)
            if n_take:
                out[filled:filled + n_take] = vals[:n_take]
                filled += n_take
        return out  # chi/pi in [0,1]


# ---------------------------------------------------------------------------
# Kernel implementations
# ---------------------------------------------------------------------------

def _mpr_post_g(g_pre, eij, chi_rad):
    """Post-collision g via mu_plane_post_relative (preserves |g|, correct in-plane)."""
    n = g_pre.shape[0]
    gmag = np.linalg.norm(g_pre, axis=1)
    g_post = np.empty_like(g_pre)
    for i in range(n):
        g_post[i] = mu_plane_post_relative(g_pre[i], eij[i], chi_rad[i], gmag[i])
    return g_post


def _eps_eij_post_g(g_pre, eij, chi_rad, rng, fresh_azimuth=False):
    """Post-collision g via eps_from_eij + update_velocities (known out-of-plane)."""
    n = g_pre.shape[0]
    g_post = np.empty_like(g_pre)
    gmag = np.linalg.norm(g_pre, axis=1)
    if fresh_azimuth:
        eps_arr = rng.uniform(0.0, 2.0 * np.pi, n)
    else:
        eps_arr = np.array([eps_from_eij(g_pre[i], eij[i], rng=rng) for i in range(n)])
    for i in range(n):
        v1 = 0.5 * g_pre[i]
        v2 = -0.5 * g_pre[i]
        v1p, v2p = update_velocities(v1, v2, chi_rad[i], eps_arr[i], 0.5 * gmag[i])
        g_post[i] = np.ravel(v1p) - np.ravel(v2p)
    return g_post


# ---------------------------------------------------------------------------
# Summary / metrics
# ---------------------------------------------------------------------------

def covariance_components(g):
    return {
        "gxx": float(np.mean(g[:, 0] * g[:, 0])),
        "gyy": float(np.mean(g[:, 1] * g[:, 1])),
        "gzz": float(np.mean(g[:, 2] * g[:, 2])),
        "gxy": float(np.mean(g[:, 0] * g[:, 1])),
    }


def corrcoef(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    sx, sy = np.std(x[mask]), np.std(y[mask])
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def summarize_kernel(name, g_pre, g_post, eij, mu, chi_rad):
    pre = covariance_components(g_pre)
    post = covariance_components(g_post)
    pre_mag = np.linalg.norm(g_pre, axis=1)
    post_mag = np.linalg.norm(g_post, axis=1)
    cos_ggp = np.einsum("ij,ij->i", g_pre, g_post) / np.maximum(pre_mag * post_mag, 1e-30)
    cos_ggp = np.clip(cos_ggp, -1.0, 1.0)

    chi_hs = chi_hs_formula(mu, _ALPHA_GLOBAL)
    chi_hs_rmse = float(np.sqrt(np.mean((chi_rad - chi_hs) ** 2))) if chi_rad is not None else np.nan

    row = {
        "kernel": name,
        "n_samples": int(g_pre.shape[0]),
        "pre_gxx": pre["gxx"], "pre_gyy": pre["gyy"],
        "pre_gzz": pre["gzz"], "pre_gxy": pre["gxy"],
        "post_gxx": post["gxx"], "post_gyy": post["gyy"],
        "post_gzz": post["gzz"], "post_gxy": post["gxy"],
        "relax_gxx": post["gxx"] / pre["gxx"],
        "relax_gyy": post["gyy"] / pre["gyy"],
        "relax_gzz": post["gzz"] / pre["gzz"],
        "relax_gxy": post["gxy"] / pre["gxy"] if abs(pre["gxy"]) > 1e-14 else np.nan,
        "mean_cos_ggp": float(np.mean(cos_ggp)),
        "corr_mu_chi": corrcoef(mu, chi_rad) if chi_rad is not None else np.nan,
        "k_hat_frac": compute_k_hat_frac(g_post, g_pre, eij),
        "chi_hs_rmse": chi_hs_rmse,
    }
    if chi_rad is not None:
        row.update(compute_chi_mu_bins(mu, chi_rad))
    return row


def ctc_reference_row(name, alpha, AR, ctc_source):
    """Load CTC data and return a statistics row (no g_post needed)."""
    pattern = os.path.join(ctc_source, f"alpha_{alpha:.2f}_r1.00_AR{AR:.1f}", "chi.txt")
    # Try with trailing zero (e.g. 1.00) and without
    candidates = glob.glob(pattern)
    if not candidates:
        pattern2 = os.path.join(ctc_source, f"alpha_{alpha}_r1.00_AR{AR}", "chi.txt")
        candidates = glob.glob(pattern2)
    if not candidates:
        return {"kernel": name, "note": "CTC data not found"}

    raw = np.loadtxt(candidates[0])
    if raw.shape[1] < 4:
        return {"kernel": name, "note": "chi.txt has < 4 columns (no mu)"}

    mu = raw[:, 3]
    chi_rad = raw[:, 1]
    valid = np.isfinite(mu) & np.isfinite(chi_rad) & (mu >= 0) & (mu <= 1)
    mu, chi_rad = mu[valid], chi_rad[valid]

    chi_hs = chi_hs_formula(mu, alpha)
    row = {
        "kernel": name,
        "n_samples": int(mu.size),
        "corr_mu_chi": corrcoef(mu, chi_rad),
        "k_hat_frac": 0.0,  # not applicable
        "chi_hs_rmse": float(np.sqrt(np.mean((chi_rad - chi_hs) ** 2))),
    }
    row.update(compute_chi_mu_bins(mu, chi_rad))
    return row


# ---------------------------------------------------------------------------
# Kernel dispatch
# ---------------------------------------------------------------------------

_ALPHA_GLOBAL = 0.8  # set from args in main()


def run_kernel(name, g_pre, eij, mu, alpha, chi_samples, rng,
               mu_chi_model=None, mu_joint_model=None):
    """Run one kernel variant and return a summary dict."""
    global _ALPHA_GLOBAL
    _ALPHA_GLOBAL = alpha

    # ---- sphere (exact hard-sphere) ----
    if name == "sphere":
        cr = np.einsum("ij,ij->i", g_pre, eij)
        g_post = g_pre - (1.0 + alpha) * cr[:, None] * eij
        chi_rad = chi_hs_formula(mu, alpha)
        return summarize_kernel(name, g_pre, g_post, eij, mu, chi_rad)

    # ---- mu_joint_ar2 (Codex lookup — for comparison) ----
    if name == "mu_joint_ar2":
        if mu_joint_model is None:
            raise ValueError("mu_joint_ar2 requires --mu-joint-lookup")
        n = g_pre.shape[0]
        g_post = np.empty_like(g_pre)
        chi_rad = np.empty(n)
        gmag = np.linalg.norm(g_pre, axis=1)
        for i in range(n):
            row = mu_joint_model.sample(alpha, mu[i], rng=rng)
            gpost_mag = gmag[i] * np.sqrt(row["q_total"] * row["eps_tr_post"])
            g_post[i] = mu_plane_post_relative(g_pre[i], eij[i], row["chi_rad"], gpost_mag, rng=rng)
            chi_rad[i] = row["chi_rad"]
        return summarize_kernel(name, g_pre, g_post, eij, mu, chi_rad)

    # ---- parse AR from name ----
    if "ar1" in name:
        AR = 1.0
    elif "ar2" in name:
        AR = 2.0
    else:
        raise ValueError(f"Unknown kernel: {name}")

    # ---- chi source ----
    if name.endswith("_mu_beta_mpr") or name.endswith("_mu_beta_eps_eij"):
        # Conditional Beta chi(mu)
        if mu_chi_model is None:
            raise ValueError(f"Kernel {name} requires --mu-chi-coeffs")
        c_a, c_b, M, N, J, beta_exp = mu_chi_model
        chi_rad = np.array([
            sample_chi_given_mu(mu[i], alpha, AR, c_a, c_b, M, N, J, beta_exp, rng=rng)
            for i in range(g_pre.shape[0])
        ])
    elif name.endswith("_mu_coupled"):
        # Deterministic hard-sphere chi(mu)
        chi_rad = chi_hs_formula(mu, alpha)
    else:
        # Marginal chi
        chi_norm = chi_samples[AR]
        chi_rad = np.pi * chi_norm

    # ---- direction source ----
    if name.endswith("_mpr") or name.endswith("_mu_coupled"):
        g_post = _mpr_post_g(g_pre, eij, chi_rad)
    elif name.endswith("_rand"):
        g_post = _eps_eij_post_g(g_pre, eij, chi_rad, rng, fresh_azimuth=True)
    else:
        # eps_from_eij (known sign-convention bug, for comparison)
        g_post = _eps_eij_post_g(g_pre, eij, chi_rad, rng, fresh_azimuth=False)

    return summarize_kernel(name, g_pre, g_post, eij, mu, chi_rad)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

PRINT_COLUMNS = [
    "kernel",
    "k_hat_frac",
    "chi_hs_rmse",
    "relax_gxy",
    "corr_mu_chi",
    "chi_mu_00_02",
    "chi_mu_04_06",
    "chi_mu_08_10",
    "post_gxy",
]


def print_table(rows):
    widths = {col: max(len(col), 14) for col in PRINT_COLUMNS}
    widths["kernel"] = max(len(r.get("kernel", "")) for r in rows) + 2

    header = " ".join(col.rjust(widths[col]) for col in PRINT_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in rows:
        vals = []
        for col in PRINT_COLUMNS:
            val = row.get(col, "n/a")
            if col == "kernel":
                vals.append(str(val).rjust(widths[col]))
            elif isinstance(val, float):
                vals.append(f"{val:14.5g}".rjust(widths[col]))
            else:
                vals.append(str(val).rjust(widths[col]))
        print(" ".join(vals))


def write_summary(rows, out_dir, metadata):
    out_dir.mkdir(parents=True, exist_ok=True)
    all_keys = list({k for r in rows for k in r.keys()})
    csv_path = out_dir / "collision_kernel_summary.csv"
    json_path = out_dir / "collision_kernel_summary.json"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump({"metadata": metadata, "rows": rows}, f, indent=2)
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Argument parsing and main
# ---------------------------------------------------------------------------

DEFAULT_KERNELS = [
    "sphere",
    "sphcyl_ar1",                   # marginal chi + eps_from_eij (existing — broken)
    "sphcyl_ar1_fresh_azimuth",     # marginal chi + random eps
    "sphcyl_ar1_mu_coupled",        # deterministic chi_hs(mu) + mpr
    "sphcyl_ar1_mu_beta_eps_eij",   # conditional chi + eps_from_eij (broken direction)
    "sphcyl_ar1_mu_beta_mpr",       # conditional chi + mpr (NEW correct model)
    "sphcyl_ar2",                   # marginal chi + eps_from_eij (existing — broken)
    "sphcyl_ar2_mu_beta_mpr",       # conditional chi + mpr (NEW correct model for AR=2)
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare collision kernel scattering geometry variants"
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--n-samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--covariance", default="1.70,0.70,0.80,-0.55",
        help="Pre-collision g covariance: xx,yy,zz,xy or xx,yy,zz,xy,xz,yz",
    )
    parser.add_argument("--coeffs", default="models/scattering_coeffs.npz",
                        help="Marginal chi polynomial model (fallback)")
    parser.add_argument("--mu-chi-coeffs", default="models/mu_chi_beta_coeffs.npz",
                        help="Conditional chi Beta model (required for *_mu_beta_* kernels)")
    parser.add_argument("--mu-joint-lookup", default="models/mu_joint_ar2_lookup.npz")
    parser.add_argument("--mu-joint-source", default="../Coll_Models/results")
    parser.add_argument("--ctc-source", default="../Coll_Models/results",
                        help="CTC data root for empirical chi|mu reference rows")
    parser.add_argument("--output-dir", default="runs/collision_kernel_diagnostics")
    parser.add_argument(
        "--kernels", default=",".join(DEFAULT_KERNELS),
        help="Comma-separated list of kernel names to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    global _ALPHA_GLOBAL
    _ALPHA_GLOBAL = args.alpha

    rng = np.random.default_rng(args.seed)
    covariance = parse_covariance(args.covariance)
    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]

    print(f"Sampling {args.n_samples} collisions, alpha={args.alpha}")
    g_pre = sample_relative_velocities(args.n_samples, covariance, rng)
    eij, mu = sample_accepted_eij(g_pre, rng)

    # Marginal chi samples (rejection sampler — used only for marginal kernels)
    scattering_pdf = ScatteringPDF(args.coeffs)
    chi_samples = {}
    for AR in [1.0, 2.0]:
        if any(f"ar{int(AR)}" in k and "mu_beta" not in k and "mu_coupled" not in k
               and "sphere" not in k for k in kernels):
            chi_samples[AR] = scattering_pdf.sample(AR, args.alpha, args.n_samples, rng)

    # Conditional chi model
    mu_chi_model = None
    if os.path.exists(args.mu_chi_coeffs):
        mu_chi_model = load_mu_chi_model(args.mu_chi_coeffs)
        print(f"  Loaded conditional chi model: {args.mu_chi_coeffs}")
    else:
        print(f"  WARNING: mu-chi model not found at {args.mu_chi_coeffs}")

    # mu-joint lookup (Codex approach)
    mu_joint_model = None
    if "mu_joint_ar2" in kernels:
        mu_joint_model = MuJointAR2Model(
            args.mu_joint_lookup, source_root=args.mu_joint_source
        )

    # Run kernels
    rows = []
    for name in kernels:
        print(f"  Running kernel: {name} ...")
        try:
            row = run_kernel(
                name, g_pre, eij, mu, args.alpha, chi_samples, rng,
                mu_chi_model=mu_chi_model,
                mu_joint_model=mu_joint_model,
            )
            rows.append(row)
        except Exception as exc:
            print(f"    FAILED: {exc}")
            rows.append({"kernel": name, "error": str(exc)})

    # CTC empirical reference rows
    print("  Loading CTC empirical reference...")
    for AR in [1.0, 2.0]:
        label = f"ctc_ar{int(AR)}_alpha{int(args.alpha*100):03d}"
        row = ctc_reference_row(label, args.alpha, AR, args.ctc_source)
        rows.append(row)

    metadata = {
        "alpha": args.alpha,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "covariance": covariance.tolist(),
        "kernels": kernels,
        "mu_chi_coeffs": args.mu_chi_coeffs,
        "notes": (
            "k_hat_frac=0 means in-plane scattering; ~0.48 means random azimuth. "
            "sphcyl_*_mpr kernels use mu_plane_post_relative (correct). "
            "sphcyl_*_eps_eij use eps_from_eij which has a sign bug (out-of-plane). "
            "CTC rows have no g_post; k_hat_frac=0 is nominal."
        ),
    }

    print()
    print_table(rows)

    csv_path, json_path = write_summary(rows, Path(args.output_dir), metadata)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {json_path}")
    print()
    print("Key interpretation:")
    print("  k_hat_frac ~ 0.00 : g_post in (g, eij) plane  [in-plane, correct geometry]")
    print("  k_hat_frac ~ 0.48 : g_post azimuth random/wrong [out-of-plane]")
    print("  relax_gxy of sphcyl_ar1_mu_beta_mpr should match sphere -> model validated")


if __name__ == "__main__":
    main()
