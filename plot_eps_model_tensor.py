#!/usr/bin/env python3
"""Validate eps_model + chi_hs against CTC empirical tensor components.

For every available (alpha, AR) case the script computes ghat_post
covariance components for three models and the CTC ground truth, then
produces three figures:

  Fig 1  k_hat_frac curves vs AR, one panel per alpha
  Fig 2  Tensor component scatter (model vs CTC) for gxx, gyy, gzz, gyz
  Fig 3  Grouped bar charts for AR=2.0 across all alphas

All computation is vectorised — typical runtime ≲ 60 s for all 66 cases.

Usage
-----
    python plot_eps_model_tensor.py [--ctc-source ../Coll_Models/results]
                                     [--eps-coeffs models/eps_azimuth_coeffs.npz]
                                     [--output-dir figures/eps_validation]
"""
import argparse
import os
import re
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.preprocessing.fit_eps_model import (
    _load_case_eps, load_eps_model, eval_kappa_vec,
)


# ---------------------------------------------------------------------------
# Geometry helpers (vectorised)
# ---------------------------------------------------------------------------

def _chi_hs_vec(mu_arr, alpha):
    mu_arr = np.clip(mu_arr, 0.0, 1.0)
    denom = np.sqrt(np.maximum(1.0 - (1.0 - alpha**2) * mu_arr**2, 1.0e-30))
    return np.arccos(np.clip((1.0 - (1.0 + alpha) * mu_arr**2) / denom, -1.0, 1.0))


def _scatter_vec_fixed_gpre(eij, chi_arr, eps_arr):
    """Vectorised scatter for fixed g_pre = (1,0,0) (DSMC convention).

    Returns ghat_post (N,3) unit vectors.
    """
    ey = eij[:, 1]
    ez = eij[:, 2]
    norm_perp = np.sqrt(ey**2 + ez**2)
    valid = norm_perp > 1.0e-10

    cos_chi = np.cos(chi_arr)
    sin_chi = np.sin(chi_arr)
    cos_eps = np.cos(eps_arr)
    sin_eps = np.sin(eps_arr)

    gpost = np.zeros((len(eij), 3))
    gpost[:, 0] = cos_chi
    np.divide(
        -sin_chi * (cos_eps * ey + sin_eps * ez),
        np.where(valid, norm_perp, 1.0),
        out=gpost[:, 1],
        where=valid,
    )
    np.divide(
        -sin_chi * (cos_eps * ez - sin_eps * ey),
        np.where(valid, norm_perp, 1.0),
        out=gpost[:, 2],
        where=valid,
    )
    # Normalise (guard rounding errors)
    mag = np.linalg.norm(gpost, axis=1, keepdims=True)
    gpost /= np.maximum(mag, 1.0e-30)
    return gpost


def _cov(g):
    return {
        "gxx": float(np.mean(g[:, 0]**2)),
        "gyy": float(np.mean(g[:, 1]**2)),
        "gzz": float(np.mean(g[:, 2]**2)),
        "gyz": float(np.mean(g[:, 1] * g[:, 2])),
    }


def _k_hat_frac_vec(gpost, eij):
    """k_hat_frac for gpost given g_pre=(1,0,0) and eij. Returns scalar."""
    ey = eij[:, 1]; ez = eij[:, 2]
    norm_perp = np.sqrt(ey**2 + ez**2)
    valid = norm_perp > 1.0e-10
    if not np.any(valid):
        return np.nan
    # n_perp_hat = (0, ey/norm, ez/norm), n2 = (0, -ez/norm, ey/norm)
    # k_hat = (1,0,0) x n_perp_hat = n2
    # but g_pre = (1,0,0), so k_hat = cross(g_pre_hat, n_perp_hat) = n2
    # |g_post . k_hat| = |gpost_y*(-ez/norm) + gpost_z*(ey/norm)|
    k_dot = np.abs(
        -gpost[valid, 1] * ez[valid] / norm_perp[valid]
        + gpost[valid, 2] * ey[valid] / norm_perp[valid]
    )
    gpost_mag = np.linalg.norm(gpost[valid], axis=1)
    return float(np.mean(k_dot / np.maximum(gpost_mag, 1.0e-30)))


# ---------------------------------------------------------------------------
# Per-case computation
# ---------------------------------------------------------------------------

_ALPHA_RE = re.compile(r"alpha_([0-9.]+)_r1\.00_AR([0-9.]+)")


def _load_ctc(chi_path):
    try:
        data = np.loadtxt(chi_path)
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 10:
        return None
    mu = data[:, 3]
    eij = data[:, 4:7]
    ghat_post_ctc = data[:, 7:10]
    chi_ctc = data[:, 1]
    eij_n = np.linalg.norm(eij, axis=1)
    post_n = np.linalg.norm(ghat_post_ctc, axis=1)
    valid = ((eij_n > 0.5) & (post_n > 0.5)
             & np.isfinite(mu) & (mu >= 0) & (mu <= 1)
             & (chi_ctc > 0.05))
    if valid.sum() < 100:
        return None
    eij = eij[valid] / eij_n[valid, None]
    ghat_post_ctc = ghat_post_ctc[valid] / post_n[valid, None]
    mu = mu[valid]
    return mu, eij, ghat_post_ctc


def compute_case(alpha, AR, chi_path, eps_model, rng):
    """Return dict with {ctc, eps_zero, eps_model, eps_iso} metrics."""
    result = _load_ctc(chi_path)
    if result is None:
        return None
    mu, eij, ghat_post_ctc = result

    # CTC: flip sign (CTC g_pre=(-1,0,0) → DSMC g_pre=(+1,0,0))
    gp_ctc = -ghat_post_ctc
    ctc = {**_cov(gp_ctc), "k_hat_frac": _k_hat_frac_vec(gp_ctc, eij)}

    chi_hs_arr = _chi_hs_vec(mu, alpha)
    N = len(mu)

    # eps_zero
    eps_zero = np.zeros(N)
    gp_zero = _scatter_vec_fixed_gpre(eij, chi_hs_arr, eps_zero)
    zero = {**_cov(gp_zero), "k_hat_frac": _k_hat_frac_vec(gp_zero, eij)}

    # eps_model (vonMises)
    if eps_model is not None:
        c_kappa, M_e, N_e, J_e, beta_exp_e = eps_model
        kappa_arr = eval_kappa_vec(mu, alpha, AR, c_kappa, M_e, N_e, J_e, beta_exp_e)
        kappa_arr = np.clip(kappa_arr, 0.0, 1.0e4)
        eps_mod = rng.vonmises(0.0, kappa_arr)
    else:
        eps_mod = np.zeros(N)
    gp_mod = _scatter_vec_fixed_gpre(eij, chi_hs_arr, eps_mod)
    mod = {**_cov(gp_mod), "k_hat_frac": _k_hat_frac_vec(gp_mod, eij)}

    # eps_isotropic
    eps_iso = rng.uniform(-np.pi, np.pi, N)
    gp_iso = _scatter_vec_fixed_gpre(eij, chi_hs_arr, eps_iso)
    iso = {**_cov(gp_iso), "k_hat_frac": _k_hat_frac_vec(gp_iso, eij)}

    return {"ctc": ctc, "eps_zero": zero, "eps_model": mod, "eps_iso": iso,
            "alpha": alpha, "AR": AR, "N": N}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_AR_COLORS = {1.1: "#1f77b4", 1.5: "#2ca02c", 2.0: "#d62728",
              2.5: "#9467bd", 3.0: "#ff7f0e"}
_AR_LABELS = {k: f"AR={k:.1f}" for k in _AR_COLORS}

SELECTED_ALPHAS = [0.50, 0.70, 0.90, 1.00]
ALL_ARS = [1.1, 1.5, 2.0, 2.5, 3.0]


def fig1_k_hat_frac(results, out_dir):
    """k_hat_frac vs AR, one panel per alpha."""
    alphas = sorted({r["alpha"] for r in results if r["alpha"] in SELECTED_ALPHAS})
    if not alphas:
        alphas = sorted({r["alpha"] for r in results})[:4]
    ncols = 2
    nrows = (len(alphas) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3.5 * nrows),
                             sharex=True, sharey=True)
    axes = np.array(axes).ravel()

    for ax_i, alpha_val in enumerate(alphas):
        ax = axes[ax_i]
        rows = sorted([r for r in results if abs(r["alpha"] - alpha_val) < 0.01],
                      key=lambda r: r["AR"])
        ar_vals = [r["AR"] for r in rows]
        ctc_k   = [r["ctc"]["k_hat_frac"] for r in rows]
        mod_k   = [r["eps_model"]["k_hat_frac"] for r in rows]
        zero_k  = [r["eps_zero"]["k_hat_frac"] for r in rows]
        iso_k   = [r["eps_iso"]["k_hat_frac"] for r in rows]

        ax.plot(ar_vals, ctc_k,  "ko-",  lw=2, ms=7, label="CTC empirical", zorder=5)
        ax.plot(ar_vals, mod_k,  "rs--", lw=2, ms=7, label="eps_model+chi_hs")
        ax.plot(ar_vals, zero_k, "b^:",  lw=1.5, ms=5, label="eps=0 (in-plane)")
        ax.plot(ar_vals, iso_k,  "gv:",  lw=1.5, ms=5, label="eps=uniform")
        ax.axhline(0, color="gray", lw=0.5, ls="--")
        ax.axhline(0.48, color="gray", lw=0.5, ls="--")
        ax.set_title(rf"$\alpha={alpha_val:.2f}$", fontsize=11)
        ax.set_ylabel("k_hat_frac")
        ax.set_xlabel("AR")
        ax.set_ylim(-0.02, 0.55)
        if ax_i == 0:
            ax.legend(fontsize=8, loc="upper right")

    for ax in axes[len(alphas):]:
        ax.set_visible(False)

    fig.suptitle("Out-of-plane fraction k_hat_frac: CTC vs eps_model+chi_hs",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_k_hat_frac.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig2_tensor_scatter(results, out_dir):
    """Scatter plot: model vs CTC for gxx, gyy, gzz, gyz."""
    components = ["gxx", "gyy", "gzz", "gyz"]
    titles = [r"$\langle\hat{g}_x^2\rangle$",
              r"$\langle\hat{g}_y^2\rangle$",
              r"$\langle\hat{g}_z^2\rangle$",
              r"$\langle\hat{g}_y\hat{g}_z\rangle$"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    ar_all = sorted({r["AR"] for r in results})
    cmap = cm.get_cmap("viridis", len(ar_all))
    ar_to_color = {ar: cmap(i) for i, ar in enumerate(ar_all)}

    for ax, comp, title in zip(axes, components, titles):
        for r in results:
            c = ar_to_color[r["AR"]]
            ctc_val = r["ctc"][comp]
            mod_val = r["eps_model"][comp]
            ax.scatter(ctc_val, mod_val, c=[c], s=40, alpha=0.85,
                       edgecolors="none",
                       label=f"AR={r['AR']:.1f}" if r["alpha"] == results[0]["alpha"] else "")

        lo = min(r["ctc"][comp] for r in results)
        hi = max(r["ctc"][comp] for r in results)
        pad = (hi - lo) * 0.05
        ref = [lo - pad, hi + pad]
        ax.plot(ref, ref, "k--", lw=1, zorder=0, label="1:1")
        ax.set_xlabel(f"CTC  {title}")
        ax.set_ylabel(f"Model  {title}")
        ax.set_title(title, fontsize=12)

    # Color-bar for AR
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(ar_all), vmax=max(ar_all)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical",
                        fraction=0.02, pad=0.04)
    cbar.set_label("AR", fontsize=10)
    cbar.set_ticks(ar_all)

    fig.suptitle("eps_model+chi_hs vs CTC — all (α, AR) cases", fontsize=12, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_tensor_scatter.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig3_bars_ar2(results, out_dir):
    """Grouped bar chart for AR=2.0 across all alphas."""
    rows = sorted([r for r in results if abs(r["AR"] - 2.0) < 0.05],
                  key=lambda r: r["alpha"])
    if not rows:
        print("  [Fig 3] No AR=2.0 cases found, skipping")
        return

    components = ["gxx", "gyy", "gzz", "gyz", "k_hat_frac"]
    comp_labels = [r"$g_{xx}$", r"$g_{yy}$", r"$g_{zz}$", r"$g_{yz}$",
                   r"$k_{\rm frac}$"]
    alphas = [r["alpha"] for r in rows]
    n_alpha = len(alphas)
    n_comp = len(components)

    fig, axes = plt.subplots(1, n_comp, figsize=(4 * n_comp, 4.5), sharey=False)
    x = np.arange(n_alpha)
    width = 0.28

    for ax, comp, label in zip(axes, components, comp_labels):
        ctc_vals  = [r["ctc"].get(comp, r["ctc"].get("k_hat_frac")) for r in rows]
        mod_vals  = [r["eps_model"].get(comp, r["eps_model"].get("k_hat_frac")) for r in rows]
        zero_vals = [r["eps_zero"].get(comp, r["eps_zero"].get("k_hat_frac")) for r in rows]

        # Fix: merge component and k_hat_frac lookup
        def _get(d, c):
            return d[c] if c in d else d["k_hat_frac"]

        ctc_vals  = [_get(r["ctc"], comp) for r in rows]
        mod_vals  = [_get(r["eps_model"], comp) for r in rows]
        zero_vals = [_get(r["eps_zero"], comp) for r in rows]

        b1 = ax.bar(x - width, ctc_vals,  width, label="CTC",
                    color="#2c7bb6", alpha=0.9, edgecolor="k", lw=0.5)
        b2 = ax.bar(x,          mod_vals,  width, label="eps_model+chi_hs",
                    color="#d7191c", alpha=0.9, edgecolor="k", lw=0.5)
        b3 = ax.bar(x + width,  zero_vals, width, label="eps=0+chi_hs",
                    color="#abdda4", alpha=0.9, edgecolor="k", lw=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:.2f}" for a in alphas], rotation=45, ha="right")
        ax.set_xlabel(r"$\alpha$")
        ax.set_title(label, fontsize=13)
        if comp == components[0]:
            ax.legend(fontsize=8)

    fig.suptitle("AR = 2.0 — tensor components vs α\n(CTC vs eps_model+chi_hs vs eps=0+chi_hs)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_bars_ar2.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def fig4_residuals(results, out_dir):
    """Relative error of model vs CTC for k_hat_frac, shown as heatmap (alpha x AR)."""
    alphas = sorted({r["alpha"] for r in results})
    ars    = sorted({r["AR"]    for r in results})

    data_grid = np.full((len(alphas), len(ars)), np.nan)
    for r in results:
        i = alphas.index(r["alpha"])
        j = ars.index(r["AR"])
        ctc = r["ctc"]["k_hat_frac"]
        mod = r["eps_model"]["k_hat_frac"]
        data_grid[i, j] = (mod - ctc) / max(ctc, 1e-4) * 100.0   # % error

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.nanmax(np.abs(data_grid))
    im = ax.imshow(data_grid, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("% error  (model − CTC) / CTC", fontsize=9)

    ax.set_xticks(range(len(ars)))
    ax.set_xticklabels([f"{a:.1f}" for a in ars])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.2f}" for a in alphas])
    ax.set_xlabel("AR")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"k_hat_frac relative error: eps_model+chi_hs vs CTC (%)")

    # Annotate cells
    for i in range(len(alphas)):
        for j in range(len(ars)):
            val = data_grid[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:+.0f}%", ha="center", va="center",
                        fontsize=7, color="k")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_khat_error_heatmap.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def summary_table(results):
    """Print a concise summary across all cases."""
    print(f"\n{'alpha':>6} {'AR':>5} {'N':>7}  "
          f"{'khat_ctc':>9} {'khat_mod':>9} {'err%':>6}  "
          f"{'gyz_ctc':>8} {'gyz_mod':>8}")
    print("-" * 74)
    for r in sorted(results, key=lambda x: (x["alpha"], x["AR"])):
        kc = r["ctc"]["k_hat_frac"]
        km = r["eps_model"]["k_hat_frac"]
        err = (km - kc) / max(kc, 1e-4) * 100
        print(f"{r['alpha']:6.2f} {r['AR']:5.1f} {r['N']:7d}  "
              f"{kc:9.4f} {km:9.4f} {err:+6.1f}%  "
              f"{r['ctc']['gyz']:8.4f} {r['eps_model']['gyz']:8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctc-source",  default="../Coll_Models/results")
    p.add_argument("--eps-coeffs",  default="models/eps_azimuth_coeffs.npz")
    p.add_argument("--output-dir",  default="figures/eps_validation")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    eps_model = load_eps_model(args.eps_coeffs)
    if eps_model is None:
        print("WARNING: eps model not found — eps=0 will be used for model variant")

    # Discover all cases
    pattern = os.path.join(args.ctc_source, "alpha_*_r1.00_AR*")
    cases = []
    for case_dir in sorted(glob.glob(pattern)):
        m = _ALPHA_RE.search(case_dir)
        if not m:
            continue
        try:
            alpha = float(m.group(1))
            AR    = float(m.group(2))
        except ValueError:
            continue
        chi_path = os.path.join(case_dir, "chi.txt")
        if os.path.exists(chi_path):
            cases.append((alpha, AR, chi_path))

    print(f"Found {len(cases)} CTC cases.  Computing tensor components ...")
    results = []
    for i, (alpha, AR, chi_path) in enumerate(cases):
        res = compute_case(alpha, AR, chi_path, eps_model, rng)
        if res is not None:
            results.append(res)
            print(f"  [{i+1:2d}/{len(cases)}] alpha={alpha:.2f} AR={AR:.1f}  "
                  f"k_hat CTC={res['ctc']['k_hat_frac']:.4f}  "
                  f"model={res['eps_model']['k_hat_frac']:.4f}")

    print(f"\nProcessed {len(results)}/{len(cases)} cases.")
    summary_table(results)

    print("\nGenerating figures ...")
    fig1_k_hat_frac(results, args.output_dir)
    fig2_tensor_scatter(results, args.output_dir)
    fig3_bars_ar2(results, args.output_dir)
    fig4_residuals(results, args.output_dir)

    print(f"\nAll figures written to {args.output_dir}/")
    print("fig1 — k_hat_frac curves vs AR per alpha")
    print("fig2 — tensor component scatter (model vs CTC), all cases")
    print("fig3 — grouped bars for AR=2.0 across all alpha")
    print("fig4 — k_hat_frac % error heatmap (alpha x AR)")


if __name__ == "__main__":
    main()
