#!/usr/bin/env python3
"""Diagnostic: does the spherocylinder collision kernel preserve the intended scattering?

The test:
  Use the ACTUAL scattering model (scattering_coeffs.npz) for AR=1, alpha=1.0
  — identical to what dsmc.py does at equilibration (p_chi_fn_eq, p_max_eq).

  For AR=1 elastic collisions the model should give isotropic scattering:
      p(chi) ∝ sin(chi * pi)  →  cos(chi) uniform on [-1, 1]

  If the kernel (sample_chi + update_velocities) is correct, the observed
  deflection angle cos(chi_obs) must also be uniform on [-1, 1].

Usage:
    python plot_hcs_cooling_diagnostic.py
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

sys.path.insert(0, ".")

from src.simulation.collision import CollisionModels, init_p_chi_distribution, sample_chi, update_velocities

# ---------------------------------------------------------------------------
# Load scattering model — same path as dsmc.py uses
# ---------------------------------------------------------------------------
MODEL_DIR = "models"
print("Loading collision models...")
models = CollisionModels(MODEL_DIR)

# AR=1, alpha=1.0 — this is exactly what dsmc.py uses for p_chi_fn_eq
p_chi_fn_eq, p_max_eq = init_p_chi_distribution(1.0, 1.0, models)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
N = 200_000
np.random.seed(42)

# ---------------------------------------------------------------------------
# Run: sample chi from the MODEL p(chi|AR=1,alpha=1), apply update_velocities
# ---------------------------------------------------------------------------
print(f"Running AR=1, alpha=1.0 kernel test ({N:,} collisions)...")

chi_samp    = np.empty(N)
cos_chi_obs = np.empty(N)

for i in range(N):
    v1 = np.random.randn(3)
    v2 = np.random.randn(3)
    g_pre = v1 - v2
    g_pre_mag = np.linalg.norm(g_pre)

    chi_s = sample_chi(p_chi_fn_eq, p_max_eq)          # chi in [0, 1]
    RR    = np.random.random()
    eps   = 2 * np.pi * RR                              # same as dsmc.py line 388-389

    chi_samp[i] = chi_s

    cr = g_pre_mag * 0.5                                # elastic: |g| preserved
    v1n, v2n = update_velocities(v1, v2, chi_s * np.pi, eps, cr)
    g_post = v1n[0] - v2n[0]
    g_post_mag = np.linalg.norm(g_post)

    if g_pre_mag > 1e-12 and g_post_mag > 1e-12:
        cos_chi_obs[i] = np.dot(g_pre, g_post) / (g_pre_mag * g_post_mag)
    else:
        cos_chi_obs[i] = np.nan

valid = np.isfinite(cos_chi_obs)
cos_obs_valid = cos_chi_obs[valid]

# KS test: is cos_chi_obs uniform on [-1, 1]?
u = (cos_obs_valid + 1.0) / 2.0
ks_stat, ks_p = kstest(u, "uniform")

print(f"\nResults:")
print(f"  cos(chi_obs): mean = {cos_obs_valid.mean():.4f}  (expected 0.0)")
print(f"  cos(chi_obs): std  = {cos_obs_valid.std():.4f}  (expected {1/np.sqrt(3):.4f})")
print(f"  KS test vs Uniform[-1,1]: stat = {ks_stat:.4f},  p = {ks_p:.4f}")
if ks_p > 0.01:
    print("  -> PASS: cos(chi_obs) is consistent with uniform [-1,1].")
    print("     The kernel correctly preserves isotropic scattering.")
else:
    print("  -> FAIL: cos(chi_obs) is NOT uniform. The kernel distorts the distribution!")

# Sanity: cos(chi_sampled) should also be uniform for AR=1, alpha=1
cos_samp_valid = np.cos(chi_samp * np.pi)
u_samp = (cos_samp_valid + 1.0) / 2.0
ks_stat_s, ks_p_s = kstest(u_samp, "uniform")
print(f"\n  Sanity — cos(chi_sampled) KS vs Uniform[-1,1]: "
      f"stat = {ks_stat_s:.4f},  p = {ks_p_s:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
BINS = 60
bins_cos = np.linspace(-1, 1, BINS + 1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel 1: cos(chi_sampled) from model — should be uniform for AR=1, alpha=1
ax = axes[0]
ax.hist(cos_samp_valid, bins=bins_cos, density=True, color="#D95F02", alpha=0.75,
        label=r"$\cos\chi$ from model")
ax.axhline(0.5, color="k", lw=1.5, ls="--", label="Uniform 0.5")
ax.set_xlabel(r"$\cos\chi_{\rm sampled}$", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    r"$\cos\chi$ from model $p(\chi|$AR=1$,\alpha=1)$" "\n"
    r"(should be flat for isotropic)",
    fontsize=10
)
ax.legend(fontsize=8)
ax.text(0.05, 0.95, f"KS p = {ks_p_s:.3f}", transform=ax.transAxes, fontsize=9, va="top")
ax.tick_params(labelsize=9)

# Panel 2: cos(chi_obs) after update_velocities — should also be uniform
ax = axes[1]
ax.hist(cos_obs_valid, bins=bins_cos, density=True, color="#2166AC", alpha=0.75,
        label=r"$\cos\chi_{\rm obs}$ (g deflection)")
ax.axhline(0.5, color="k", lw=1.5, ls="--", label="Uniform 0.5")
ax.set_xlabel(r"$\cos\chi_{\rm obs}$", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title(
    r"$\cos\chi_{\rm obs}$ after update_velocities" "\n"
    r"(AR=1, $\alpha=1$ model input)",
    fontsize=10
)
ax.legend(fontsize=8)
ax.text(0.05, 0.95, f"KS p = {ks_p:.3f}", transform=ax.transAxes, fontsize=9, va="top")
ax.tick_params(labelsize=9)

# Panel 3: overlay
ax = axes[2]
ax.hist(cos_samp_valid, bins=bins_cos, density=True, alpha=0.5,
        color="#D95F02", label=r"$\cos\chi$ sampled")
ax.hist(cos_obs_valid, bins=bins_cos, density=True, alpha=0.5,
        color="#2166AC", label=r"$\cos\chi_{\rm obs}$")
ax.axhline(0.5, color="k", lw=1.5, ls="--", label="Expected (uniform)")
ax.set_xlabel(r"$\cos\chi$", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("Overlay: sampled vs observed\n(should both be flat at 0.5)", fontsize=10)
ax.legend(fontsize=8)
ax.tick_params(labelsize=9)

result_str = "PASS" if ks_p > 0.01 else "FAIL"
fig.suptitle(
    f"AR=1, α=1 scattering kernel test  —  {result_str}  "
    f"(KS p={ks_p:.3f},  N={N//1000:,}k)",
    fontsize=11
)
fig.tight_layout()
out_path = "plot_hcs_cooling_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
