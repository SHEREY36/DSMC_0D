"""Publication-quality figure functions for the HCS non-Gaussian paper.

Each fig_N_* function takes pre-loaded campaign data and writes a figure to
output_path.  All data loading goes through non_gaussian.py helpers so that
the theta-divergence masking is applied consistently.
"""

import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .non_gaussian import (
    aggregate_histograms,
    aggregate_histograms_with_validity_filter,
    aggregate_ng_summaries_from_moments,
    brey_a2_tr,
    histogram_ratio_to_reference,
    load_base_ensemble,
    load_theta_star_table,
    aggregate_moments_timeseries,
    maxwell_energy_coupling_pdf,
    maxwell_speed_pdf,
    rayleigh_rot_speed_pdf,
    sonine_speed_ratio,
    sonine_rot_speed_ratio,
)


# ---------------------------------------------------------------------------
# Constants / style
# ---------------------------------------------------------------------------

AR_LABELS = {1.5: "AR150", 2.0: "AR200", 2.5: "AR250", 3.0: "AR300"}
AR_LABEL_TEX = {1.5: r"$\mathrm{AR}=1.5$", 2.0: r"$\mathrm{AR}=2.0$",
                2.5: r"$\mathrm{AR}=2.5$", 3.0: r"$\mathrm{AR}=3.0$"}
ALPHA_DIR = {0.60: "alpha_060", 0.65: "alpha_065", 0.70: "alpha_070",
             0.75: "alpha_075", 0.80: "alpha_080", 0.85: "alpha_085",
             0.90: "alpha_090", 0.95: "alpha_095"}

_VIRIDIS = plt.get_cmap("viridis")
_PLASMA = plt.get_cmap("plasma")


def _ar_color(ar, ar_list):
    n = max(len(ar_list) - 1, 1)
    return _VIRIDIS(ar_list.index(ar) / n)


def _alpha_color(alpha, alpha_list):
    n = max(len(alpha_list) - 1, 1)
    return _PLASMA(alpha_list.index(alpha) / n)


def _style_ax(ax):
    ax.tick_params(axis="both", direction="in", which="both", right=True, top=True)
    ax.grid(True, alpha=0.22)


def _save(fig, path):
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _results_dir(root, ar, alpha):
    ar_dir = AR_LABELS[ar]
    alpha_dir = ALPHA_DIR[alpha]
    return os.path.join(root, ar_dir, alpha_dir, "results")


def load_campaign_summaries(root, ar_values=None, alpha_values=None,
                             theta_abs_max=2.0):
    """Load theta-truncated cumulant aggregates for all (AR, alpha) cases.

    Returns rows: list of dicts with AR, alpha, cumulants, moments, nu, etc.
    theta_abs_max controls the divergence cutoff applied to ng_moments theta column.
    """
    if ar_values is None:
        ar_values = [1.5, 2.0, 2.5, 3.0]
    if alpha_values is None:
        alpha_values = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    rows = []
    for ar in ar_values:
        for alpha in alpha_values:
            rdir = _results_dir(root, ar, alpha)
            if not os.path.isdir(rdir):
                continue
            try:
                agg = aggregate_ng_summaries_from_moments(rdir, theta_abs_max=theta_abs_max)
            except FileNotFoundError:
                continue
            row = {"AR": ar, "alpha": alpha,
                   "n_realizations": agg["n_realizations"],
                   "n_particle_samples": agg["n_particle_samples"],
                   "collision_frequency": agg["collision_frequency"]["mean"],
                   "collision_frequency_stderr": agg["collision_frequency"]["stderr"]}
            for col, stats in agg["cumulants"].items():
                row[col] = stats["mean"]
                row[f"{col}_stderr"] = stats["stderr"]
            for col, stats in agg["moments"].items():
                row[col] = stats["mean"]
                row[f"{col}_stderr"] = stats["stderr"]
            rows.append(row)
    rows.sort(key=lambda r: (r["AR"], r["alpha"]))
    return rows


def load_campaign_histograms(root, ar_values=None, alpha_values=None,
                              min_valid_fraction=0.85, theta_abs_max=2.0):
    """Load validity-filtered averaged histograms for all (AR, alpha) cases.

    Each histogram is averaged only over seeds whose ng_moments theta column
    stays below theta_abs_max for at least min_valid_fraction of their samples.
    This removes realizations where the hcs_rescale artifact (Trot → 0) has
    contaminated the accumulated histogram data.

    Returns hist_data: dict keyed by (ar, alpha) → {speed, rot_speed, energy_coupling, ...}
    """
    if ar_values is None:
        ar_values = [1.5, 2.0, 2.5, 3.0]
    if alpha_values is None:
        alpha_values = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    specs = {
        "speed": "_ng_hist_speed.txt",
        "rot_speed": "_ng_hist_rot_speed.txt",
        "energy_coupling": "_ng_hist_energy_coupling.txt",
        "energy_tr": "_ng_hist_energy_tr.txt",
        "energy_rot": "_ng_hist_energy_rot.txt",
    }
    hist_data = {}
    for ar in ar_values:
        for alpha in alpha_values:
            rdir = _results_dir(root, ar, alpha)
            if not os.path.isdir(rdir):
                continue
            case = {}
            for name, suffix in specs.items():
                try:
                    hist = aggregate_histograms_with_validity_filter(
                        rdir, suffix,
                        min_valid_fraction=min_valid_fraction,
                        theta_abs_max=theta_abs_max,
                    )
                    if not np.any(hist["density_mean"] > 0.0):
                        continue
                    case[name] = hist
                except FileNotFoundError:
                    pass
            if case:
                hist_data[(ar, alpha)] = case
    return hist_data


# ---------------------------------------------------------------------------
# Fig 1 — θ(τ) transient evolution
# ---------------------------------------------------------------------------

def fig1_theta_transient(root, output_path, tau_max=120.0,
                          fixed_alpha=0.70, fixed_ar=2.0,
                          ar_sweep=(1.5, 2.0, 2.5, 3.0),
                          alpha_sweep=(0.60, 0.70, 0.80, 0.90, 0.95)):
    """Two-panel θ(τ) transient from base .txt (pre-production window).

    Panel (a): fixed α, curves for multiple AR.
    Panel (b): fixed AR, curves for multiple α.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=False)

    # Panel (a): vary AR
    ax = axes[0]
    ar_list = list(ar_sweep)
    for ar in ar_list:
        rdir = _results_dir(root, ar, fixed_alpha)
        if not os.path.isdir(rdir):
            continue
        try:
            tau_g, mean, stderr = load_base_ensemble(rdir, tau_max=tau_max)
        except FileNotFoundError:
            continue
        c = _ar_color(ar, ar_list)
        mask = np.isfinite(mean)
        ax.plot(tau_g[mask], mean[mask], color=c, lw=1.8, label=AR_LABEL_TEX[ar])
        ax.fill_between(tau_g[mask], mean[mask] - stderr[mask],
                        mean[mask] + stderr[mask], color=c, alpha=0.15, lw=0)
    ax.axhline(1.0, color="0.3", lw=0.9, ls="--", alpha=0.6, label=r"$\theta=1$ (equipartition)")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\theta = T_{\mathrm{tr}}/T_{\mathrm{rot}}$")
    ax.set_title(fr"$\alpha = {fixed_alpha:.2f}$")
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax)

    # Panel (b): vary α
    ax = axes[1]
    alpha_list = list(alpha_sweep)
    for alpha in alpha_list:
        rdir = _results_dir(root, fixed_ar, alpha)
        if not os.path.isdir(rdir):
            continue
        try:
            tau_g, mean, stderr = load_base_ensemble(rdir, tau_max=tau_max)
        except FileNotFoundError:
            continue
        c = _alpha_color(alpha, alpha_list)
        mask = np.isfinite(mean)
        ax.plot(tau_g[mask], mean[mask], color=c, lw=1.8,
                label=fr"$\alpha={alpha:.2f}$")
        ax.fill_between(tau_g[mask], mean[mask] - stderr[mask],
                        mean[mask] + stderr[mask], color=c, alpha=0.15, lw=0)
    ax.axhline(1.0, color="0.3", lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\theta = T_{\mathrm{tr}}/T_{\mathrm{rot}}$")
    ax.set_title(fr"$\mathrm{{AR}} = {fixed_ar:g}$")
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax)

    fig.suptitle(r"Temperature ratio $\theta(\tau)$ relaxation to HCS", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 2 — Steady-state θ*(α) from LAMMPS table
# ---------------------------------------------------------------------------

def fig2_theta_star(models_dir, output_path,
                    ar_values=(1.5, 2.0, 2.5, 3.0),
                    ar_label_map=None):
    """θ*(α) per AR from LAMMPS theta_target_table. Sphere baseline θ*=1."""
    ar_label_map = ar_label_map or {"1.5": "15", "2.0": "20", "2.5": "25", "3.0": "30"}
    ar_list = list(ar_values)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    for ar in ar_list:
        ar_str = f"{ar:g}"
        label_code = ar_label_map.get(ar_str)
        if label_code is None:
            continue
        try:
            table = load_theta_star_table(models_dir, label_code)
        except FileNotFoundError:
            continue
        alphas = sorted(table.keys())
        thetas = [table[a] for a in alphas]
        c = _ar_color(ar, ar_list)
        ax.plot(alphas, thetas, marker="o", ms=4.5, lw=1.8,
                color=c, label=AR_LABEL_TEX[ar])

    ax.axhline(1.0, color="0.25", lw=1.0, ls="--",
               label=r"$\theta^*=1$ (spheres)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\theta^*$")
    ax.set_title(r"Steady-state temperature ratio $\theta^*(\alpha)$")
    ax.legend(fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 3 — Cooling rate ζ*/ν (sphere baseline only)
# ---------------------------------------------------------------------------

def fig3_cooling_rate(output_path, alpha_range=(0.60, 0.95)):
    """ζ*/ν vs α — sphere Haff baseline only.

    Spherocylinder curves require separate unscaled runs; this is a stub.
    """
    alpha = np.linspace(alpha_range[0], alpha_range[1], 200)
    zeta_sphere = (5.0 / 12.0) * (1.0 - alpha ** 2)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(alpha, zeta_sphere, color="0.2", lw=1.6, ls="--",
            label=r"Smooth spheres: $\frac{5}{12}(1-\alpha^2)$")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\zeta^*/\nu$")
    ax.set_title(r"Reduced cooling rate $\zeta^*/\nu$")
    ax.legend(fontsize=9)
    ax.text(0.98, 0.92,
            "Spherocylinder curves require\nseparate unscaled runs",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color="0.45", style="italic")
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 4 — Collision frequency ν(AR)/ν_ref vs AR
# ---------------------------------------------------------------------------

def fig4_collision_frequency(rows, output_path,
                              fixed_alpha=0.80,
                              ar_ref_for_sphere=None):
    """ν(AR)/ν_sphere vs AR at fixed α.

    ν is from ng_summary (collision_frequency = NColl/(2*Np*t_final)).
    Normalises by the smallest AR in the dataset if no sphere (AR=1.0) available.
    """
    sub = [r for r in rows if abs(r["alpha"] - fixed_alpha) < 1e-4]
    if not sub:
        return
    sub.sort(key=lambda r: r["AR"])
    ars = np.array([r["AR"] for r in sub])
    nus = np.array([r["collision_frequency"] for r in sub])
    errs = np.array([r["collision_frequency_stderr"] for r in sub])

    # Reference: use AR=1.5 (smallest in scope) as the denominator
    nu_ref = nus[0]
    if ar_ref_for_sphere is not None:
        nu_ref = ar_ref_for_sphere
    ratio = nus / nu_ref
    ratio_err = errs / nu_ref

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.errorbar(ars, ratio, yerr=ratio_err, marker="o", ms=5, lw=1.8,
                capsize=3, color=_VIRIDIS(0.55))
    ax.axhline(1.0, color="0.25", lw=0.9, ls="--", alpha=0.6)
    ax.set_xlabel(r"$\mathrm{AR}$")
    ax.set_ylabel(r"$\nu(\mathrm{AR})/\nu_{\mathrm{ref}}$")
    ax.set_title(fr"Collision frequency ratio, $\alpha={fixed_alpha:.2f}$ "
                 fr"(ref AR={ars[0]:g})")
    ax.set_xticks(ars)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 5 — a₂ᵗʳ vs α with Brey baseline
# ---------------------------------------------------------------------------

def fig5_a2tr(rows, output_path, ar_values=(1.5, 2.0, 2.5, 3.0)):
    """a₂ᵗʳ vs α — one curve per AR; Brey first-Sonine dashed baseline."""
    ar_list = list(ar_values)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for ar in ar_list:
        sub = [r for r in rows if abs(r["AR"] - ar) < 1e-4]
        if not sub:
            continue
        sub.sort(key=lambda r: r["alpha"])
        x = np.array([r["alpha"] for r in sub])
        y = np.array([r.get("a2_tr", np.nan) for r in sub], dtype=float)
        err = np.array([r.get("a2_tr_stderr", np.nan) for r in sub], dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        c = _ar_color(ar, ar_list)
        ax.errorbar(x[mask], y[mask], yerr=err[mask], marker="o", ms=4.5,
                    lw=1.8, capsize=2.5, color=c, label=AR_LABEL_TEX[ar])

    # Brey smooth-sphere baseline
    alpha_ref = np.linspace(0.60, 0.96, 200)
    ax.plot(alpha_ref, brey_a2_tr(alpha_ref), color="0.2", lw=1.3, ls="--",
            label=r"Brey (spheres, Eq. 18)")
    ax.axhline(0.0, color="0.5", lw=0.7, alpha=0.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a_{2,\mathrm{tr}}$")
    ax.set_title(r"Translational cumulant $a_{2,\mathrm{tr}}(\alpha)$")
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 6 — a₃ᵗʳ (sixth-moment ratio) vs α
# ---------------------------------------------------------------------------

def fig6_a3tr(rows, output_path, ar_values=(1.5, 2.0, 2.5, 3.0)):
    """a₃ᵗʳ = (8/105)*⟨c⁶⟩ - 1 vs α — one curve per AR; no analytic baseline."""
    ar_list = list(ar_values)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for ar in ar_list:
        sub = [r for r in rows if abs(r["AR"] - ar) < 1e-4]
        if not sub:
            continue
        sub.sort(key=lambda r: r["alpha"])
        x = np.array([r["alpha"] for r in sub])
        y = np.array([r.get("a3_tr", np.nan) for r in sub], dtype=float)
        err = np.array([r.get("a3_tr_stderr", np.nan) for r in sub], dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        c = _ar_color(ar, ar_list)
        ax.errorbar(x[mask], y[mask], yerr=err[mask], marker="o", ms=4.5,
                    lw=1.8, capsize=2.5, color=c, label=AR_LABEL_TEX[ar])

    ax.axhline(0.0, color="0.25", lw=1.0, ls="--", alpha=0.7,
               label=r"$a_{3,\mathrm{tr}}=0$ (Maxwellian)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a_{3,\mathrm{tr}}$")
    ax.set_title(r"Sixth-moment ratio $a_{3,\mathrm{tr}}(\alpha)$")
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 7 — Marginal translational VDF ratio φ_c / φ_{c,M}
# ---------------------------------------------------------------------------

def fig7_speed_ratio(hist_data, rows, output_path,
                     ar_panels=(1.5, 2.0, 3.0),
                     alpha_curves=(0.70, 0.90),
                     c_max_linear=3.4, c_max_log=5.5):
    """φ_c(c)/φ_{c,M}(c) vs c — 3-panel by AR, each panel 2 α curves.

    Main axis: linear y in [0.88, 1.12].
    Inset: log-linear showing tail out to c_max_log.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=False)
    alpha_list = list(alpha_curves)

    row_lookup = {(r["AR"], r["alpha"]): r for r in rows}

    for ax, ar in zip(axes, ar_panels):
        for alpha in alpha_list:
            hist = hist_data.get((ar, alpha), {}).get("speed")
            if hist is None:
                continue
            ref = maxwell_speed_pdf(hist["centers"])
            ratio = np.where(ref > 0.0, hist["density_mean"] / ref, np.nan)
            stderr_ratio = np.where(ref > 0.0, hist["density_stderr"] / ref, np.nan)
            mask = np.isfinite(ratio) & (hist["centers"] <= c_max_linear)
            c = _alpha_color(alpha, alpha_list)
            ax.plot(hist["centers"][mask], ratio[mask], lw=1.6, color=c,
                    label=fr"$\alpha={alpha:.2f}$")
            ax.fill_between(hist["centers"][mask],
                            ratio[mask] - stderr_ratio[mask],
                            ratio[mask] + stderr_ratio[mask],
                            color=c, alpha=0.12, lw=0)

            # Sonine overlay using measured a₂ᵗʳ for this case
            row = row_lookup.get((ar, alpha))
            if row and np.isfinite(row.get("a2_tr", np.nan)):
                c_ref = np.linspace(0.01, c_max_linear, 300)
                sonine = sonine_speed_ratio(c_ref, row["a2_tr"])
                ax.plot(c_ref, sonine, lw=1.0, ls=":", color=c, alpha=0.75)

        ax.axhline(1.0, color="0.2", lw=0.9, ls="--", alpha=0.6)
        ax.set_xlim(0.0, c_max_linear)
        ax.set_ylim(0.88, 1.12)
        ax.set_xlabel(r"$c = |\mathbf{v}|/\sqrt{2T_{\mathrm{tr}}/m}$")
        ax.set_ylabel(r"$\phi_c / \phi_{c,M}$")
        ax.set_title(AR_LABEL_TEX[ar])
        ax.legend(fontsize=8)
        _style_ax(ax)

        # Log tail inset
        inset = ax.inset_axes([0.52, 0.10, 0.45, 0.42])
        for alpha in alpha_list:
            hist = hist_data.get((ar, alpha), {}).get("speed")
            if hist is None:
                continue
            c_bins = hist["centers"]
            density = hist["density_mean"]
            ref = maxwell_speed_pdf(c_bins)
            mask_log = (density > 0.0) & (ref > 0.0) & (c_bins >= 1.0) & (c_bins <= c_max_log)
            if not np.any(mask_log):
                continue
            col = _alpha_color(alpha, alpha_list)
            inset.semilogy(c_bins[mask_log], density[mask_log], lw=1.1, color=col)

        c_ref_log = np.linspace(1.0, c_max_log, 300)
        inset.semilogy(c_ref_log, maxwell_speed_pdf(c_ref_log), color="0.2",
                       lw=0.9, ls="--")
        inset.set_xlim(1.0, c_max_log)
        inset.set_xlabel(r"$c$", fontsize=7)
        inset.set_ylabel(r"$\phi_c$", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(True, alpha=0.2)

    fig.suptitle(r"Translational speed VDF ratio $\phi_c/\phi_{c,M}$", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 8 — a₂ʳᵒᵗ vs α
# ---------------------------------------------------------------------------

def fig8_a2rot(rows, output_path, ar_values=(1.5, 2.0, 2.5, 3.0)):
    """a₂ʳᵒᵗ vs α — one curve per AR; a₂ʳᵒᵗ=0 smooth-sphere baseline."""
    ar_list = list(ar_values)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for ar in ar_list:
        sub = [r for r in rows if abs(r["AR"] - ar) < 1e-4]
        if not sub:
            continue
        sub.sort(key=lambda r: r["alpha"])
        x = np.array([r["alpha"] for r in sub])
        y = np.array([r.get("a2_rot", np.nan) for r in sub], dtype=float)
        err = np.array([r.get("a2_rot_stderr", np.nan) for r in sub], dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        c = _ar_color(ar, ar_list)
        ax.errorbar(x[mask], y[mask], yerr=err[mask], marker="o", ms=4.5,
                    lw=1.8, capsize=2.5, color=c, label=AR_LABEL_TEX[ar])

    ax.axhline(0.0, color="0.25", lw=1.0, ls="--",
               label=r"$a_{2,\mathrm{rot}}=0$ (smooth spheres)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a_{2,\mathrm{rot}}$")
    ax.set_title(r"Rotational cumulant $a_{2,\mathrm{rot}}(\alpha)$")
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 9 — a₁₁ (cross-cumulant) vs α
# ---------------------------------------------------------------------------

def fig9_a11(rows, output_path, ar_values=(1.5, 2.0, 2.5, 3.0)):
    """a₁₁ = (2/3)*⟨c²w²⟩ - 1 vs α — one curve per AR; a₁₁=0 baseline."""
    ar_list = list(ar_values)
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    for ar in ar_list:
        sub = [r for r in rows if abs(r["AR"] - ar) < 1e-4]
        if not sub:
            continue
        sub.sort(key=lambda r: r["alpha"])
        x = np.array([r["alpha"] for r in sub])
        y = np.array([r.get("a11", np.nan) for r in sub], dtype=float)
        err = np.array([r.get("a11_stderr", np.nan) for r in sub], dtype=float)
        mask = np.isfinite(y)
        if not np.any(mask):
            continue
        c = _ar_color(ar, ar_list)
        ax.errorbar(x[mask], y[mask], yerr=err[mask], marker="o", ms=4.5,
                    lw=1.8, capsize=2.5, color=c, label=AR_LABEL_TEX[ar])

    ax.axhline(0.0, color="0.25", lw=1.0, ls="--",
               label=r"$a_{11}=0$ (factorizable VDF)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$a_{11}$")
    ax.set_title(r"Cross-cumulant $a_{11}(\alpha)$")
    ax.legend(fontsize=8, ncol=2)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 10 — Marginal rotational VDF ratio φ_w / φ_{w,M}
# ---------------------------------------------------------------------------

def fig10_rot_speed_ratio(hist_data, rows, output_path,
                           ar_panels=(1.5, 2.0, 3.0),
                           alpha_curves=(0.70, 0.90),
                           w_max_linear=3.2, w_max_log=5.5):
    """φ_w(w)/φ_{w,M}(w) vs w — 3-panel by AR, each panel 2 α curves."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=False)
    alpha_list = list(alpha_curves)
    row_lookup = {(r["AR"], r["alpha"]): r for r in rows}

    for ax, ar in zip(axes, ar_panels):
        for alpha in alpha_list:
            hist = hist_data.get((ar, alpha), {}).get("rot_speed")
            if hist is None:
                continue
            ref = rayleigh_rot_speed_pdf(hist["centers"])
            ratio = np.where(ref > 0.0, hist["density_mean"] / ref, np.nan)
            stderr_ratio = np.where(ref > 0.0, hist["density_stderr"] / ref, np.nan)
            mask = np.isfinite(ratio) & (hist["centers"] <= w_max_linear)
            c = _alpha_color(alpha, alpha_list)
            ax.plot(hist["centers"][mask], ratio[mask], lw=1.6, color=c,
                    label=fr"$\alpha={alpha:.2f}$")
            ax.fill_between(hist["centers"][mask],
                            ratio[mask] - stderr_ratio[mask],
                            ratio[mask] + stderr_ratio[mask],
                            color=c, alpha=0.12, lw=0)

            row = row_lookup.get((ar, alpha))
            if row and np.isfinite(row.get("a2_rot", np.nan)):
                w_ref = np.linspace(0.01, w_max_linear, 300)
                sonine = sonine_rot_speed_ratio(w_ref, row["a2_rot"])
                ax.plot(w_ref, sonine, lw=1.0, ls=":", color=c, alpha=0.75)

        ax.axhline(1.0, color="0.2", lw=0.9, ls="--", alpha=0.6)
        ax.set_xlim(0.0, w_max_linear)
        ax.set_ylim(0.82, 1.20)
        ax.set_xlabel(r"$w = \sqrt{E_r/T_{\mathrm{rot}}}$")
        ax.set_ylabel(r"$\phi_w / \phi_{w,M}$")
        ax.set_title(AR_LABEL_TEX[ar])
        ax.legend(fontsize=8)
        _style_ax(ax)

        # Log tail inset
        inset = ax.inset_axes([0.52, 0.10, 0.45, 0.42])
        for alpha in alpha_list:
            hist = hist_data.get((ar, alpha), {}).get("rot_speed")
            if hist is None:
                continue
            w_bins = hist["centers"]
            density = hist["density_mean"]
            ref = rayleigh_rot_speed_pdf(w_bins)
            mask_log = (density > 0.0) & (ref > 0.0) & (w_bins >= 0.5) & (w_bins <= w_max_log)
            if not np.any(mask_log):
                continue
            col = _alpha_color(alpha, alpha_list)
            inset.semilogy(w_bins[mask_log], density[mask_log], lw=1.1, color=col)

        w_ref_log = np.linspace(0.5, w_max_log, 300)
        inset.semilogy(w_ref_log, rayleigh_rot_speed_pdf(w_ref_log), color="0.2",
                       lw=0.9, ls="--")
        inset.set_xlim(0.5, w_max_log)
        inset.set_xlabel(r"$w$", fontsize=7)
        inset.set_ylabel(r"$\phi_w$", fontsize=7)
        inset.tick_params(labelsize=6)
        inset.grid(True, alpha=0.2)

    fig.suptitle(r"Rotational speed VDF ratio $\phi_w/\phi_{w,M}$", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 11 — Coupling distribution φ_{cw}(x) vs x = c²w²
# ---------------------------------------------------------------------------

def fig11_coupling_vdf(hist_data, output_path,
                        fixed_ar=2.0, fixed_alpha=0.70,
                        ar_sweep=(1.5, 2.0, 2.5, 3.0),
                        alpha_sweep=(0.60, 0.70, 0.80, 0.90),
                        x_max=16.0):
    """φ_{cw}(x) vs x on log-log — two panels: sweep α (fixed AR) and sweep AR (fixed α)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))

    x_ref = np.logspace(-2, np.log10(x_max), 400)
    ref_pdf = maxwell_energy_coupling_pdf(x_ref)

    # Panel (a): fix AR, sweep α
    ax = axes[0]
    alpha_list = list(alpha_sweep)
    for alpha in alpha_list:
        hist = hist_data.get((fixed_ar, alpha), {}).get("energy_coupling")
        if hist is None:
            continue
        x = hist["centers"]
        y = hist["density_mean"]
        mask = (y > 0.0) & (x > 0.0) & (x <= x_max)
        if not np.any(mask):
            continue
        c = _alpha_color(alpha, alpha_list)
        ax.loglog(x[mask], y[mask], lw=1.6, color=c, label=fr"$\alpha={alpha:.2f}$")

    ax.loglog(x_ref, ref_pdf, color="0.2", lw=1.3, ls="--",
              label=r"$2e^{-2\sqrt{x}}$ (Maxwellian)")
    ax.set_xlim(1e-1, x_max)
    ax.set_ylim(1e-5, 3.0)
    ax.set_xlabel(r"$x = \epsilon_t \epsilon_r$")
    ax.set_ylabel(r"$\phi_{cw}(x)$")
    ax.set_title(fr"$\mathrm{{AR}}={fixed_ar:g}$, varying $\alpha$")
    ax.legend(fontsize=8)
    _style_ax(ax)

    # Panel (b): fix α, sweep AR
    ax = axes[1]
    ar_list = list(ar_sweep)
    for ar in ar_list:
        hist = hist_data.get((ar, fixed_alpha), {}).get("energy_coupling")
        if hist is None:
            continue
        x = hist["centers"]
        y = hist["density_mean"]
        mask = (y > 0.0) & (x > 0.0) & (x <= x_max)
        if not np.any(mask):
            continue
        c = _ar_color(ar, ar_list)
        ax.loglog(x[mask], y[mask], lw=1.6, color=c, label=AR_LABEL_TEX[ar])

    ax.loglog(x_ref, ref_pdf, color="0.2", lw=1.3, ls="--",
              label=r"$2e^{-2\sqrt{x}}$")
    ax.set_xlim(1e-1, x_max)
    ax.set_ylim(1e-5, 3.0)
    ax.set_xlabel(r"$x = \epsilon_t \epsilon_r$")
    ax.set_ylabel(r"$\phi_{cw}(x)$")
    ax.set_title(fr"$\alpha={fixed_alpha:.2f}$, varying AR")
    ax.legend(fontsize=8)
    _style_ax(ax)

    fig.suptitle(r"Trans-rot coupling distribution $\phi_{cw}(x)$", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 12 — Transient cumulant evolution a₂ᵗʳ(τ), a₂ʳᵒᵗ(τ), a₁₁(τ)
# ---------------------------------------------------------------------------

def fig12_cumulant_transient(root, output_path, ar=2.0, alpha=0.70,
                              sample_start_tau=500.0, sample_end_tau=1500.0,
                              theta_abs_max=2.0):
    """Stacked 3-panel showing a₂ᵗʳ, a₂ʳᵒᵗ, a₁₁ vs τ with per-seed traces.

    Individual seeds shown as thin light lines; ensemble mean as thick line.
    Production window [sample_start_tau, sample_end_tau] shaded.
    theta_abs_max: divergence cutoff — samples with theta > this value are excluded.
    """
    rdir = _results_dir(root, ar, alpha)
    try:
        ts = aggregate_moments_timeseries(rdir, theta_abs_max=theta_abs_max)
    except FileNotFoundError:
        print(f"fig12: no moments data for AR={ar}, α={alpha}")
        return

    tau = ts["tau_grid"]
    specs = [
        ("a2_tr", r"$a_{2,\mathrm{tr}}(\tau)$"),
        ("a2_rot", r"$a_{2,\mathrm{rot}}(\tau)$"),
        ("a11", r"$a_{11}(\tau)$"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True)
    shade_color = "#CCDDEE"

    for ax, (col, ylabel) in zip(axes, specs):
        data = ts["per_col"].get(col)
        if data is None:
            ax.set_ylabel(ylabel)
            continue

        # Individual seed traces
        traces = data["traces"]
        for i in range(traces.shape[0]):
            ax.plot(tau, traces[i], color="0.6", lw=0.5, alpha=0.35)

        # Ensemble mean
        ax.plot(tau, data["mean"], color="C0", lw=2.0, label="ensemble mean")

        # Production window shade
        ax.axvspan(sample_start_tau, sample_end_tau, color=shade_color,
                   alpha=0.35, zorder=0)

        ax.axhline(0.0, color="0.3", lw=0.8, ls="--", alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=10)
        _style_ax(ax)

    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel(r"$\tau$")
    fig.suptitle(
        fr"Cumulant transients, $\mathrm{{AR}}={ar:g}$, $\alpha={alpha:.2f}$"
        "\n" r"shaded = production window",
        y=1.01
    )
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Fig 13 — Effective restitution coefficient α_eff (stub)
# ---------------------------------------------------------------------------

def fig13_alpha_eff(output_path, alpha_range=(0.60, 0.95)):
    """α_eff vs α — diagonal α_eff=α (sphere) only.

    Full spherocylinder curves require ζ*/ν from separate unscaled runs.
    """
    alpha = np.linspace(alpha_range[0], alpha_range[1], 200)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot(alpha, alpha, color="0.2", lw=1.4, ls="--",
            label=r"$\alpha_{\mathrm{eff}}=\alpha$ (spheres)")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\alpha_{\mathrm{eff}}$")
    ax.set_title(r"Effective restitution coefficient $\alpha_{\mathrm{eff}}(\alpha,\mathrm{AR})$")
    ax.text(0.98, 0.10,
            "Spherocylinder curves require\nζ*/ν from unscaled runs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="0.45", style="italic")
    ax.legend(fontsize=9)
    _style_ax(ax)
    fig.tight_layout()
    _save(fig, output_path)
