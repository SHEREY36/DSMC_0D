#!/usr/bin/env python3
"""Aggregate and plot HCS non-Gaussian speed/energy diagnostics."""

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from src.postprocessing.non_gaussian import (
    aggregate_histograms,
    aggregate_non_gaussian_summaries,
    exponential_energy_pdf,
    histogram_ratio_to_reference,
    maxwell_energy_coupling_pdf,
    maxwell_energy_pdf,
    maxwell_speed_pdf,
    rayleigh_rot_speed_pdf,
)


AR_LABELS = {
    "AR100": 1.0,
    "AR110": 1.1,
    "AR125": 1.25,
    "AR150": 1.5,
    "AR200": 2.0,
    "AR250": 2.5,
    "AR300": 3.0,
}


def _alpha_from_dir(path):
    return int(path.name.split("_", 1)[1]) / 100.0


def discover_cases(root):
    cases = []
    for ar_dir in sorted(root.glob("AR*")):
        if ar_dir.name not in AR_LABELS:
            continue
        for alpha_dir in sorted(ar_dir.glob("alpha_*")):
            results_dir = alpha_dir / "results"
            if results_dir.is_dir():
                cases.append((AR_LABELS[ar_dir.name], _alpha_from_dir(alpha_dir), results_dir))
    return cases


def aggregate_campaign(root):
    rows = []
    histograms = {}
    for AR, alpha, results_dir in discover_cases(root):
        result = aggregate_non_gaussian_summaries(results_dir)
        row = {
            "AR": AR,
            "alpha": alpha,
            "n_realizations": result["n_realizations"],
            "complete_realizations": result["complete_realizations"],
            "expected_output_samples": result["expected_output_samples"],
            "n_particle_samples": result["n_particle_samples"],
            "collision_frequency": result["collision_frequency"]["mean"],
            "collision_frequency_stderr": result["collision_frequency"]["stderr"],
        }
        for group in ("cumulants", "moments"):
            for name, stats in result[group].items():
                row[name] = stats["mean"]
                row[f"{name}_stderr"] = stats["stderr"]
        rows.append(row)
        if result["n_particle_samples"] > 0:
            case_hists = {}
            histogram_specs = {
                "speed": ("_ng_hist_speed.txt", maxwell_speed_pdf),
                "rot_speed": ("_ng_hist_rot_speed.txt", rayleigh_rot_speed_pdf),
                "energy_tr": ("_ng_hist_energy_tr.txt", maxwell_energy_pdf),
                "energy_rot": ("_ng_hist_energy_rot.txt", exponential_energy_pdf),
                "energy_coupling": (
                    "_ng_hist_energy_coupling.txt",
                    maxwell_energy_coupling_pdf,
                ),
            }
            for name, (suffix, reference) in histogram_specs.items():
                try:
                    hist = aggregate_histograms(results_dir, suffix)
                except FileNotFoundError:
                    continue
                case_hists[name] = histogram_ratio_to_reference(hist, reference)
            histograms[(AR, alpha)] = case_hists
    rows.sort(key=lambda item: (item["AR"], item["alpha"]))
    return rows, histograms


def write_summary_csv(rows, path):
    fieldnames = [
        "AR", "alpha", "n_realizations", "complete_realizations",
        "expected_output_samples", "n_particle_samples",
        "collision_frequency", "collision_frequency_stderr",
        "a2_tr", "a2_tr_stderr", "a3_tr", "a3_tr_stderr",
        "a2_rot", "a2_rot_stderr", "a11", "a11_stderr",
        "c2", "c2_stderr", "c4", "c4_stderr", "c6", "c6_stderr",
        "w2", "w2_stderr", "w4", "w4_stderr", "c2w2", "c2w2_stderr",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, np.nan) for name in fieldnames})


def plot_sample_coverage(rows, path):
    ars = sorted({row["AR"] for row in rows})
    alphas = sorted({row["alpha"] for row in rows})
    coverage = np.full((len(ars), len(alphas)), np.nan)
    lookup = {(row["AR"], row["alpha"]): row for row in rows}
    for i, AR in enumerate(ars):
        for j, alpha in enumerate(alphas):
            row = lookup[(AR, alpha)]
            coverage[i, j] = (
                row.get("complete_realizations", 0)
                / max(row.get("n_realizations", 1), 1)
            )

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(coverage, aspect="auto", cmap="Greens", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(alphas)), [f"{a:.2f}" for a in alphas])
    ax.set_yticks(np.arange(len(ars)), [f"{ar:g}" for ar in ars])
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("AR")
    ax.set_title(r"Non-Gaussian sampling completeness")
    for i, AR in enumerate(ars):
        for j, alpha in enumerate(alphas):
            row = lookup[(AR, alpha)]
            text = f"{row.get('complete_realizations', 0)}/{row['n_realizations']}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(["incomplete", "complete"])
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.axhline(0.0 if "cumulant" in ylabel.lower() else 1.0, color="0.2", lw=0.8, alpha=0.45)


def plot_cumulants(rows, path):
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True)
    specs = [
        ("a2_tr", r"$a_{2,\mathrm{tr}}$"),
        ("a3_tr", r"$a_{3,\mathrm{tr}}$"),
        ("a2_rot", r"$a_{2,\mathrm{rot}}$"),
        ("a11", r"$a_{11}$"),
    ]
    ars = sorted({row["AR"] for row in rows})
    cmap = plt.get_cmap("viridis")
    colors = {AR: cmap(i / max(len(ars) - 1, 1)) for i, AR in enumerate(ars)}
    for ax, (key, ylabel) in zip(axes.flat, specs):
        for AR in ars:
            sub = [row for row in rows if np.isclose(row["AR"], AR)]
            x = np.array([row["alpha"] for row in sub])
            y = np.array([row.get(key, np.nan) for row in sub], dtype=float)
            err = np.array([row.get(f"{key}_stderr", np.nan) for row in sub], dtype=float)
            mask = np.isfinite(y)
            if not np.any(mask):
                continue
            ax.errorbar(
                x[mask], y[mask], yerr=err[mask], marker="o", lw=1.6,
                ms=4.5, capsize=2.5, color=colors[AR], label=f"AR={AR:g}"
            )
        _style_axes(ax, r"$\alpha$", f"{ylabel} cumulant")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("HCS non-Gaussian cumulants from speed/energy moments", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_sanity_moments(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    specs = [("c2", r"$\langle c^2\rangle$", 1.5), ("w2", r"$\langle w^2\rangle$", 1.0)]
    ars = sorted({row["AR"] for row in rows})
    cmap = plt.get_cmap("viridis")
    colors = {AR: cmap(i / max(len(ars) - 1, 1)) for i, AR in enumerate(ars)}
    for ax, (key, ylabel, target) in zip(axes, specs):
        for AR in ars:
            sub = [row for row in rows if np.isclose(row["AR"], AR)]
            x = np.array([row["alpha"] for row in sub])
            y = np.array([row.get(key, np.nan) for row in sub], dtype=float)
            mask = np.isfinite(y)
            if np.any(mask):
                ax.plot(x[mask], y[mask], marker="o", lw=1.5, ms=4, color=colors[AR], label=f"AR={AR:g}")
        ax.axhline(target, color="0.15", lw=1.0, alpha=0.6)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle("Reduced-moment normalization checks", y=0.98)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _finite_ratio(hist, x_max=None):
    x = hist["centers"]
    ratio = hist["ratio"]
    err = hist["density_stderr"] / np.where(hist["reference"] > 0.0, hist["reference"], np.nan)
    mask = np.isfinite(ratio)
    if x_max is not None:
        mask &= x <= x_max
    return x[mask], ratio[mask], err[mask]


def plot_distribution_ratios(histograms, output_dir):
    alphas = sorted({alpha for _, alpha in histograms})
    ars = sorted({AR for AR, _ in histograms})
    cmap = plt.get_cmap("viridis")
    colors = {AR: cmap(i / max(len(ars) - 1, 1)) for i, AR in enumerate(ars)}
    specs = [
        ("speed", r"$c$", r"$\phi(c)/\phi_M(c)$", 3.4, True),
        ("energy_tr", r"$\epsilon_t=c^2$", r"$\phi(\epsilon_t)/\phi_M(\epsilon_t)$", 8.0, True),
        ("rot_speed", r"$w$", r"$\phi(w)/\phi_M(w)$", 3.2, False),
        ("energy_rot", r"$\epsilon_r=w^2$", r"$\phi(\epsilon_r)/e^{-\epsilon_r}$", 8.0, False),
        ("energy_coupling", r"$x=\epsilon_t\epsilon_r$", r"$\phi(x)/\phi_M(x)$", 16.0, False),
    ]
    for alpha in alphas:
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.2))
        for ax, (name, xlabel, ylabel, xmax, include_spheres) in zip(axes.flat, specs):
            for AR in ars:
                if not include_spheres and np.isclose(AR, 1.0):
                    continue
                hist = histograms.get((AR, alpha), {}).get(name)
                if hist is None:
                    continue
                x, ratio, err = _finite_ratio(hist, x_max=xmax)
                if x.size == 0:
                    continue
                ax.plot(x, ratio, lw=1.6, color=colors[AR], label=f"AR={AR:g}")
                ax.fill_between(
                    x, ratio - err, ratio + err, color=colors[AR],
                    alpha=0.10, linewidth=0
                )
            ax.axhline(1.0, color="0.15", lw=0.9, alpha=0.7)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0.0, 2.4)
            ax.grid(True, alpha=0.25)
        for ax in axes.flat[len(specs):]:
            ax.axis("off")
        axes[0, 0].legend(ncol=2, fontsize=8)
        fig.suptitle(fr"HCS distribution ratios at $\alpha={alpha:.2f}$", y=0.98)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"hcs_ng_distribution_ratios_alpha_{int(round(alpha * 100)):03d}.png",
            dpi=220
        )
        plt.close(fig)


def plot_density_by_ar(histograms, output_dir):
    ars = sorted({AR for AR, _ in histograms})
    cmap = plt.get_cmap("plasma")
    for AR in ars:
        alphas = sorted(alpha for ar, alpha in histograms if np.isclose(ar, AR))
        if not alphas:
            continue
        colors = {
            alpha: cmap(i / max(len(alphas) - 1, 1))
            for i, alpha in enumerate(alphas)
        }

        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))

        ax = axes[0]
        c_ref = np.linspace(1.0e-3, 4.5, 600)
        ax.semilogy(c_ref, maxwell_speed_pdf(c_ref), color="black", lw=1.4, label="Maxwellian")
        for alpha in alphas:
            hist = histograms.get((AR, alpha), {}).get("speed")
            if hist is None:
                continue
            x = hist["centers"]
            y = hist["density_mean"]
            mask = (x > 0.0) & (x <= 4.5) & (y > 0.0)
            if np.any(mask):
                ax.semilogy(
                    x[mask], y[mask], lw=1.4, color=colors[alpha],
                    label=fr"$\alpha={alpha:.2f}$"
                )
        ax.set_xlabel(r"$c=|\mathbf{v}|/\sqrt{2T_{\mathrm{tr}}/m}$")
        ax.set_ylabel(r"$\phi(c)$")
        ax.set_title("Translational speed")
        ax.set_xlim(0.0, 4.5)
        ax.set_ylim(1.0e-6, 2.0)
        ax.grid(True, which="both", alpha=0.25)

        ax = axes[1]
        if np.isclose(AR, 1.0):
            for empty_ax in axes[1:]:
                empty_ax.axis("off")
                empty_ax.text(
                    0.5, 0.5, "No rotational-energy mode for spheres",
                    ha="center", va="center", transform=empty_ax.transAxes
                )
        else:
            e_ref = np.linspace(1.0e-3, 10.0, 600)
            ax.semilogy(e_ref, exponential_energy_pdf(e_ref), color="black", lw=1.4, label=r"$e^{-\epsilon_r}$")
            for alpha in alphas:
                hist = histograms.get((AR, alpha), {}).get("energy_rot")
                if hist is None:
                    continue
                x = hist["centers"]
                y = hist["density_mean"]
                mask = (x > 0.0) & (x <= 10.0) & (y > 0.0)
                if np.any(mask):
                    ax.semilogy(
                        x[mask], y[mask], lw=1.4, color=colors[alpha],
                        label=fr"$\alpha={alpha:.2f}$"
                    )
            ax.set_xlabel(r"$\epsilon_r=E_r/T_{\mathrm{rot}}$")
            ax.set_ylabel(r"$\phi(\epsilon_r)$")
            ax.set_title("Rotational energy")
            ax.set_xlim(0.0, 10.0)
            ax.set_ylim(1.0e-6, 2.0)
            ax.grid(True, which="both", alpha=0.25)

            ax = axes[2]
            x_ref = np.linspace(1.0e-3, 16.0, 700)
            ax.semilogy(
                x_ref, maxwell_energy_coupling_pdf(x_ref),
                color="black", lw=1.4, label=r"$2e^{-2\sqrt{x}}$"
            )
            for alpha in alphas:
                hist = histograms.get((AR, alpha), {}).get("energy_coupling")
                if hist is None:
                    continue
                x = hist["centers"]
                y = hist["density_mean"]
                mask = (x > 0.0) & (x <= 16.0) & (y > 0.0)
                if np.any(mask):
                    ax.semilogy(
                        x[mask], y[mask], lw=1.4, color=colors[alpha],
                        label=fr"$\alpha={alpha:.2f}$"
                    )
            ax.set_xlabel(r"$x=\epsilon_t\epsilon_r$")
            ax.set_ylabel(r"$\phi(x)$")
            ax.set_title("Energy coupling")
            ax.set_xlim(0.0, 16.0)
            ax.set_ylim(1.0e-6, 2.0)
            ax.grid(True, which="both", alpha=0.25)

        handles = []
        labels = []
        for legend_ax in axes:
            h_ax, l_ax = legend_ax.get_legend_handles_labels()
            for handle, label in zip(h_ax, l_ax):
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
        fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
        fig.suptitle(fr"HCS scalar distributions for AR={AR:g}", y=0.97)
        fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
        fig.savefig(output_dir / f"hcs_ng_density_AR{int(round(AR * 100)):03d}.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/hcs_ng_long_window")
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    figures_dir = Path(args.figures_dir) if args.figures_dir else root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows, histograms = aggregate_campaign(root)
    if not rows:
        raise RuntimeError(f"No HCS non-Gaussian cases found under {root}")

    write_summary_csv(rows, root / "hcs_ng_summary.csv")
    plot_sample_coverage(rows, figures_dir / "hcs_ng_sampling_coverage.png")
    plot_cumulants(rows, figures_dir / "hcs_ng_cumulants_vs_alpha.png")
    plot_sanity_moments(rows, figures_dir / "hcs_ng_reduced_moment_checks.png")
    plot_distribution_ratios(histograms, figures_dir)
    plot_density_by_ar(histograms, figures_dir)

    print(f"Aggregated {len(rows)} cases into {root / 'hcs_ng_summary.csv'}")
    print(f"Wrote figures to {figures_dir}")


if __name__ == "__main__":
    main()
