"""Fit and investigate the f_tr (translational dissipation share) distribution.

Loads ftr_data.txt for AR=2.0, r=1.00 at multiple alpha values from results2/results/,
fits a Laplace distribution (loc, scale) per alpha, saves the table to models/, and
produces three investigation figures:

  figures/ftr_distributions.png  -- per-alpha histogram + fitted Laplace PDF
  figures/ftr_comparison.png     -- raw distributions overlaid for all alpha
  figures/ftr_params_vs_alpha.png -- Laplace loc and scale trends vs alpha
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import laplace as laplace_dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import yaml
from src.preprocessing.ftr_distribution import (
    load_ftr_data, fit_ftr_laplace, build_ftr_table, save_ftr_table,
)

DISPLAY_CLIP = (-5.0, 5.0)
N_PDF_POINTS = 500


def plot_ftr_distributions(results, alpha_values, figures_dir):
    """Figure 1: per-alpha histogram + fitted Laplace PDF."""
    ncols = 5
    nrows = (len(alpha_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
    axes = np.array(axes).flatten()

    x_plot = np.linspace(*DISPLAY_CLIP, N_PDF_POINTS)

    for i, alpha in enumerate(alpha_values):
        ax = axes[i]
        res = results[alpha]
        f_tr, loc, scale = res['data'], res['loc'], res['scale']

        f_clip = f_tr[(f_tr >= DISPLAY_CLIP[0]) & (f_tr <= DISPLAY_CLIP[1])]
        ax.hist(f_clip, bins=80, density=True, color='steelblue', alpha=0.6, label='data')
        ax.plot(x_plot, laplace_dist.pdf(x_plot, loc, scale), 'r-', lw=1.8, label='Laplace')

        ax.set_title(f'α = {alpha:.2f}  loc={loc:.3f}  scale={scale:.3f}', fontsize=8)
        ax.set_xlabel(r'$f_\mathrm{tr}$', fontsize=9)
        ax.set_xlim(DISPLAY_CLIP)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=8)
    for j in range(len(alpha_values), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(r'$f_\mathrm{tr}$ distributions (Laplace fit): AR=2.0, $\theta=1.0$',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(figures_dir, 'ftr_distributions.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ftr_comparison(results, alpha_values, figures_dir):
    """Figure 2: raw distributions overlaid — visual collapse check."""
    cmap = plt.cm.plasma
    colors = [cmap(i / max(len(alpha_values) - 1, 1)) for i in range(len(alpha_values))]

    fig, ax = plt.subplots(figsize=(8, 5))
    x_plot = np.linspace(*DISPLAY_CLIP, N_PDF_POINTS)

    for color, alpha in zip(colors, alpha_values):
        res = results[alpha]
        f_tr, loc, scale = res['data'], res['loc'], res['scale']
        f_clip = f_tr[(f_tr >= DISPLAY_CLIP[0]) & (f_tr <= DISPLAY_CLIP[1])]
        ax.hist(f_clip, bins=80, density=True, histtype='step',
                color=color, lw=1.2, label=f'α={alpha:.2f}')
        ax.plot(x_plot, laplace_dist.pdf(x_plot, loc, scale), '--',
                color=color, lw=0.8, alpha=0.6)

    ax.set_xlabel(r'$f_\mathrm{tr}$', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xlim(DISPLAY_CLIP)
    ax.set_title(r'$f_\mathrm{tr}$ raw distributions: AR=2.0, $\theta=1.0$  '
                 r'(solid=data, dashed=Laplace fit)', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = os.path.join(figures_dir, 'ftr_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ftr_params_vs_alpha(results, alpha_values, figures_dir):
    """Figure 3: Laplace loc and scale vs alpha."""
    als    = np.array(alpha_values)
    locs   = np.array([results[a]['loc']   for a in alpha_values])
    scales = np.array([results[a]['scale'] for a in alpha_values])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(als, locs, 'o-', color='steelblue', lw=1.5, ms=7)
    ax1.axhline(np.mean(locs), ls='--', color='gray', lw=1,
                label=f'mean={np.mean(locs):.3f}')
    ax1.set_xlabel(r'$e$ (coeff. of restitution)', fontsize=11)
    ax1.set_ylabel(r'loc  (median of $f_\mathrm{tr}$)', fontsize=11)
    ax1.set_title('Laplace location', fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(als, scales, 'o-', color='tomato', lw=1.5, ms=7)
    ax2.set_xlabel(r'$e$ (coeff. of restitution)', fontsize=11)
    ax2.set_ylabel(r'scale  (spread of $f_\mathrm{tr}$)', fontsize=11)
    ax2.set_title('Laplace scale', fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(r'Fitted Laplace parameters vs $\alpha$: AR=2.0, $\theta=1.0$',
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(figures_dir, 'ftr_params_vs_alpha.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Fit Laplace f_tr distribution from CTC data"
    )
    parser.add_argument('--config', default='config/default.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ftr_cfg = config['preprocessing']['ftr']
    results_root   = ftr_cfg['results_root']
    alpha_values   = [float(a) for a in ftr_cfg['alpha_values']]
    r              = float(ftr_cfg['r'])
    AR             = float(ftr_cfg['AR'])
    ftr_params_file = ftr_cfg['ftr_params_file']
    figures_dir    = config['postprocessing']['figures_dir']
    model_dir      = config['preprocessing']['model_output_dir']

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Fitting f_tr Laplace distributions: AR={AR}, r={r}, "
          f"{len(alpha_values)} alpha values")

    results = {}
    table = {}
    for alpha in alpha_values:
        folder = f"alpha_{alpha:.3f}_r{r:.2f}_AR{AR:.1f}"
        path = os.path.join(results_root, folder, "ftr_data.txt")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping alpha={alpha:.3f}")
            continue
        print(f"  Loading alpha={alpha:.3f}...")
        f_tr = load_ftr_data(results_root, alpha, r, AR)
        loc, scale = fit_ftr_laplace(f_tr)
        print(f"    n={len(f_tr)}, loc={loc:.4f}, scale={scale:.4f}")
        results[alpha] = dict(data=f_tr, loc=loc, scale=scale)
        key = f"({alpha:.3f}, {AR:.1f})"
        table[key] = {"loc": loc, "scale": scale}

    fitted_alphas = [a for a in alpha_values if a in results]
    if not fitted_alphas:
        print("No data found. Check preprocessing.ftr.results_root in config.")
        return

    save_ftr_table(table, ftr_params_file)
    print(f"\nSaved f_tr table: {ftr_params_file}")

    print("\nGenerating plots...")
    plot_ftr_distributions(results, fitted_alphas, figures_dir)
    plot_ftr_comparison(results, fitted_alphas, figures_dir)
    plot_ftr_params_vs_alpha(results, fitted_alphas, figures_dir)
    print("Done.")


if __name__ == '__main__':
    main()
