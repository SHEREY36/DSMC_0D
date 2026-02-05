"""Orchestrator for fitting all collision sub-models.

Can also be used standalone for visualization and inspection of fitted models.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from .data_loader import load_all_data, load_chi_data
from .gmm_energy import (
    preprocess_data, train_gmm, find_best_gmm_bic,
    export_conditional_gmm_npz
)
from .scattering_angle import (
    fit_scattering_models, p_chi_AR_alpha
)
from .dissipation import (
    build_gamma_max_table, build_one_hit_table, save_table
)

from sklearn.preprocessing import StandardScaler


def fit_gmm(config, output_dir, plot=False):
    """Fit the GMM energy redistribution model.

    Loads CTC data, preprocesses, trains GMM, and saves the model + scaler.
    """
    gmm_cfg = config['preprocessing']['gmm']
    base_dir = gmm_cfg['base_dir']
    n_components = gmm_cfg['n_components']
    max_bic = gmm_cfg['max_components_bic']
    r_start = gmm_cfg.get('r_range_start', 1)
    r_end = gmm_cfg.get('r_range_end', 13)

    print("Loading GMM training data...")
    bigZ = load_all_data(base_dir, r_range=range(r_start, r_end))
    print(f"  Loaded {bigZ.shape[0]} samples from {r_end - r_start} folders")

    bigZ_proc = preprocess_data(bigZ)
    scaler = StandardScaler().fit(bigZ_proc)
    bigZ_scaled = scaler.transform(bigZ_proc)

    if plot:
        print("Running BIC model selection...")
        best_n, bics = find_best_gmm_bic(bigZ_scaled, max_components=max_bic)
        fig, ax = plt.subplots()
        ax.plot(range(1, max_bic + 1), bics, marker='^')
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("BIC Score")
        ax.set_title("GMM BIC Model Selection")
        ax.grid(True)
        fig.savefig(os.path.join(output_dir, "gmm_bic.png"), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print(f"  BIC plot saved. Best n={best_n}")

    print(f"Training GMM with {n_components} components...")
    gmm_model = train_gmm(bigZ_scaled, n_components=n_components)

    # Export conditional GMM as .npz (pre-computed inverses + Cholesky)
    ar_label = os.path.basename(base_dir)
    npz_path = os.path.join(output_dir, f"gmm_cond_{ar_label}.npz")
    export_conditional_gmm_npz(gmm_model, scaler, npz_path)

    return gmm_model, scaler


def fit_scattering(config, output_dir, plot=False):
    """Fit scattering angle polynomial models.

    Fits P_elastic(chi, AR) and delta_p(chi, AR, alpha) from chi.txt data.
    """
    scat_cfg = config['preprocessing']['scattering']
    ctc_dir = config['preprocessing']['ctc_data_dir']
    root_dir = os.path.join(ctc_dir, "Alpha")

    alpha_dirs = scat_cfg['alpha_dirs']
    ar_dirs = scat_cfg['ar_dirs']
    K = scat_cfg['polynomial_K']
    M = scat_cfg['polynomial_M']
    N = scat_cfg['polynomial_N']
    beta = scat_cfg['beta']

    print("Fitting scattering angle models...")
    a_elastic, a_inelastic, M_out, N_out, K_out, beta_out = \
        fit_scattering_models(root_dir, alpha_dirs, ar_dirs, K=K, M=M, N=N,
                              beta=beta)

    npz_path = os.path.join(output_dir, "scattering_coeffs.npz")
    np.savez(npz_path,
             a_elastic=a_elastic,
             a_inelastic=a_inelastic,
             M=M_out, N=N_out, K=K_out, beta=beta_out)
    print(f"  Scattering coefficients saved to {npz_path}")

    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        chi_test = np.linspace(0, 1, 1000)
        for i, ar in enumerate(ar_dirs[:6]):
            ax = axes.flat[i]
            AR_val = ar / 10.0
            for alpha in [0.7, 0.85, 0.95, 1.0]:
                p_vals = p_chi_AR_alpha(chi_test, AR_val, alpha, a_elastic,
                                        a_inelastic, M_out, N_out, K_out,
                                        beta_out)
                ax.plot(chi_test, p_vals, label=f"alpha={alpha}")
            ax.set_title(f"AR={AR_val}")
            ax.set_xlabel("chi/pi")
            ax.set_ylabel("p(chi)")
            ax.legend(fontsize=8)
            ax.grid(True)
        fig.suptitle("Scattering Angle PDFs")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "scattering_fits.png"), dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print("  Scattering fit plots saved.")

    return a_elastic, a_inelastic, M_out, N_out, K_out, beta_out


def build_lookup_tables(config, output_dir):
    """Build and save gamma_max and one-hit probability lookup tables."""
    ctc_dir = config['preprocessing']['ctc_data_dir']
    root_dir = os.path.join(ctc_dir, "Alpha")
    scat_cfg = config['preprocessing']['scattering']
    alpha_dirs = scat_cfg['alpha_dirs']
    ar_dirs = scat_cfg['ar_dirs']

    print("Building gamma_max lookup table...")
    gamma_table = build_gamma_max_table(root_dir, alpha_dirs, ar_dirs)
    gamma_path = os.path.join(output_dir, "gamma_max_table.json")
    save_table(gamma_table, gamma_path)
    print(f"  {len(gamma_table)} entries saved to {gamma_path}")

    print("Building one-hit probability lookup table...")
    onehit_table = build_one_hit_table(root_dir, alpha_dirs, ar_dirs)
    onehit_path = os.path.join(output_dir, "one_hit_table.json")
    save_table(onehit_table, onehit_path)
    print(f"  {len(onehit_table)} entries saved to {onehit_path}")

    return gamma_table, onehit_table


def run_all(config, plot=False):
    """Run all pre-processing steps: fit models and save artifacts."""
    output_dir = config['preprocessing']['model_output_dir']
    os.makedirs(output_dir, exist_ok=True)

    gmm_model, scaler = fit_gmm(config, output_dir, plot=plot)
    scat_results = fit_scattering(config, output_dir, plot=plot)
    gamma_table, onehit_table = build_lookup_tables(config, output_dir)

    print("\nAll models fitted and saved successfully.")
    return {
        'gmm_model': gmm_model,
        'scaler': scaler,
        'scattering': scat_results,
        'gamma_max_table': gamma_table,
        'one_hit_table': onehit_table,
    }
