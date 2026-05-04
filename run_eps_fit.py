#!/usr/bin/env python3
"""Fit and save the azimuthal eps von Mises model p(eps | mu, alpha, AR).

Reads 10-column CTC chi.txt data (new format with ghat_pre + ghat_post columns)
from Coll_Models/results, fits von Mises concentration kappa as a polynomial
in (AR, phi(alpha), mu), and saves to models/eps_azimuth_coeffs.npz.

If chi.txt files still have only 4 columns (old format), the script prints a
warning and exits — re-run the CTC Fortran code first.

Usage:
    python run_eps_fit.py [--config config/default.yaml]
"""
import argparse
import yaml

from src.preprocessing.fit_eps_model import fit_eps_model, save_eps_model


def parse_args():
    p = argparse.ArgumentParser(description="Fit azimuthal eps von Mises model")
    p.add_argument("--config", default="config/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg.get("preprocessing", {}).get("mu_chi", {})
    source_root = mc.get("source_root", "../Coll_Models/results")

    eps_cfg = cfg.get("preprocessing", {}).get("eps_azimuth", {})
    output_file = eps_cfg.get("output_file", "models/eps_azimuth_coeffs.npz")
    n_mu_bins   = int(eps_cfg.get("n_mu_bins", mc.get("n_mu_bins", 20)))
    M = int(eps_cfg.get("polynomial_M", mc.get("polynomial_M", 2)))
    N = int(eps_cfg.get("polynomial_N", mc.get("polynomial_N", 2)))
    J = int(eps_cfg.get("polynomial_J", mc.get("polynomial_J", 3)))
    beta_exp = float(eps_cfg.get("beta_exp", mc.get("beta_exp", 0.5)))
    max_kappa = float(eps_cfg.get("max_kappa", 20.0))

    print(f"Fitting eps azimuth model from: {source_root}")
    print(f"  n_mu_bins={n_mu_bins}, M={M}, N={N}, J={J}, beta_exp={beta_exp}, max_kappa={max_kappa}")

    c_kappa, M_fit, N_fit, J_fit, beta_exp_fit, diagnostics = fit_eps_model(
        source_root=source_root,
        n_mu_bins=n_mu_bins,
        M=M, N=N, J=J,
        beta_exp=beta_exp,
        max_kappa=max_kappa,
    )

    print("\nPer-case isotropy diagnostics (R_bar ≈ 0 → isotropic):")
    for d in diagnostics:
        print(f"  alpha={d['alpha']:.2f} AR={d['AR']:.1f}: "
              f"n={d['n']}, R_bar={d['R_bar']:.4f}, kappa={d['kappa']:.4f}")

    mean_kappa = sum(d['kappa'] for d in diagnostics) / max(len(diagnostics), 1)
    print(f"\nMean kappa across all cases: {mean_kappa:.4f}")
    if mean_kappa < 0.5:
        print("  → Distribution is approximately isotropic (kappa ≈ 0).")
        print("    Set simulation.use_isotropic_eps: true in config to use uniform eps.")
    else:
        print(f"  → Non-trivial concentration detected. Fitted model saved.")

    save_eps_model(output_file, c_kappa, M_fit, N_fit, J_fit, beta_exp_fit)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
