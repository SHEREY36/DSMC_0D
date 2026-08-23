#!/usr/bin/env python3
"""Fit and save the conditional chi model p(chi | mu, alpha, AR).

Reads CTC data from Coll_Models/results (alpha_*_r1.00_AR*/chi.txt),
fits a conditional Beta distribution with polynomial parameters, and
saves the result to models/mu_chi_beta_coeffs.npz.

Usage:
    python run_mu_chi_fit.py [--config config/default.yaml]
"""
import argparse
import yaml

from src.preprocessing.mu_chi_model import fit_mu_chi_model, save_mu_chi_model


def parse_args():
    p = argparse.ArgumentParser(description="Fit conditional chi Beta model")
    p.add_argument("--config", default="config/default.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    mc = cfg.get("preprocessing", {}).get("mu_chi", {})
    source_root = mc.get("source_root", "../Coll_Models/results")
    output_file = mc.get("output_file", "models/mu_chi_beta_coeffs.npz")
    n_mu_bins = int(mc.get("n_mu_bins", 20))
    M = int(mc.get("polynomial_M", 2))
    N = int(mc.get("polynomial_N", 2))
    J = int(mc.get("polynomial_J", 3))
    beta_exp = float(mc.get("beta_exp", 0.5))

    print(f"Fitting conditional chi model from: {source_root}")
    print(f"  n_mu_bins={n_mu_bins}, M={M}, N={N}, J={J}, beta_exp={beta_exp}")

    c_a, c_b, M_fit, N_fit, J_fit, beta_fit = fit_mu_chi_model(
        source_root,
        n_mu_bins=n_mu_bins,
        M=M, N=N, J=J,
        beta_exp=beta_exp,
        verbose=True,
    )

    save_mu_chi_model(output_file, c_a, c_b, M_fit, N_fit, J_fit, beta_fit)
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
