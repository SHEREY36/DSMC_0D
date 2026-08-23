#!/usr/bin/env python3
"""Generate all paper figures for the HCS non-Gaussian campaign.

Usage:
    python run_hcs_ng_postprocessing.py [options]

Figures produced (in figures_paper/):
    fig01_theta_transient.png         Fig 1  — θ(τ) relaxation
    fig02_theta_star.png              Fig 2  — θ*(α) from LAMMPS table
    fig03_cooling_rate.png            Fig 3  — ζ*/ν (sphere baseline only)
    fig04_collision_frequency.png     Fig 4  — ν(AR)/ν_ref vs AR
    fig05_a2tr.png                    Fig 5  — a₂ᵗʳ vs α + Brey baseline
    fig06_a3tr.png                    Fig 6  — a₃ᵗʳ vs α
    fig07_speed_vdf_ratio.png         Fig 7  — φ_c/φ_{c,M} vs c
    fig08_a2rot.png                   Fig 8  — a₂ʳᵒᵗ vs α
    fig09_a11.png                     Fig 9  — a₁₁ vs α
    fig10_rot_speed_vdf_ratio.png     Fig 10 — φ_w/φ_{w,M} vs w
    fig11_coupling_vdf.png            Fig 11 — φ_{cw}(x) log-log
    fig12_cumulant_transient.png      Fig 12 — a₂ᵗʳ(τ), a₂ʳᵒᵗ(τ), a₁₁(τ)
    fig13_alpha_eff.png               Fig 13 — α_eff stub
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from src.postprocessing.hcs_ng_paper_plots import (
    fig1_theta_transient,
    fig2_theta_star,
    fig3_cooling_rate,
    fig4_collision_frequency,
    fig5_a2tr,
    fig6_a3tr,
    fig7_speed_ratio,
    fig8_a2rot,
    fig9_a11,
    fig10_rot_speed_ratio,
    fig11_coupling_vdf,
    fig12_cumulant_transient,
    fig13_alpha_eff,
    load_campaign_summaries,
    load_campaign_histograms,
)


AR_VALUES = [1.5, 2.0, 2.5, 3.0]
ALPHA_VALUES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# AR labels for theta_target_table files
AR_LABEL_MAP = {"1.5": "15", "2.0": "20", "2.5": "25", "3.0": "30"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/hcs_ng_long_window",
                   help="Campaign root directory (default: runs/hcs_ng_long_window)")
    p.add_argument("--figures-dir", default=None,
                   help="Output directory for figures (default: {root}/figures_paper)")
    p.add_argument("--models-dir", default="models",
                   help="Models directory with theta_target_table_AR*.json (default: models)")
    p.add_argument("--skip", default="",
                   help="Comma-separated figure numbers to skip, e.g. '3,13'")
    p.add_argument("--fig12-ar", type=float, default=2.0,
                   help="AR for representative case in Fig 12 (default: 2.0)")
    p.add_argument("--fig12-alpha", type=float, default=0.70,
                   help="alpha for representative case in Fig 12 (default: 0.70)")
    p.add_argument("--fig1-alpha", type=float, default=0.70,
                   help="Fixed alpha for Fig 1 panel (a) (default: 0.70)")
    p.add_argument("--fig1-ar", type=float, default=2.0,
                   help="Fixed AR for Fig 1 panel (b) (default: 2.0)")
    p.add_argument("--fig4-alpha", type=float, default=0.80,
                   help="Fixed alpha for Fig 4 collision frequency (default: 0.80)")
    p.add_argument("--theta-abs-max", type=float, default=2.0,
                   help="Max theta before divergence is declared (default: 2.0). "
                        "Samples with theta > this value are excluded from cumulant "
                        "averages and histogram seeds with many such samples are dropped.")
    p.add_argument("--min-valid-fraction", type=float, default=0.85,
                   help="Min fraction of valid (pre-divergence) samples a seed must "
                        "have for its histogram to be included (default: 0.85)")
    return p.parse_args()


def _should_run(skip_set, fig_num):
    return str(fig_num) not in skip_set


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Error: campaign root not found: {root}")

    figures_dir = Path(args.figures_dir) if args.figures_dir else root / "figures_paper"
    figures_dir.mkdir(parents=True, exist_ok=True)

    models_dir = args.models_dir
    skip = set(s.strip() for s in args.skip.split(",") if s.strip())

    print(f"Campaign root : {root}")
    print(f"Figures dir  : {figures_dir}")
    print(f"Models dir   : {models_dir}")
    if skip:
        print(f"Skipping figs: {', '.join(sorted(skip, key=int))}")
    print()

    # ------------------------------------------------------------------
    # Load campaign data (used by most figures)
    # ------------------------------------------------------------------
    theta_abs_max = args.theta_abs_max
    min_valid_fraction = args.min_valid_fraction
    print(f"Theta divergence cutoff : {theta_abs_max}")
    print(f"Min histogram valid frac: {min_valid_fraction}")
    print()

    print("Loading campaign summaries (theta-truncated cumulants)...")
    rows = load_campaign_summaries(str(root), ar_values=AR_VALUES,
                                   alpha_values=ALPHA_VALUES,
                                   theta_abs_max=theta_abs_max)
    print(f"  {len(rows)} (AR, alpha) cases loaded.")

    print("Loading campaign histograms (validity-filtered)...")
    hist_data = load_campaign_histograms(str(root), ar_values=AR_VALUES,
                                         alpha_values=ALPHA_VALUES,
                                         min_valid_fraction=min_valid_fraction,
                                         theta_abs_max=theta_abs_max)
    print(f"  {len(hist_data)} cases with histogram data.")
    print()

    # ------------------------------------------------------------------
    # Figure 1 — θ(τ) transient
    # ------------------------------------------------------------------
    if _should_run(skip, 1):
        print("Fig 1: θ(τ) transient...")
        fig1_theta_transient(
            str(root),
            figures_dir / "fig01_theta_transient.png",
            fixed_alpha=args.fig1_alpha,
            fixed_ar=args.fig1_ar,
            ar_sweep=tuple(AR_VALUES),
            alpha_sweep=(0.60, 0.70, 0.80, 0.90, 0.95),
        )

    # ------------------------------------------------------------------
    # Figure 2 — θ*(α) from LAMMPS
    # ------------------------------------------------------------------
    if _should_run(skip, 2):
        print("Fig 2: θ*(α) from LAMMPS table...")
        fig2_theta_star(
            models_dir,
            figures_dir / "fig02_theta_star.png",
            ar_values=tuple(AR_VALUES),
            ar_label_map=AR_LABEL_MAP,
        )

    # ------------------------------------------------------------------
    # Figure 3 — ζ*/ν (sphere baseline)
    # ------------------------------------------------------------------
    if _should_run(skip, 3):
        print("Fig 3: ζ*/ν cooling rate (sphere baseline stub)...")
        fig3_cooling_rate(figures_dir / "fig03_cooling_rate.png")

    # ------------------------------------------------------------------
    # Figure 4 — ν(AR)/ν_ref vs AR
    # ------------------------------------------------------------------
    if _should_run(skip, 4):
        print(f"Fig 4: ν(AR)/ν_ref vs AR at α={args.fig4_alpha:.2f}...")
        fig4_collision_frequency(
            rows,
            figures_dir / "fig04_collision_frequency.png",
            fixed_alpha=args.fig4_alpha,
        )

    # ------------------------------------------------------------------
    # Figure 5 — a₂ᵗʳ vs α
    # ------------------------------------------------------------------
    if _should_run(skip, 5):
        print("Fig 5: a₂ᵗʳ vs α + Brey baseline...")
        fig5_a2tr(rows, figures_dir / "fig05_a2tr.png",
                  ar_values=tuple(AR_VALUES))

    # ------------------------------------------------------------------
    # Figure 6 — a₃ᵗʳ vs α
    # ------------------------------------------------------------------
    if _should_run(skip, 6):
        print("Fig 6: a₃ᵗʳ vs α...")
        fig6_a3tr(rows, figures_dir / "fig06_a3tr.png",
                  ar_values=tuple(AR_VALUES))

    # ------------------------------------------------------------------
    # Figure 7 — φ_c/φ_{c,M} translational speed VDF
    # ------------------------------------------------------------------
    if _should_run(skip, 7):
        print("Fig 7: φ_c/φ_{c,M} translational speed ratio...")
        fig7_speed_ratio(
            hist_data, rows,
            figures_dir / "fig07_speed_vdf_ratio.png",
            ar_panels=(1.5, 2.0, 3.0),
            alpha_curves=(0.70, 0.90),
        )

    # ------------------------------------------------------------------
    # Figure 8 — a₂ʳᵒᵗ vs α
    # ------------------------------------------------------------------
    if _should_run(skip, 8):
        print("Fig 8: a₂ʳᵒᵗ vs α...")
        fig8_a2rot(rows, figures_dir / "fig08_a2rot.png",
                   ar_values=tuple(AR_VALUES))

    # ------------------------------------------------------------------
    # Figure 9 — a₁₁ vs α
    # ------------------------------------------------------------------
    if _should_run(skip, 9):
        print("Fig 9: a₁₁ vs α...")
        fig9_a11(rows, figures_dir / "fig09_a11.png",
                 ar_values=tuple(AR_VALUES))

    # ------------------------------------------------------------------
    # Figure 10 — φ_w/φ_{w,M} rotational speed VDF
    # ------------------------------------------------------------------
    if _should_run(skip, 10):
        print("Fig 10: φ_w/φ_{w,M} rotational speed ratio...")
        fig10_rot_speed_ratio(
            hist_data, rows,
            figures_dir / "fig10_rot_speed_vdf_ratio.png",
            ar_panels=(1.5, 2.0, 3.0),
            alpha_curves=(0.70, 0.90),
        )

    # ------------------------------------------------------------------
    # Figure 11 — φ_{cw}(x) coupling distribution
    # ------------------------------------------------------------------
    if _should_run(skip, 11):
        print("Fig 11: φ_{cw}(x) coupling distribution...")
        fig11_coupling_vdf(
            hist_data,
            figures_dir / "fig11_coupling_vdf.png",
            fixed_ar=2.0,
            fixed_alpha=0.70,
            ar_sweep=tuple(AR_VALUES),
            alpha_sweep=(0.60, 0.70, 0.80, 0.90),
        )

    # ------------------------------------------------------------------
    # Figure 12 — Cumulant transient a₂ᵗʳ(τ), a₂ʳᵒᵗ(τ), a₁₁(τ)
    # ------------------------------------------------------------------
    if _should_run(skip, 12):
        print(f"Fig 12: cumulant transients (AR={args.fig12_ar}, "
              f"α={args.fig12_alpha:.2f})...")
        fig12_cumulant_transient(
            str(root),
            figures_dir / "fig12_cumulant_transient.png",
            ar=args.fig12_ar,
            alpha=args.fig12_alpha,
            theta_abs_max=theta_abs_max,
        )

    # ------------------------------------------------------------------
    # Figure 13 — α_eff (stub)
    # ------------------------------------------------------------------
    if _should_run(skip, 13):
        print("Fig 13: α_eff (diagonal stub)...")
        fig13_alpha_eff(figures_dir / "fig13_alpha_eff.png")

    print()
    print(f"Done. Figures written to {figures_dir}/")


if __name__ == "__main__":
    main()
