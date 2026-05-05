#!/usr/bin/env python3
"""Compute CTC-homogenized rank-2 stress-transport weights.

This diagnostic compares the rank-2 angular relaxation efficiency measured
from CTC endpoint data with the current DSMC hard-sphere chi(mu, alpha)
angular operator.  It intentionally does not modify the DSMC pipeline.

The CTC incoming relative velocity is assumed to be fixed as ghat0=(-1,0,0),
so c = ghat0 . ghat_post = -ghat_post_x.

Usage
-----
    python diagnose_stress_transport_weight.py
    python diagnose_stress_transport_weight.py --ar 2.0
    python diagnose_stress_transport_weight.py --output-dir runs/stress_transport
"""
import argparse
import csv
import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_CTC_ROOT = "/home/muhammed/Documents/Thesis/Coll_Models/results"
CASE_RE = re.compile(r"alpha_([0-9]+(?:\.[0-9]+)?)_r1\.00_AR([0-9]+(?:\.[0-9]+)?)$")


@dataclass
class StressTransportRow:
    alpha: float
    AR: float
    n_events: int
    lambda2_ctc: float
    beta_eta_ctc: float
    lambda2_dsmc: float
    beta_eta_dsmc: float
    w_eta: float
    lambda2_ctc_q: float
    beta_eta_ctc_q: float
    w_eta_q: float
    mean_q: float
    min_q: float
    max_q: float
    max_ghat_post_norm_error: float
    row_count_mismatch: bool
    w_eta_gt_one: bool
    w_eta_q_gt_one: bool


def p2(c):
    """Second Legendre polynomial P2(c)."""
    return 0.5 * (3.0 * c * c - 1.0)


def cos_chi_hs(mu, alpha):
    """cos(chi_hs) for accepted hard-sphere geometry."""
    mu = np.asarray(mu, dtype=float)
    denom = np.sqrt(np.maximum(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    out = (1.0 - (1.0 + alpha) * mu * mu) / denom
    return np.clip(out, -1.0, 1.0)


def lambda2_dsmc_hs(alpha, n_grid=200001):
    """Compute <P2(cos chi_hs)> under accepted DSMC mu density 2*mu."""
    mu = np.linspace(0.0, 1.0, int(n_grid))
    weights = 2.0 * mu
    numerator = np.trapezoid(weights * p2(cos_chi_hs(mu, alpha)), mu)
    denominator = np.trapezoid(weights, mu)
    if denominator <= 0.0:
        raise ValueError("Invalid accepted-mu quadrature denominator")
    return float(numerator / denominator)


def parse_case_dir(case_dir):
    """Return (alpha, AR) for a valid CTC case directory, otherwise None."""
    match = CASE_RE.search(os.path.basename(str(case_dir)))
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def load_case_arrays(case_dir):
    """Load and validate one CTC case.

    Returns (chi, ef, row_count_mismatch).  The arrays are truncated to the
    shared row count if the two output files differ in length.
    """
    chi_path = os.path.join(case_dir, "chi.txt")
    ef_path = os.path.join(case_dir, "Ef.txt")
    if not os.path.exists(chi_path):
        raise FileNotFoundError(f"Missing chi.txt: {chi_path}")
    if not os.path.exists(ef_path):
        raise FileNotFoundError(f"Missing Ef.txt: {ef_path}")

    chi = np.loadtxt(chi_path)
    ef = np.loadtxt(ef_path)
    if chi.ndim == 1:
        chi = chi.reshape(1, -1)
    if ef.ndim == 1:
        ef = ef.reshape(1, -1)
    if chi.shape[1] < 10:
        raise ValueError(f"{chi_path} must have at least 10 columns, got {chi.shape[1]}")
    if ef.shape[1] < 6:
        raise ValueError(f"{ef_path} must have at least 6 columns, got {ef.shape[1]}")

    row_count_mismatch = chi.shape[0] != ef.shape[0]
    n_rows = min(chi.shape[0], ef.shape[0])
    if n_rows <= 0:
        raise ValueError(f"No rows available in {case_dir}")
    return chi[:n_rows], ef[:n_rows], row_count_mismatch


def compute_case(case_dir, norm_tol=5.0e-3, n_grid=200001):
    """Compute one row of stress-transport diagnostics for a CTC case."""
    parsed = parse_case_dir(case_dir)
    if parsed is None:
        raise ValueError(f"Cannot parse alpha/AR from case directory: {case_dir}")
    alpha, AR = parsed
    chi, ef, row_count_mismatch = load_case_arrays(case_dir)

    ghat_post = np.asarray(chi[:, 7:10], dtype=float)
    ef = np.asarray(ef, dtype=float)
    finite_mask = np.all(np.isfinite(ghat_post), axis=1)
    finite_mask &= np.all(np.isfinite(ef[:, :6]), axis=1)
    finite_mask &= ef[:, 0] > 0.0
    finite_mask &= ef[:, 3] > 0.0
    if not np.any(finite_mask):
        raise ValueError(f"No finite valid events in {case_dir}")

    ghat_post = ghat_post[finite_mask]
    ef = ef[finite_mask]
    norms = np.linalg.norm(ghat_post, axis=1)
    max_norm_error = float(np.max(np.abs(norms - 1.0)))
    norm_mask = norms > 1.0e-12
    if not np.all(norm_mask):
        ghat_post = ghat_post[norm_mask]
        ef = ef[norm_mask]
        norms = norms[norm_mask]
    if ghat_post.shape[0] == 0:
        raise ValueError(f"No nonzero ghat_post rows in {case_dir}")
    if max_norm_error > norm_tol:
        raise ValueError(
            f"{case_dir}: max |norm(ghat_post)-1|={max_norm_error:.3e} "
            f"exceeds --norm-tol={norm_tol:.3e}"
        )

    ghat_post = ghat_post / norms[:, None]
    c = -ghat_post[:, 0]
    p2_vals = p2(c)
    lambda2_ctc = float(np.mean(p2_vals))
    beta_eta_ctc = 1.0 - lambda2_ctc

    q = ef[:, 3] / ef[:, 0]
    q_sum = float(np.sum(q))
    if q_sum <= 0.0:
        raise ValueError(f"Non-positive q denominator in {case_dir}")
    lambda2_ctc_q = float(np.sum(q * p2_vals) / q_sum)
    beta_eta_ctc_q = 1.0 - lambda2_ctc_q

    lambda2_dsmc = lambda2_dsmc_hs(alpha, n_grid=n_grid)
    beta_eta_dsmc = 1.0 - lambda2_dsmc
    if beta_eta_dsmc <= 0.0:
        raise ValueError(f"Non-positive DSMC beta_eta for alpha={alpha:g}")

    w_eta = beta_eta_ctc / beta_eta_dsmc
    w_eta_q = beta_eta_ctc_q / beta_eta_dsmc

    return StressTransportRow(
        alpha=alpha,
        AR=AR,
        n_events=int(ghat_post.shape[0]),
        lambda2_ctc=lambda2_ctc,
        beta_eta_ctc=beta_eta_ctc,
        lambda2_dsmc=lambda2_dsmc,
        beta_eta_dsmc=beta_eta_dsmc,
        w_eta=float(w_eta),
        lambda2_ctc_q=lambda2_ctc_q,
        beta_eta_ctc_q=beta_eta_ctc_q,
        w_eta_q=float(w_eta_q),
        mean_q=float(np.mean(q)),
        min_q=float(np.min(q)),
        max_q=float(np.max(q)),
        max_ghat_post_norm_error=max_norm_error,
        row_count_mismatch=bool(row_count_mismatch),
        w_eta_gt_one=bool(w_eta > 1.0),
        w_eta_q_gt_one=bool(w_eta_q > 1.0),
    )


def discover_cases(ctc_root):
    """Find all valid alpha_*_r1.00_AR* case directories."""
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(ctc_root, "alpha_*_r1.00_AR*"))):
        if parse_case_dir(case_dir) is not None:
            cases.append(case_dir)
    return cases


def filter_rows(rows, alpha=None, AR=None):
    """Filter computed rows by optional alpha and AR."""
    out = []
    for row in rows:
        if alpha is not None and not np.isclose(row.alpha, alpha, atol=5.0e-8):
            continue
        if AR is not None and not np.isclose(row.AR, AR, atol=5.0e-8):
            continue
        out.append(row)
    return out


def write_outputs(rows, output_dir):
    """Write CSV and JSON diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stress_transport_weights.csv"
    json_path = output_dir / "stress_transport_weights.json"

    fieldnames = list(StressTransportRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    payload = {
        "assumptions": {
            "ctc_root_weight": "W_n = 1",
            "ghat0": [-1.0, 0.0, 0.0],
            "primary_weight": "angular-only q=1 w_eta",
            "q_weighted_columns": "diagnostic only",
            "dsmc_reference": "accepted mu density 2*mu with active chi_hs(mu, alpha)",
        },
        "rows": [asdict(row) for row in rows],
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def print_summary(rows):
    """Print a compact console summary grouped by AR."""
    if not rows:
        print("No rows to summarize.")
        return
    w_gt_one = [r for r in rows if r.w_eta_gt_one]
    wq_gt_one = [r for r in rows if r.w_eta_q_gt_one]
    mismatches = [r for r in rows if r.row_count_mismatch]

    print(f"Computed stress-transport weights for {len(rows)} case(s).")
    print(f"  angular-only w_eta > 1: {len(w_gt_one)}")
    print(f"  q-weighted  w_eta_q > 1: {len(wq_gt_one)}")
    print(f"  row count mismatches: {len(mismatches)}")

    for AR in sorted({r.AR for r in rows}):
        subset = sorted([r for r in rows if np.isclose(r.AR, AR)], key=lambda r: r.alpha)
        w_min = min(r.w_eta for r in subset)
        w_max = max(r.w_eta for r in subset)
        print(f"  AR={AR:g}: angular-only w_eta range [{w_min:.6f}, {w_max:.6f}]")

    ar2 = sorted([r for r in rows if np.isclose(r.AR, 2.0)], key=lambda r: r.alpha)
    if ar2:
        print("\nAR=2 angular-only diagnostic:")
        print("  alpha    beta_CTC    beta_DSMC    w_eta    w_eta_q")
        for row in ar2:
            print(
                f"  {row.alpha:5.2f}  {row.beta_eta_ctc:10.6f}  "
                f"{row.beta_eta_dsmc:10.6f}  {row.w_eta:8.6f}  {row.w_eta_q:8.6f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compute CTC/DSMC rank-2 stress-transport weight w_eta."
    )
    parser.add_argument("--ctc-root", default=DEFAULT_CTC_ROOT,
                        help="Root containing alpha_*_r1.00_AR* CTC result folders.")
    parser.add_argument("--output-dir", default="runs/stress_transport_weight",
                        help="Directory for stress_transport_weights.{csv,json}.")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Optional alpha filter, e.g. 0.90.")
    parser.add_argument("--ar", type=float, default=None,
                        help="Optional AR filter, e.g. 2.0.")
    parser.add_argument("--norm-tol", type=float, default=5.0e-3,
                        help="Allowed max |norm(ghat_post)-1| before failing a case.")
    parser.add_argument("--n-grid", type=int, default=200001,
                        help="Quadrature grid size for DSMC accepted-mu integral.")
    args = parser.parse_args()

    cases = discover_cases(args.ctc_root)
    if not cases:
        raise FileNotFoundError(f"No valid CTC cases found under {args.ctc_root}")

    rows = []
    errors = []
    for case_dir in cases:
        try:
            row = compute_case(case_dir, norm_tol=args.norm_tol, n_grid=args.n_grid)
        except Exception as exc:
            errors.append((case_dir, str(exc)))
            continue
        rows.append(row)

    rows = filter_rows(rows, alpha=args.alpha, AR=args.ar)
    rows.sort(key=lambda r: (r.AR, r.alpha))
    if not rows:
        raise ValueError("No rows left after applying filters.")

    csv_path, json_path = write_outputs(rows, args.output_dir)
    print_summary(rows)
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {json_path}")

    if errors:
        print("\nSkipped cases with errors:")
        for case_dir, message in errors:
            print(f"  {case_dir}: {message}")


if __name__ == "__main__":
    main()
