#!/usr/bin/env python3
"""Estimate p_eta from the CTC rank-2 stress-transport operator.

The diagnostic compares the weak-form action of three binary collision maps on
the deviatoric relative-velocity dyad:

    B(g) = g g - (|g|^2 / 3) I.

For each event, the scalar energy change q = Etr_f / Etr_i is kept fixed and
only the angular branch is varied:

    CTC:        g' = sqrt(q) * ghat_post_CTC
    scalar:     g' = sqrt(q) * ghat_pre
    full DSMC:  g' = sqrt(q) * ghat_post_DSMC(chi_hs, eps_model)

The fitted p_eta is the fraction of the full-angular DSMC branch needed above
the scalar-only branch to match the CTC rank-2 operator projection.
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

from src.preprocessing.fit_eps_model import load_eps_model, sample_eps_given_mu
from src.simulation.dsmc import _chi_hs
from src.simulation.mu_joint import mu_plane_post_relative_with_eps


DEFAULT_CTC_ROOT = "/home/muhammed/Documents/Thesis/Coll_Models/results"
CASE_RE = re.compile(r"alpha_([0-9]+(?:\.[0-9]+)?)_r1\.00_AR([0-9]+(?:\.[0-9]+)?)$")
GHAT0 = np.array([-1.0, 0.0, 0.0], dtype=float)
IDENTITY = np.eye(3)


@dataclass
class PEtaOperatorRow:
    alpha: float
    AR: float
    n_events: int
    n_used: int
    n_rotations: int
    p_eta_lsq: float
    p_eta_mean: float
    p_eta_std: float
    p_eta_min: float
    p_eta_max: float
    p_eta_xx_yy: float
    p_eta_xx_yy_zz: float
    p_eta_xy: float
    p_eta_xz: float
    p_eta_yz: float
    phi_ctc_xx_yy: float
    phi_scalar_xx_yy: float
    phi_full_xx_yy: float
    mean_q: float
    min_q: float
    max_q: float
    max_ghat_post_norm_error: float
    row_count_mismatch: bool
    denominator_min_abs: float
    denominator_max_abs: float
    finite: bool


def parse_case_dir(case_dir):
    match = CASE_RE.search(os.path.basename(str(case_dir)))
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def discover_cases(ctc_root):
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(ctc_root, "alpha_*_r1.00_AR*"))):
        if parse_case_dir(case_dir) is not None:
            cases.append(case_dir)
    return cases


def load_case_arrays(case_dir):
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

    mismatch = chi.shape[0] != ef.shape[0]
    n = min(chi.shape[0], ef.shape[0])
    if n <= 0:
        raise ValueError(f"No usable rows in {case_dir}")
    return chi[:n], ef[:n], mismatch


def deviatoric_dyad(g):
    g2 = np.einsum("...i,...i->...", g, g)
    return np.einsum("...i,...j->...ij", g, g) - (g2[..., None, None] / 3.0) * IDENTITY


def tensor_basis():
    basis = []
    s = np.zeros((3, 3)); s[0, 0] = 1.0 / np.sqrt(2.0); s[1, 1] = -1.0 / np.sqrt(2.0)
    basis.append(("xx_yy", s))
    s = np.zeros((3, 3)); s[0, 0] = 1.0 / np.sqrt(6.0); s[1, 1] = 1.0 / np.sqrt(6.0); s[2, 2] = -2.0 / np.sqrt(6.0)
    basis.append(("xx_yy_zz", s))
    for label, i, j in (("xy", 0, 1), ("xz", 0, 2), ("yz", 1, 2)):
        s = np.zeros((3, 3))
        s[i, j] = s[j, i] = 1.0 / np.sqrt(2.0)
        basis.append((label, s))
    return basis


def random_rotations(n, rng):
    """Uniform random SO(3) matrices from unit quaternions."""
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q4 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)

    mats = np.empty((n, 3, 3), dtype=float)
    mats[:, 0, 0] = 1.0 - 2.0 * (q2 * q2 + q3 * q3)
    mats[:, 0, 1] = 2.0 * (q1 * q2 - q3 * q4)
    mats[:, 0, 2] = 2.0 * (q1 * q3 + q2 * q4)
    mats[:, 1, 0] = 2.0 * (q1 * q2 + q3 * q4)
    mats[:, 1, 1] = 1.0 - 2.0 * (q1 * q1 + q3 * q3)
    mats[:, 1, 2] = 2.0 * (q2 * q3 - q1 * q4)
    mats[:, 2, 0] = 2.0 * (q1 * q3 - q2 * q4)
    mats[:, 2, 1] = 2.0 * (q2 * q3 + q1 * q4)
    mats[:, 2, 2] = 1.0 - 2.0 * (q1 * q1 + q2 * q2)
    return mats


def rotate_vectors(vectors, rotations):
    return np.einsum("nij,nj->ni", rotations, vectors)


def sample_full_dsmc_directions(mu, eij, alpha, AR, eps_model, rng):
    out = np.empty_like(eij)
    vrel = GHAT0
    for i in range(eij.shape[0]):
        chi_rad = _chi_hs(float(mu[i]), float(alpha))
        if eps_model is None:
            eps_rad = 0.0
        else:
            c_kappa, M, N, J, beta_exp = eps_model
            eps_rad = sample_eps_given_mu(
                float(mu[i]), float(alpha), float(AR),
                c_kappa, M, N, J, beta_exp, rng=rng
            )
        gpost = mu_plane_post_relative_with_eps(vrel, eij[i], chi_rad, 1.0, eps_rad, rng=rng)
        norm = np.linalg.norm(gpost)
        out[i] = gpost / max(norm, 1.0e-30)
    return out


def _basis_projection(S, B):
    return np.einsum("ij,nij->n", S, B)


def compute_case(case_dir, eps_model=None, n_rotations=1, seed=12345,
                 norm_tol=5.0e-3):
    parsed = parse_case_dir(case_dir)
    if parsed is None:
        raise ValueError(f"Cannot parse alpha/AR from {case_dir}")
    alpha, AR = parsed
    chi, ef, row_count_mismatch = load_case_arrays(case_dir)
    n_events = chi.shape[0]

    eij = np.asarray(chi[:, 4:7], dtype=float)
    ghat_post = np.asarray(chi[:, 7:10], dtype=float)
    ef = np.asarray(ef, dtype=float)
    finite = np.all(np.isfinite(eij), axis=1)
    finite &= np.all(np.isfinite(ghat_post), axis=1)
    finite &= np.all(np.isfinite(ef[:, :6]), axis=1)
    finite &= ef[:, 0] > 0.0
    finite &= ef[:, 3] > 0.0

    eij = eij[finite]
    ghat_post = ghat_post[finite]
    ef = ef[finite]
    if eij.shape[0] == 0:
        raise ValueError(f"No finite valid events in {case_dir}")

    eij_norm = np.linalg.norm(eij, axis=1)
    gpost_norm = np.linalg.norm(ghat_post, axis=1)
    max_norm_error = float(np.max(np.abs(gpost_norm - 1.0)))
    valid = (eij_norm > 1.0e-12) & (gpost_norm > 1.0e-12)
    eij = eij[valid] / eij_norm[valid, None]
    ghat_post = ghat_post[valid] / gpost_norm[valid, None]
    ef = ef[valid]
    if eij.shape[0] == 0:
        raise ValueError(f"No nonzero valid events in {case_dir}")
    if max_norm_error > norm_tol:
        raise ValueError(
            f"{case_dir}: max |norm(ghat_post)-1|={max_norm_error:.3e} "
            f"exceeds norm_tol={norm_tol:.3e}"
        )

    dot = eij @ GHAT0
    flip = dot < 0.0
    eij = eij.copy()
    eij[flip] *= -1.0
    mu = np.clip(np.abs(dot), 0.0, 1.0)
    q = ef[:, 3] / ef[:, 0]
    qsqrt = np.sqrt(q)

    rng = np.random.default_rng(int(seed))
    gpre_base = np.broadcast_to(GHAT0, eij.shape)
    gscalar_base = qsqrt[:, None] * gpre_base
    gctc_base = qsqrt[:, None] * ghat_post
    gfull_hat = sample_full_dsmc_directions(mu, eij, alpha, AR, eps_model, rng)
    gfull_base = qsqrt[:, None] * gfull_hat

    phis = {name: [] for name, _ in tensor_basis()}
    nums = []
    dens = []
    phi_debug = None

    for _ in range(int(n_rotations)):
        rotations = random_rotations(gpre_base.shape[0], rng)
        gpre = rotate_vectors(gpre_base, rotations)
        gscalar = rotate_vectors(gscalar_base, rotations)
        gctc = rotate_vectors(gctc_base, rotations)
        gfull = rotate_vectors(gfull_base, rotations)

        Bpre = deviatoric_dyad(gpre)
        dB_scalar = deviatoric_dyad(gscalar) - Bpre
        dB_ctc = deviatoric_dyad(gctc) - Bpre
        dB_full = deviatoric_dyad(gfull) - Bpre

        for name, S in tensor_basis():
            pre_proj = _basis_projection(S, Bpre)
            phi_scalar = float(np.mean(pre_proj * _basis_projection(S, dB_scalar)))
            phi_ctc = float(np.mean(pre_proj * _basis_projection(S, dB_ctc)))
            phi_full = float(np.mean(pre_proj * _basis_projection(S, dB_full)))
            num = phi_ctc - phi_scalar
            den = phi_full - phi_scalar
            phis[name].append((phi_ctc, phi_scalar, phi_full, num, den))

    p_by_basis = {}
    den_values = []
    num_values = []
    for name, values in phis.items():
        arr = np.array(values, dtype=float)
        phi_ctc, phi_scalar, phi_full, num, den = np.mean(arr, axis=0)
        p_by_basis[name] = num / den if abs(den) > 1.0e-30 else np.nan
        num_values.append(num)
        den_values.append(den)
        if name == "xx_yy":
            phi_debug = (phi_ctc, phi_scalar, phi_full)

    num_values = np.array(num_values, dtype=float)
    den_values = np.array(den_values, dtype=float)
    valid_basis = np.isfinite(num_values) & np.isfinite(den_values) & (np.abs(den_values) > 1.0e-30)
    p_vals = np.array([p_by_basis[name] for name, _ in tensor_basis()], dtype=float)
    p_valid = p_vals[np.isfinite(p_vals)]
    p_lsq = float(np.sum(den_values[valid_basis] * num_values[valid_basis])
                  / np.sum(den_values[valid_basis] ** 2))
    finite = bool(np.isfinite(p_lsq) and p_valid.size > 0)

    phi_ctc_xx, phi_scalar_xx, phi_full_xx = phi_debug
    return PEtaOperatorRow(
        alpha=alpha,
        AR=AR,
        n_events=int(n_events),
        n_used=int(gpre_base.shape[0]),
        n_rotations=int(n_rotations),
        p_eta_lsq=p_lsq,
        p_eta_mean=float(np.mean(p_valid)) if p_valid.size else float("nan"),
        p_eta_std=float(np.std(p_valid)) if p_valid.size else float("nan"),
        p_eta_min=float(np.min(p_valid)) if p_valid.size else float("nan"),
        p_eta_max=float(np.max(p_valid)) if p_valid.size else float("nan"),
        p_eta_xx_yy=float(p_by_basis["xx_yy"]),
        p_eta_xx_yy_zz=float(p_by_basis["xx_yy_zz"]),
        p_eta_xy=float(p_by_basis["xy"]),
        p_eta_xz=float(p_by_basis["xz"]),
        p_eta_yz=float(p_by_basis["yz"]),
        phi_ctc_xx_yy=float(phi_ctc_xx),
        phi_scalar_xx_yy=float(phi_scalar_xx),
        phi_full_xx_yy=float(phi_full_xx),
        mean_q=float(np.mean(q)),
        min_q=float(np.min(q)),
        max_q=float(np.max(q)),
        max_ghat_post_norm_error=max_norm_error,
        row_count_mismatch=bool(row_count_mismatch),
        denominator_min_abs=float(np.min(np.abs(den_values[valid_basis]))),
        denominator_max_abs=float(np.max(np.abs(den_values[valid_basis]))),
        finite=finite,
    )


def filter_cases(cases, alpha=None, AR=None):
    out = []
    for case in cases:
        parsed = parse_case_dir(case)
        if parsed is None:
            continue
        a, ar = parsed
        if alpha is not None and not np.isclose(a, alpha, atol=5.0e-8):
            continue
        if AR is not None and not np.isclose(ar, AR, atol=5.0e-8):
            continue
        out.append(case)
    return out


def write_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ctc_p_eta_operator.csv"
    json_path = output_dir / "ctc_p_eta_operator.json"
    fieldnames = list(PEtaOperatorRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    payload = {
        "assumptions": {
            "pre_collision_ghat": GHAT0.tolist(),
            "stress_mode": "B_ij(g)=g_i*g_j-|g|^2*delta_ij/3",
            "projection": "<(S:B_pre) S:(B_post-B_pre)>",
            "scalar_branch": "sqrt(q)*ghat_pre",
            "full_branch": "sqrt(q)*DSMC chi_hs plus eps model when available",
            "random_rotations": "restore isotropic l=2 basis from fixed-ghat0 CTC frame",
        },
        "rows": [asdict(row) for row in rows],
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return csv_path, json_path


def print_summary(rows):
    print(f"Computed CTC p_eta operator diagnostic for {len(rows)} case(s).")
    for AR in sorted({r.AR for r in rows}):
        subset = sorted([r for r in rows if np.isclose(r.AR, AR)], key=lambda r: r.alpha)
        vals = [r.p_eta_lsq for r in subset if np.isfinite(r.p_eta_lsq)]
        if not vals:
            continue
        print(
            f"  AR={AR:g}: p_eta_lsq range [{min(vals):.4f}, {max(vals):.4f}], "
            f"mean={np.mean(vals):.4f}"
        )
        for r in subset:
            print(
                f"    alpha={r.alpha:.2f}: p_lsq={r.p_eta_lsq:.4f}, "
                f"basis_mean={r.p_eta_mean:.4f}, basis_std={r.p_eta_std:.4f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate p_eta from CTC rank-2 stress-transport operator"
    )
    parser.add_argument("--ctc-root", default=DEFAULT_CTC_ROOT)
    parser.add_argument("--output-dir", default="runs/ctc_p_eta_operator")
    parser.add_argument("--eps-model", default="models/eps_azimuth_coeffs.npz")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--ar", type=float, default=None)
    parser.add_argument("--n-rotations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--norm-tol", type=float, default=5.0e-3)
    args = parser.parse_args()

    eps_model = load_eps_model(args.eps_model)
    if eps_model is None:
        print(f"Warning: eps model not found at {args.eps_model}; using eps=0 full branch.")

    cases = filter_cases(discover_cases(args.ctc_root), alpha=args.alpha, AR=args.ar)
    if not cases:
        raise FileNotFoundError("No matching CTC cases found")

    rows = []
    for case in cases:
        row = compute_case(
            case, eps_model=eps_model, n_rotations=args.n_rotations,
            seed=args.seed, norm_tol=args.norm_tol
        )
        rows.append(row)

    csv_path, json_path = write_outputs(rows, args.output_dir)
    print_summary(rows)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
