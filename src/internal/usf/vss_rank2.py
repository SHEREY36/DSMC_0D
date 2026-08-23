import glob
import json
import os
import re

import numpy as np


CASE_RE = re.compile(
    r"alpha_([0-9]+(?:\.[0-9]+)?)_r1\.00_AR([0-9]+(?:\.[0-9]+)?)$"
)


def p2(x):
    """Second Legendre polynomial P2(x)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (3.0 * x * x - 1.0)


def vss_rank2_moment(alpha_eff):
    """Return <1 - P2(cos chi)> for the VSS-style exponent."""
    alpha_eff = float(alpha_eff)
    if alpha_eff <= 0.0:
        raise ValueError(f"alpha_eff must be > 0, got {alpha_eff}")
    return 6.0 * alpha_eff / ((alpha_eff + 1.0) * (alpha_eff + 2.0))


def alpha_eff_from_beta(beta, branch="forward"):
    """Invert the VSS rank-2 moment.

    branch='forward' returns the alpha_eff >= 1 root, which is the smooth
    forward-peaked analogue of the current p_eta mixture.
    """
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0 or beta > 1.0:
        raise ValueError(f"beta must be finite and in (0, 1], got {beta}")
    disc = beta * beta - 36.0 * beta + 36.0
    if disc < 0.0:
        raise ValueError(f"VSS inversion discriminant is negative for beta={beta}")
    root = np.sqrt(max(disc, 0.0))
    if branch == "forward":
        return float((6.0 - 3.0 * beta + root) / (2.0 * beta))
    if branch == "backward":
        return float((6.0 - 3.0 * beta - root) / (2.0 * beta))
    raise ValueError(f"branch must be 'forward' or 'backward', got {branch!r}")


def sample_vss_chi(alpha_eff, rng=np.random):
    """Sample chi from cos(chi)=2*R^(1/alpha_eff)-1."""
    alpha_eff = float(alpha_eff)
    if alpha_eff <= 0.0:
        raise ValueError(f"alpha_eff must be > 0, got {alpha_eff}")
    R = float(rng.random())
    cos_chi = 2.0 * (R ** (1.0 / alpha_eff)) - 1.0
    return float(np.arccos(np.clip(cos_chi, -1.0, 1.0)))


def parse_case_dir(case_dir):
    match = CASE_RE.search(os.path.basename(str(case_dir)))
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def beta_ctc_from_chi_file(chi_path):
    """Compute <1 - P2(cos chi)> from a CTC chi.txt file."""
    chi = np.loadtxt(chi_path)
    chi = np.atleast_2d(chi)
    if chi.shape[1] < 2:
        raise ValueError(f"{chi_path} must have at least two columns")
    chi_rad = np.asarray(chi[:, 1], dtype=float)
    mask = np.isfinite(chi_rad)
    mask &= chi_rad >= 0.0
    mask &= chi_rad <= np.pi
    if not np.any(mask):
        raise ValueError(f"No finite chi values in {chi_path}")
    cos_chi = np.cos(chi_rad[mask])
    return float(np.mean(1.0 - p2(cos_chi))), int(np.count_nonzero(mask))


def build_vss_alpha_eff_table(ctc_root, AR, p_eta, output_path=None):
    """Build alpha_eff rows from CTC marginal chi data for one AR."""
    AR = float(AR)
    p_eta = float(p_eta)
    if not np.isfinite(p_eta) or p_eta <= 0.0 or p_eta > 1.0:
        raise ValueError(f"p_eta must be finite and in (0, 1], got {p_eta}")

    rows = []
    pattern = os.path.join(str(ctc_root), "alpha_*_r1.00_AR*")
    for case_dir in sorted(glob.glob(pattern)):
        parsed = parse_case_dir(case_dir)
        if parsed is None:
            continue
        alpha, case_AR = parsed
        if not np.isclose(case_AR, AR, atol=1.0e-12):
            continue
        beta_ctc, n_events = beta_ctc_from_chi_file(
            os.path.join(case_dir, "chi.txt")
        )
        beta_target = p_eta * beta_ctc
        alpha_eff = alpha_eff_from_beta(beta_target, branch="forward")
        rows.append({
            "alpha": float(alpha),
            "AR": float(case_AR),
            "p_eta": float(p_eta),
            "beta_ctc": float(beta_ctc),
            "beta_target": float(beta_target),
            "alpha_eff": float(alpha_eff),
            "n_events": int(n_events),
            "source": str(case_dir),
        })

    if not rows:
        raise FileNotFoundError(
            f"No CTC case folders found for AR={AR:g} under {ctc_root}"
        )
    rows.sort(key=lambda row: row["alpha"])
    payload = {
        "metadata": {
            "model": "vss_rank2",
            "formula": "cos_chi = 2*R^(1/alpha_eff) - 1",
            "rank2_moment": "6*alpha_eff/((alpha_eff+1)*(alpha_eff+2))",
            "branch": "forward",
            "AR": float(AR),
            "p_eta": float(p_eta),
            "ctc_root": str(ctc_root),
        },
        "rows": rows,
    }

    if output_path is not None:
        output_dir = os.path.dirname(str(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    return payload


def load_vss_alpha_eff_table(filepath):
    with open(filepath, "r") as f:
        payload = json.load(f)
    rows = payload.get("rows", payload if isinstance(payload, list) else None)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No VSS alpha_eff rows found in {filepath}")
    table = {}
    for row in rows:
        alpha = float(row["alpha"])
        AR = float(row["AR"])
        alpha_eff = float(row["alpha_eff"])
        if not np.isfinite(alpha_eff) or alpha_eff <= 0.0:
            raise ValueError(
                f"Invalid alpha_eff={alpha_eff} for alpha={alpha:g}, AR={AR:g}"
            )
        table[(alpha, AR)] = alpha_eff
    return table


def lookup_vss_alpha_eff(table, alpha, AR):
    """Interpolate alpha_eff in alpha for an exact AR table."""
    alpha = float(alpha)
    AR = float(AR)
    pairs = sorted((a, value) for (a, ar), value in table.items()
                   if np.isclose(ar, AR, atol=1.0e-12))
    if not pairs:
        available = sorted({key[1] for key in table})
        raise KeyError(
            f"VSS alpha_eff table not available for AR={AR:g}. "
            f"Available AR values: {available}"
        )

    alphas = np.array([p[0] for p in pairs], dtype=float)
    values = np.array([p[1] for p in pairs], dtype=float)
    idx = np.where(np.isclose(alphas, alpha, atol=1.0e-12))[0]
    if idx.size:
        return float(values[idx[0]])
    if alpha <= alphas[0]:
        if len(alphas) == 1:
            return float(values[0])
        slope = (values[1] - values[0]) / (alphas[1] - alphas[0])
        return float(values[0] + slope * (alpha - alphas[0]))
    if alpha >= alphas[-1]:
        if len(alphas) == 1:
            return float(values[-1])
        slope = (values[-1] - values[-2]) / (alphas[-1] - alphas[-2])
        return float(values[-1] + slope * (alpha - alphas[-1]))
    return float(np.interp(alpha, alphas, values))
