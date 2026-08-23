import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np


CASE_RE = re.compile(
    r"alpha_([0-9]+(?:\.[0-9]+)?)_r1\.00_AR([0-9]+(?:\.[0-9]+)?)$"
)


@dataclass
class C2FitResult:
    alpha: float
    AR: float
    C2: float
    f_rank0: float
    slope: float
    stderr: float
    intercept: float
    r2: float
    n_events: int
    n_valid: int
    max_abs_epsilon: float
    weight_min: float
    weight_max: float
    direction_results: dict


def compute_rank2_a2(vel, mass=1.0):
    """Compute the dimensionless rank-2 anisotropy invariant from velocities."""
    vel = np.asarray(vel, dtype=float)
    if vel.ndim != 2 or vel.shape[1] != 3:
        raise ValueError(f"vel must have shape (N, 3), got {vel.shape}")
    if vel.shape[0] == 0:
        return 0.0

    peculiar = vel - np.mean(vel, axis=0)
    K = float(mass) * (peculiar.T @ peculiar) / float(vel.shape[0])
    T = float(np.trace(K) / 3.0)
    if not np.isfinite(T) or T <= 0.0:
        return 0.0

    dev = K - T * np.eye(3)
    a2 = float(np.trace(dev @ dev) / (8.0 * T * T))
    if not np.isfinite(a2):
        return 0.0
    return max(0.0, a2)


def apply_rank2_ftr_correction(C_alpha, theta, C2=0.0, a2=0.0):
    """Return f_tr with the optional C2*a2 rank-2 multiplier."""
    theta = max(float(theta), 1.0e-10)
    factor = 1.0 + float(C2) * max(0.0, float(a2))
    return float(C_alpha) * factor * 3.0 * theta / (3.0 * theta + 2.0)


def apply_rank0_ftr_probe(C_alpha, theta, delta=0.0):
    """Return rank-0 f_tr multiplied by a finite-difference probe factor."""
    theta = max(float(theta), 1.0e-10)
    factor = 1.0 + float(delta)
    return float(C_alpha) * factor * 3.0 * theta / (3.0 * theta + 2.0)


def parse_case_dir(case_dir):
    match = CASE_RE.search(os.path.basename(str(case_dir)))
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def load_C2_table(filepath):
    """Load a C2(alpha, AR) table keyed by (alpha, AR)."""
    with open(filepath, "r") as f:
        payload = json.load(f)
    rows = payload.get("rows", payload if isinstance(payload, list) else None)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"No C2 rows found in {filepath}")

    table = {}
    for row in rows:
        alpha = float(row["alpha"])
        AR = float(row["AR"])
        C2 = float(row["C2"])
        if not np.isfinite(C2):
            raise ValueError(
                f"Invalid C2={C2} for alpha={alpha:g}, AR={AR:g}"
            )
        table[(alpha, AR)] = C2
    return table


def lookup_C2(table, alpha, AR):
    """Interpolate C2 in alpha for an exact AR table."""
    alpha = float(alpha)
    AR = float(AR)
    pairs = sorted((a, value) for (a, ar), value in table.items()
                   if np.isclose(ar, AR, atol=1e-12))
    if not pairs:
        available = sorted({key[1] for key in table})
        raise KeyError(
            f"C2 table not available for AR={AR:g}. "
            f"Available AR values: {available}"
        )

    alphas = np.array([p[0] for p in pairs], dtype=float)
    values = np.array([p[1] for p in pairs], dtype=float)
    idx = np.where(np.isclose(alphas, alpha, atol=1e-12))[0]
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


def p2(x):
    """Second Legendre polynomial P2(x)."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (3.0 * x * x - 1.0)


def _normalize_vectors(vectors):
    vectors = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(vectors, axis=1)
    safe = norm > 1.0e-30
    result = np.zeros_like(vectors)
    result[safe] = vectors[safe] / norm[safe, None]
    return result, safe


def _load_case_arrays(case_dir, min_abs_dissipation):
    ftr = np.loadtxt(os.path.join(case_dir, "ftr_data.txt"))
    uvec = np.loadtxt(os.path.join(case_dir, "uvec.dat"))
    ftr = np.atleast_2d(ftr)
    uvec = np.atleast_2d(uvec)
    if uvec.shape[1] < 6:
        raise ValueError(
            f"{os.path.join(case_dir, 'uvec.dat')} must have at least 6 columns"
        )
    n = min(ftr.shape[0], uvec.shape[0])
    if n == 0:
        raise ValueError(f"No rows found in {case_dir}")
    ftr = ftr[:n]
    uvec = uvec[:n]

    f_values = np.asarray(ftr[:, 0], dtype=float)
    diss = np.asarray(ftr[:, 2], dtype=float)
    u1, u1_ok = _normalize_vectors(uvec[:, 0:3])
    u2, u2_ok = _normalize_vectors(uvec[:, 3:6])

    mask = np.isfinite(f_values) & np.isfinite(diss)
    mask &= np.all(np.isfinite(u1), axis=1) & np.all(np.isfinite(u2), axis=1)
    mask &= f_values > -900.0
    mask &= np.abs(diss) > float(min_abs_dissipation)
    mask &= u1_ok & u2_ok
    if not np.any(mask):
        raise ValueError(f"No valid f_tr rows found in {case_dir}")
    return f_values[mask], u1[mask], u2[mask], int(n)


def _weighted_mean(values, weights):
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.nan
    return float(np.sum(weights * values) / denom)


_DIRECTIONS = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def _direction_fit(f_values, u1, u2, eps, direction):
    p1 = p2(u1 @ direction)
    p2_values = p2(u2 @ direction)
    r_values = p1 * p2_values
    f_rank0 = float(np.mean(f_values))
    a2_values = eps * eps
    means = []
    weight_min = np.inf
    weight_max = -np.inf
    for e in eps:
        w_plus = (1.0 + e * p1) * (1.0 + e * p2_values)
        w_minus = (1.0 - e * p1) * (1.0 - e * p2_values)
        weight_min = min(weight_min, float(np.min(w_plus)), float(np.min(w_minus)))
        weight_max = max(weight_max, float(np.max(w_plus)), float(np.max(w_minus)))
        means.append(0.5 * (
            _weighted_mean(f_values, w_plus)
            + _weighted_mean(f_values, w_minus)
        ))
    means = np.asarray(means, dtype=float)

    coeff = np.polyfit(a2_values, means - f_rank0, deg=1)
    slope = float(coeff[0])
    intercept = float(coeff[1])
    pred = slope * a2_values + intercept
    resid = (means - f_rank0) - pred
    ss_res = float(np.sum(resid * resid))
    centered = (means - f_rank0) - float(np.mean(means - f_rank0))
    ss_tot = float(np.sum(centered * centered))
    r2_fit = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
    C2 = slope / f_rank0 if abs(f_rank0) > 1.0e-30 else 0.0
    if not np.isfinite(C2):
        C2 = 0.0
    return {
        "C2": float(C2),
        "slope": slope,
        "intercept": intercept,
        "r2": float(r2_fit),
        "mean_P2_u1": float(np.mean(p1)),
        "mean_P2_u2": float(np.mean(p2_values)),
        "mean_P2_product": float(np.mean(r_values)),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
    }


def estimate_C2_from_case(case_dir, epsilon_values=None,
                          min_abs_dissipation=1.0e-30,
                          bootstrap_samples=200, rng_seed=12345):
    """Estimate C2 for one current-format CTC case from rod orientations."""
    parsed = parse_case_dir(case_dir)
    if parsed is None:
        raise ValueError(f"Cannot parse alpha/AR from case directory: {case_dir}")
    alpha, AR = parsed
    if epsilon_values is None:
        epsilon_values = [0.02, 0.04, 0.06, 0.08]
    eps = np.asarray([abs(float(e)) for e in epsilon_values], dtype=float)
    if eps.size == 0 or np.any(~np.isfinite(eps)) or np.any(eps <= 0.0):
        raise ValueError("epsilon_values must contain positive finite values")

    f_values, u1, u2, n_events = _load_case_arrays(case_dir, min_abs_dissipation)
    f_rank0 = float(np.mean(f_values))

    direction_results = {
        name: _direction_fit(f_values, u1, u2, eps, direction)
        for name, direction in _DIRECTIONS.items()
    }
    slope = 0.5 * (
        direction_results["y"]["slope"] + direction_results["z"]["slope"]
    )
    C2 = slope / f_rank0 if abs(f_rank0) > 1.0e-30 else 0.0
    if not np.isfinite(C2):
        C2 = 0.0
    yz_r2 = 0.5 * (direction_results["y"]["r2"] + direction_results["z"]["r2"])
    intercept = 0.5 * (
        direction_results["y"]["intercept"]
        + direction_results["z"]["intercept"]
    )
    weight_min = min(
        direction_results["y"]["weight_min"],
        direction_results["z"]["weight_min"],
    )
    weight_max = max(
        direction_results["y"]["weight_max"],
        direction_results["z"]["weight_max"],
    )
    direction_results["yz_mean"] = {
        "C2": float(C2),
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(yz_r2),
        "weight_min": float(weight_min),
        "weight_max": float(weight_max),
    }

    stderr = 0.0
    if int(bootstrap_samples) > 0 and f_values.size > 1:
        rng = np.random.default_rng(int(rng_seed))
        boot = np.empty(int(bootstrap_samples), dtype=float)
        n = f_values.size
        for i in range(int(bootstrap_samples)):
            idx = rng.integers(0, n, size=n)
            boot_case = _estimate_slope_no_bootstrap(
                f_values[idx], u1[idx], u2[idx], eps
            )
            boot[i] = boot_case
        stderr = float(np.std(boot, ddof=1))

    return C2FitResult(
        alpha=float(alpha), AR=float(AR), C2=float(C2),
        f_rank0=f_rank0, slope=slope, stderr=stderr,
        intercept=intercept, r2=float(yz_r2), n_events=n_events,
        n_valid=int(f_values.size), max_abs_epsilon=float(np.max(eps)),
        weight_min=float(weight_min), weight_max=float(weight_max),
        direction_results=direction_results,
    )


def _estimate_slope_no_bootstrap(f_values, u1, u2, eps):
    y = _direction_fit(f_values, u1, u2, eps, _DIRECTIONS["y"])
    z = _direction_fit(f_values, u1, u2, eps, _DIRECTIONS["z"])
    return 0.5 * (y["slope"] + z["slope"])


def build_C2_table(ctc_root, AR, epsilon_values=None, output_path=None,
                   min_abs_dissipation=1.0e-30, bootstrap_samples=200,
                   rng_seed=12345):
    """Build C2 rows from current CTC artifacts for one AR."""
    AR = float(AR)
    rows = []
    pattern = os.path.join(str(ctc_root), "alpha_*_r1.00_AR*")
    for case_dir in sorted(glob.glob(pattern)):
        parsed = parse_case_dir(case_dir)
        if parsed is None:
            continue
        _, case_AR = parsed
        if not np.isclose(case_AR, AR, atol=1.0e-12):
            continue
        result = estimate_C2_from_case(
            case_dir, epsilon_values=epsilon_values,
            min_abs_dissipation=min_abs_dissipation,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed,
        )
        rows.append(result.__dict__.copy())

    if not rows:
        raise FileNotFoundError(
            f"No CTC case folders found for AR={AR:g} under {ctc_root}"
        )
    rows.sort(key=lambda row: row["alpha"])
    eps_list = (
        [0.02, 0.04, 0.06, 0.08]
        if epsilon_values is None
        else [abs(float(e)) for e in epsilon_values]
    )
    payload = {
        "metadata": {
            "model": "rank2_C2",
            "ctc_root": str(ctc_root),
            "method": "orientation_symmetric_reweighting",
            "epsilon_values": eps_list,
            "a2_relation": "a2 = epsilon^2",
            "deployed_direction": "yz_mean",
            "diagnostic_direction": "x",
            "uvec_columns": {
                "U1_pre": "1:3",
                "U2_pre": "4:6",
                "U1_post_ignored": "7:9",
                "U2_post_ignored": "10:12",
            },
            "ftr_convention": (
                "ftr_data column 1 is signed "
                "(delta_Et_inel - delta_Et_el) / delta_E_diss"
            ),
            "bootstrap_samples": int(bootstrap_samples),
            "rng_seed": int(rng_seed),
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
