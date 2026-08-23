"""Estimate rank-zero dissipative routing coefficients from CTC events.

The public DSMC HCS kernel routes dissipated collision energy with

    f_tr = C^(0)(alpha, AR) * 3*theta / (3*theta + 2).

This module estimates the scalar coefficient C^(0) directly from CTC collision
records.  The important convention is that the CTC Fortran output stores a
signed translational routing response,

    ftr_signed = (Etr_f_inelastic - Etr_f_elastic) / delta_E_diss,

where the elastic pass supplies the conservative translational exchange
baseline.  The DSMC runtime instead uses a positive translational removal
fraction in ``Etrans_f = ... - f_tr * delta_E``.  Therefore this estimator uses
``f_req = -ftr_signed`` and moment-matches the dissipative energy moment:

    C0 = sum(w * f_req * delta_E) / sum(w * B(theta) * delta_E),
    B(theta) = 3*theta / (3*theta + 2).

The existing CTC grid is currently sampled at r=theta=1.0, so theta-dependence
is recorded as a limitation in the diagnostics rather than inferred.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.preprocessing.relaxation import Zr, prepare_theta


CASE_RE = re.compile(r"alpha_([0-9.]+)_r([0-9.]+)_AR([0-9.]+)$")
DEFAULT_CTC_ROOT = Path("/home/muhammed/Documents/Thesis/Coll_Models/results")
TABLE_MODEL_NAME = "rank0_C0_ctc"
HCS_BALANCE_MODEL_NAME = "rank0_C0_hcs_balance"
HCS_FIXED_POINT_MODEL_NAME = "rank0_C0_hcs_fixed_point"
HCS_SELF_CONSISTENT_MODEL_NAME = "rank0_C0_hcs_self_consistent"
HCS_GRID_SELF_CONSISTENT_MODEL_NAME = "rank0_C0_hcs_grid_self_consistent"
HCS_GRID_SCALED_MODEL_NAME = "rank0_C0_hcs_grid_scaled_self_consistent"
HCS_GRID_CTC_BALANCE_MODEL_NAME = "rank0_C0_hcs_grid_ctc_balance"
HCS_GRID_CTC_BALANCE_REG_MODEL_NAME = "rank0_C0_hcs_grid_ctc_balance_regularized"


@dataclass(frozen=True)
class CTCCase:
    alpha: float
    theta: float
    AR: float
    path: Path


def ar_label(AR: float) -> str:
    """Return the repository's AR label, e.g. 2.0 -> AR20."""
    return f"AR{int(round(float(AR) * 10)):02d}"


def table_key(alpha: float, AR: float) -> str:
    return f"({float(alpha):.3f}, {float(AR):.1f})"


def mode_neutral_factor(theta: float | np.ndarray) -> float | np.ndarray:
    theta = np.asarray(theta, dtype=float)
    out = 3.0 * theta / (3.0 * theta + 2.0)
    return float(out) if out.ndim == 0 else out


def parse_case_dir(path: str | Path) -> CTCCase | None:
    path = Path(path)
    match = CASE_RE.match(path.name)
    if not match:
        return None
    return CTCCase(
        alpha=float(match.group(1)),
        theta=float(match.group(2)),
        AR=float(match.group(3)),
        path=path,
    )


def discover_cases(
    root: str | Path,
    AR: float | None = None,
    alphas: Iterable[float] | None = None,
) -> list[CTCCase]:
    root = Path(root)
    alpha_filter = None
    if alphas is not None:
        alpha_filter = np.array([float(a) for a in alphas], dtype=float)

    cases = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        case = parse_case_dir(child)
        if case is None:
            continue
        if AR is not None and not np.isclose(case.AR, float(AR), atol=1.0e-12):
            continue
        if alpha_filter is not None and not np.any(
            np.isclose(alpha_filter, case.alpha, atol=1.0e-12)
        ):
            continue
        required = [child / "ftr_data.txt", child / "NPhit.txt"]
        if all(path.exists() for path in required):
            cases.append(case)
    return cases


def _load_2d(path: Path, min_cols: int) -> np.ndarray:
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < min_cols:
        raise ValueError(f"Expected at least {min_cols} columns in {path}")
    return data


def load_case_arrays(case_dir: str | Path) -> dict[str, np.ndarray]:
    case_dir = Path(case_dir)
    arrays = {
        "ftr": _load_2d(case_dir / "ftr_data.txt", 3),
        "nphit": np.atleast_1d(np.loadtxt(case_dir / "NPhit.txt")).astype(int),
    }
    ef_path = case_dir / "Ef.txt"
    if ef_path.exists() and ef_path.stat().st_size > 0:
        arrays["ef"] = _load_2d(ef_path, 7)
    chi_path = case_dir / "chi.txt"
    if chi_path.exists() and chi_path.stat().st_size > 0:
        arrays["chi"] = _load_2d(chi_path, 1)
    return arrays


def collision_weights(
    arrays: dict[str, np.ndarray],
    weight_mode: str = "area",
) -> np.ndarray:
    """Return CTC-to-DSMC collision-measure weights.

    ``area`` is the default for the current Fortran initializer, where the two
    transverse impact coordinates are sampled uniformly in a square/disk-like
    area proposal.  ``uniform-b`` is a diagnostic alternative for data where the
    scalar impact parameter itself was sampled uniformly, requiring a factor b.
    """
    n = arrays["ftr"].shape[0]
    if weight_mode == "area":
        return np.ones(n, dtype=float)
    if weight_mode == "uniform-b":
        if "chi" not in arrays:
            raise ValueError("weight_mode='uniform-b' requires chi.txt")
        b = arrays["chi"][:, 0]
        return np.maximum(np.asarray(b, dtype=float), 0.0)
    raise ValueError(f"Unknown weight_mode={weight_mode!r}; use area or uniform-b")


def _ratio_estimate(
    f_req: np.ndarray,
    delta_E: np.ndarray,
    weights: np.ndarray,
    theta: float,
) -> tuple[float, float, float]:
    B = mode_neutral_factor(theta)
    numerator = float(np.sum(weights * f_req * delta_E))
    denominator = float(np.sum(weights * B * delta_E))
    if not np.isfinite(denominator) or abs(denominator) <= 1.0e-30:
        return math.nan, numerator, denominator
    return float(numerator / denominator), numerator, denominator


def estimate_case_c0_fast_stats(
    case: CTCCase,
    *,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
) -> tuple[float, dict[str, object]]:
    """Estimate C0 from only the scalar files needed by the theta-grid path."""
    ftr_path = case.path / "ftr_data.txt"
    nphit_path = case.path / "NPhit.txt"
    ftr = np.loadtxt(ftr_path, usecols=(0, 2))
    ftr = np.atleast_2d(ftr)
    nphit = np.atleast_1d(np.loadtxt(nphit_path)).astype(int)
    if nphit.shape[0] != ftr.shape[0]:
        raise ValueError(
            f"Row mismatch in {case.path}: ftr_data has {ftr.shape[0]} rows, "
            f"NPhit has {nphit.shape[0]}"
        )

    f_req = -ftr[:, 0]
    delta_E = ftr[:, 1]
    if weight_mode == "area":
        weights = np.ones(ftr.shape[0], dtype=float)
    elif weight_mode == "uniform-b":
        chi = np.loadtxt(case.path / "chi.txt", usecols=(0,))
        weights = np.maximum(np.atleast_1d(chi).astype(float), 0.0)
    else:
        raise ValueError(f"Unknown weight_mode={weight_mode!r}; use area or uniform-b")

    mask = (
        np.isfinite(f_req)
        & np.isfinite(delta_E)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    if single_hit_only:
        mask &= nphit == 1
    mask &= delta_E > min_delta_E

    n_total = int(ftr.shape[0])
    n_single_hit = int(np.sum(nphit == 1))
    n_used = int(np.sum(mask))
    if case.alpha >= 1.0 - 1.0e-12:
        C0 = 0.0
        numerator = denominator = 0.0
        status = "elastic_alpha_by_convention"
    elif n_used == 0:
        C0 = math.nan
        numerator = denominator = math.nan
        status = "no_valid_dissipative_events"
    else:
        C0, numerator, denominator = _ratio_estimate(
            f_req[mask], delta_E[mask], weights[mask], case.theta
        )
        status = "ok" if np.isfinite(C0) else "singular_moment"

    used_f_req = f_req[mask]
    used_delta_E = delta_E[mask]
    used_weights = weights[mask]
    diagnostics: dict[str, object] = {
        "alpha": float(case.alpha),
        "AR": float(case.AR),
        "theta_used": float(case.theta),
        "case_dir": str(case.path),
        "status": status,
        "weight_mode": weight_mode,
        "single_hit_only": bool(single_hit_only),
        "n_total": n_total,
        "n_single_hit": n_single_hit,
        "n_used": n_used,
        "mean_f_req": float(np.mean(used_f_req)) if n_used else math.nan,
        "median_f_req": float(np.median(used_f_req)) if n_used else math.nan,
        "std_f_req": float(np.std(used_f_req, ddof=1)) if n_used > 1 else math.nan,
        "weighted_mean_delta_E": (
            float(np.average(used_delta_E, weights=used_weights)) if n_used else math.nan
        ),
        "frac_f_req_outside_0_1": (
            float(np.mean((used_f_req < 0.0) | (used_f_req > 1.0)))
            if n_used else math.nan
        ),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "C0": float(C0),
    }
    return float(C0), diagnostics


def estimate_case_ctc_reservoir_stats(
    case: CTCCase,
    *,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
) -> dict[str, object]:
    """Return CTC event-level reservoir-balance statistics for one theta."""
    ftr_path = case.path / "ftr_data.txt"
    nphit_path = case.path / "NPhit.txt"
    ftr = np.loadtxt(ftr_path, usecols=(0, 1, 2))
    ftr = np.atleast_2d(ftr)
    nphit = np.atleast_1d(np.loadtxt(nphit_path)).astype(int)
    if nphit.shape[0] != ftr.shape[0]:
        raise ValueError(
            f"Row mismatch in {case.path}: ftr_data has {ftr.shape[0]} rows, "
            f"NPhit has {nphit.shape[0]}"
        )

    f_req = -ftr[:, 0]
    A = ftr[:, 1]
    delta_E = ftr[:, 2]
    if weight_mode == "area":
        weights = np.ones(ftr.shape[0], dtype=float)
    elif weight_mode == "uniform-b":
        chi = np.loadtxt(case.path / "chi.txt", usecols=(0,))
        weights = np.maximum(np.atleast_1d(chi).astype(float), 0.0)
    else:
        raise ValueError(f"Unknown weight_mode={weight_mode!r}; use area or uniform-b")

    mask = (
        np.isfinite(f_req)
        & np.isfinite(A)
        & np.isfinite(delta_E)
        & np.isfinite(weights)
        & (weights >= 0.0)
        & (delta_E > min_delta_E)
    )
    if single_hit_only:
        mask &= nphit == 1

    n_used = int(np.sum(mask))
    if n_used:
        w = weights[mask]
        dEtr = A[mask] - f_req[mask] * delta_E[mask]
        dErot = -A[mask] - (1.0 - f_req[mask]) * delta_E[mask]
        mean_dEtr = float(np.average(dEtr, weights=w))
        mean_dErot = float(np.average(dErot, weights=w))
        residual = float((2.0 / (3.0 * case.theta)) * mean_dEtr - mean_dErot)
        status = "ok"
    else:
        mean_dEtr = mean_dErot = residual = math.nan
        status = "no_valid_dissipative_events"

    return {
        "alpha": float(case.alpha),
        "AR": float(case.AR),
        "theta_used": float(case.theta),
        "case_dir": str(case.path),
        "status": status,
        "weight_mode": weight_mode,
        "single_hit_only": bool(single_hit_only),
        "n_total": int(ftr.shape[0]),
        "n_single_hit": int(np.sum(nphit == 1)),
        "n_used": n_used,
        "ctc_mean_delta_Etr": mean_dEtr,
        "ctc_mean_delta_Erot": mean_dErot,
        "ctc_theta_residual": residual,
    }


def _bootstrap_c0(
    f_req: np.ndarray,
    delta_E: np.ndarray,
    weights: np.ndarray,
    theta: float,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    if samples <= 0 or f_req.size < 2:
        return {
            "bootstrap_samples": int(samples),
            "C0_bootstrap_stderr": None,
            "C0_ci95_low": None,
            "C0_ci95_high": None,
        }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    n = f_req.size
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        values[i], _, _ = _ratio_estimate(
            f_req[idx], delta_E[idx], weights[idx], theta
        )
    values = values[np.isfinite(values)]
    if values.size == 0:
        stderr = low = high = None
    else:
        stderr = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        low, high = [float(x) for x in np.percentile(values, [2.5, 97.5])]
    return {
        "bootstrap_samples": int(samples),
        "C0_bootstrap_stderr": stderr,
        "C0_ci95_low": low,
        "C0_ci95_high": high,
    }


def scalar_reservoir_check(
    f_req: np.ndarray,
    delta_Et_el: np.ndarray,
    delta_E: np.ndarray,
    C0: float,
    theta: float,
) -> dict[str, float | str]:
    """A lightweight reservoir-only sanity check.

    This is intentionally not a calibration target.  It reports the mean
    translational and rotational energy increments implied by the fitted rank-0
    route at the sampled CTC state.
    """
    B = mode_neutral_factor(theta)
    f_model = float(C0) * B
    dEtr = delta_Et_el - f_model * delta_E
    dErot = -delta_Et_el - (1.0 - f_model) * delta_E
    mean_dEtr = float(np.mean(dEtr)) if dEtr.size else math.nan
    mean_dErot = float(np.mean(dErot)) if dErot.size else math.nan
    if mean_dEtr < 0.0 and mean_dErot < 0.0:
        trend = "both reservoirs cool"
    elif mean_dEtr < 0.0:
        trend = "translational reservoir cools faster"
    elif mean_dErot < 0.0:
        trend = "rotational reservoir cools faster"
    else:
        trend = "net heating/exchange dominated"
    return {
        "f_tr_model_at_theta": float(f_model),
        "mean_delta_Etr_model": mean_dEtr,
        "mean_delta_Erot_model": mean_dErot,
        "trend": trend,
    }


def hcs_balance_coefficient(mean_A: float, mean_delta_E: float, theta: float) -> float:
    """Return the effective rank-zero C from HCS reservoir balance."""
    denom = mode_neutral_factor(theta) * float(mean_delta_E)
    if not np.isfinite(denom) or abs(denom) <= 1.0e-30:
        return math.nan
    return float(1.0 + float(mean_A) / denom)


def estimate_case_c0(
    case: CTCCase,
    *,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 12345,
) -> tuple[float, dict[str, object]]:
    arrays = load_case_arrays(case.path)
    ftr = arrays["ftr"]
    nphit = arrays["nphit"]
    if nphit.shape[0] != ftr.shape[0]:
        raise ValueError(
            f"Row mismatch in {case.path}: ftr_data has {ftr.shape[0]} rows, "
            f"NPhit has {nphit.shape[0]}"
        )

    ftr_signed = ftr[:, 0]
    f_req = -ftr_signed
    delta_Et_el = ftr[:, 1]
    delta_E = ftr[:, 2]
    weights = collision_weights(arrays, weight_mode)

    finite = (
        np.isfinite(f_req)
        & np.isfinite(delta_E)
        & np.isfinite(delta_Et_el)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    if single_hit_only:
        finite &= nphit == 1
    dissipative = finite & (delta_E > min_delta_E)

    n_total = int(ftr.shape[0])
    n_single_hit = int(np.sum(nphit == 1))
    n_used = int(np.sum(dissipative))

    if case.alpha >= 1.0 - 1.0e-12:
        C0 = 0.0
        numerator = denominator = 0.0
        status = "elastic_alpha_by_convention"
    elif n_used == 0:
        C0 = math.nan
        numerator = denominator = math.nan
        status = "no_valid_dissipative_events"
    else:
        C0, numerator, denominator = _ratio_estimate(
            f_req[dissipative],
            delta_E[dissipative],
            weights[dissipative],
            case.theta,
        )
        status = "ok" if np.isfinite(C0) else "singular_moment"

    used_f_req = f_req[dissipative]
    used_delta_E = delta_E[dissipative]
    used_weights = weights[dissipative]
    diagnostics: dict[str, object] = {
        "alpha": float(case.alpha),
        "AR": float(case.AR),
        "theta_used": float(case.theta),
        "case_dir": str(case.path),
        "status": status,
        "weight_mode": weight_mode,
        "single_hit_only": bool(single_hit_only),
        "n_total": n_total,
        "n_single_hit": n_single_hit,
        "n_used": n_used,
        "mean_f_req": float(np.mean(used_f_req)) if n_used else math.nan,
        "median_f_req": float(np.median(used_f_req)) if n_used else math.nan,
        "std_f_req": float(np.std(used_f_req, ddof=1)) if n_used > 1 else math.nan,
        "weighted_mean_delta_E": (
            float(np.average(used_delta_E, weights=used_weights)) if n_used else math.nan
        ),
        "frac_f_req_outside_0_1": (
            float(np.mean((used_f_req < 0.0) | (used_f_req > 1.0)))
            if n_used else math.nan
        ),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "C0": float(C0),
    }
    if n_used and np.isfinite(C0):
        diagnostics.update(
            _bootstrap_c0(
                used_f_req,
                used_delta_E,
                used_weights,
                case.theta,
                bootstrap_samples,
                bootstrap_seed,
            )
        )
        diagnostics.update(
            scalar_reservoir_check(
                used_f_req,
                delta_Et_el[dissipative],
                used_delta_E,
                C0,
                case.theta,
            )
        )
    else:
        diagnostics.update(
            {
                "bootstrap_samples": int(bootstrap_samples),
                "C0_bootstrap_stderr": None,
                "C0_ci95_low": None,
                "C0_ci95_high": None,
                "f_tr_model_at_theta": math.nan,
                "mean_delta_Etr_model": math.nan,
                "mean_delta_Erot_model": math.nan,
                "trend": "not evaluated",
            }
        )
    return float(C0), diagnostics


def _load_theta_target(theta_table: dict[str, float], alpha: float, AR: float) -> float:
    key = table_key(alpha, AR)
    if key in theta_table:
        return float(theta_table[key])

    rows = []
    for raw_key, value in theta_table.items():
        a_str, ar_str = raw_key.strip()[1:-1].split(",")
        a_i = float(a_str.strip())
        ar_i = float(ar_str.strip())
        if np.isclose(ar_i, float(AR), atol=1.0e-12):
            rows.append((a_i, float(value)))
    if not rows:
        raise KeyError(f"No theta_target rows for AR={AR:g}")
    rows.sort()
    alphas = np.array([row[0] for row in rows], dtype=float)
    values = np.array([row[1] for row in rows], dtype=float)
    return float(np.interp(float(alpha), alphas, values))


def load_theta_table(path: str | Path) -> dict[str, float]:
    with open(path) as f:
        raw = json.load(f)
    return {str(key): float(value) for key, value in raw.items()}


def _select_case_mask(
    arrays: dict[str, np.ndarray],
    *,
    single_hit_only: bool,
    min_total_energy: float = 1.0e-12,
) -> np.ndarray:
    ftr = arrays["ftr"]
    nphit = arrays["nphit"]
    if nphit.shape[0] != ftr.shape[0]:
        raise ValueError(
            f"Row mismatch: ftr_data has {ftr.shape[0]} rows, NPhit has {nphit.shape[0]}"
        )
    if "ef" not in arrays:
        raise FileNotFoundError("hcs-balance estimator requires Ef.txt")
    ef = arrays["ef"]
    if ef.shape[0] != ftr.shape[0]:
        raise ValueError(
            f"Row mismatch: Ef has {ef.shape[0]} rows, ftr_data has {ftr.shape[0]}"
        )
    Erot_i = ef[:, 1] + ef[:, 2]
    mask = np.isfinite(Erot_i) & (Erot_i > min_total_energy)
    if single_hit_only:
        mask &= nphit == 1
    return mask


def estimate_case_hcs_balance_c0(
    case: CTCCase,
    *,
    models,
    theta_target: float,
    beta_a: float,
    beta_b: float,
    single_hit_only: bool = True,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 12345,
    exchange_seed: int = 12345,
) -> tuple[float, dict[str, object]]:
    """Estimate effective C0 from the actual HCS reservoir-balance closure."""
    arrays = load_case_arrays(case.path)
    mask = _select_case_mask(arrays, single_hit_only=single_hit_only)
    ef = arrays["ef"]
    n_total = int(ef.shape[0])
    n_single_hit = int(np.sum(arrays["nphit"] == 1))
    n_used = int(np.sum(mask))
    theta = float(theta_target)
    B = float(mode_neutral_factor(theta))

    if case.alpha >= 1.0 - 1.0e-12:
        C0 = 0.0
        mean_A = mean_delta_E = numerator = denominator = 0.0
        status = "elastic_alpha_by_convention"
        A = delta_E = np.array([], dtype=float)
    elif n_used == 0:
        C0 = math.nan
        mean_A = mean_delta_E = numerator = denominator = math.nan
        status = "no_valid_energy_rows"
        A = delta_E = np.array([], dtype=float)
    else:
        Etr_i_raw = ef[mask, 0]
        Er1_i = ef[mask, 1]
        Er2_i = ef[mask, 2]
        Erot_i = Er1_i + Er2_i
        Etr_i = 1.5 * theta * Erot_i
        Etotal_i = Etr_i + Erot_i
        epsilon_tr_i = Etr_i / Etotal_i
        epsilon_rot_1_i = Er1_i / Erot_i

        rng = np.random.default_rng(exchange_seed)
        np.random.seed(exchange_seed)
        P_r = min(1.0 / Zr(theta, eta=1.0, alpha=case.alpha), 0.5)
        relax = rng.random(n_used) < 2.0 * P_r
        epsilon_tr_f = epsilon_tr_i.copy()
        theta_for_gmm = prepare_theta(theta)
        for idx in np.where(relax)[0]:
            sample = models.cond_gmm.sample_conditionals(
                r=theta_for_gmm,
                e_tr=float(epsilon_tr_i[idx]),
                e_r1=float(epsilon_rot_1_i[idx]),
                n_samples=1,
            )
            epsilon_tr_f[idx] = float(sample[0, 0])

        A = (epsilon_tr_f - epsilon_tr_i) * Etotal_i
        gamma_factor = (
            float(models.get_gamma_max(case.alpha, case.AR))
            * float(models.get_one_hit(case.alpha, case.AR))
            * float(beta_a) / (float(beta_a) + float(beta_b))
        )
        delta_E = gamma_factor * Etotal_i
        mean_A = float(np.mean(A))
        mean_delta_E = float(np.mean(delta_E))
        C0 = hcs_balance_coefficient(mean_A, mean_delta_E, theta)
        numerator = float(mean_A)
        denominator = float(B * mean_delta_E)
        status = "ok" if np.isfinite(C0) else "singular_balance"

    diagnostics: dict[str, object] = {
        "alpha": float(case.alpha),
        "AR": float(case.AR),
        "theta_used": theta,
        "B_theta": B,
        "case_dir": str(case.path),
        "status": status,
        "estimator": "hcs-balance",
        "single_hit_only": bool(single_hit_only),
        "n_total": n_total,
        "n_single_hit": n_single_hit,
        "n_used": n_used,
        "mean_A": float(mean_A),
        "mean_delta_E_closure": float(mean_delta_E),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "C0": float(C0),
        "negative_due_to_exchange": bool(
            np.isfinite(mean_A)
            and np.isfinite(denominator)
            and mean_A < -denominator
        ),
    }
    if n_used and np.isfinite(C0):
        diagnostics.update(
            _bootstrap_hcs_balance(
                A,
                delta_E,
                theta,
                bootstrap_samples,
                bootstrap_seed,
            )
        )
    else:
        diagnostics.update(
            {
                "bootstrap_samples": int(bootstrap_samples),
                "C0_bootstrap_stderr": None,
                "C0_ci95_low": None,
                "C0_ci95_high": None,
            }
        )
    return float(C0), diagnostics


def _sample_hcs_pair_ensemble(theta: float, n_samples: int, seed: int):
    """Sample the accepted-collision pair-energy ensemble used by DSMC HCS.

    The NTC accepted-collision measure is proportional to relative speed.  For
    Maxwellian translational velocities this gives
    ``E_tr_pair / T_tr ~ Gamma(shape=2, scale=1)``.  Each 2D rotational energy
    is exponential with mean ``T_rot``.
    """
    rng = np.random.default_rng(seed)
    Trot = 1.0
    Ttr = float(theta) * Trot
    Etr = rng.gamma(shape=2.0, scale=Ttr, size=n_samples)
    Er1 = rng.exponential(scale=Trot, size=n_samples)
    Er2 = rng.exponential(scale=Trot, size=n_samples)
    Erot = Er1 + Er2
    Etotal = Etr + Erot
    epsilon_tr_i = Etr / Etotal
    epsilon_rot_1_i = Er1 / Erot
    return rng, Etr, Erot, Etotal, epsilon_tr_i, epsilon_rot_1_i


def _sample_conditionals_vectorized(gmm, r, e_tr, e_r1, seed: int):
    """Vectorized equivalent of ConditionalGMM.sample_conditionals when possible."""
    required = (
        "weights",
        "inv_xx",
        "logdet_xx",
        "A",
        "mu_y",
        "L",
        "scaler_mean",
        "scaler_scale",
        "mu_x",
        "_log_weights",
        "_const",
    )
    if not all(hasattr(gmm, name) for name in required):
        return None

    e_tr = np.asarray(e_tr, dtype=float)
    e_r1 = np.asarray(e_r1, dtype=float)
    if e_tr.size == 0:
        return np.empty((0, int(getattr(gmm, "D_y", 2))), dtype=float)

    eps = 1.0e-8
    x_proc = np.empty((e_tr.size, 3), dtype=float)
    x_proc[:, 0] = np.log(float(r))
    clipped = np.clip(np.column_stack([e_tr, e_r1]), eps, 1.0 - eps)
    x_proc[:, 1:] = np.log(clipped / (1.0 - clipped))
    x_scaled = (x_proc - gmm.scaler_mean[:3]) / gmm.scaler_scale[:3]

    diff = x_scaled[:, None, :] - gmm.mu_x[None, :, :]
    mahal = np.einsum("nmi,mij,nmj->nm", diff, gmm.inv_xx, diff)
    log_resp = (
        gmm._log_weights[None, :]
        + float(gmm._const)
        - 0.5 * gmm.logdet_xx[None, :]
        - 0.5 * mahal
    )
    log_resp -= np.max(log_resp, axis=1, keepdims=True)
    resp = np.exp(log_resp)
    resp_sum = np.sum(resp, axis=1, keepdims=True)
    bad = resp_sum[:, 0] <= 1.0e-300
    resp = np.divide(
        resp,
        resp_sum,
        out=np.full_like(resp, 1.0 / resp.shape[1]),
        where=resp_sum > 1.0e-300,
    )
    if np.any(bad):
        resp[bad, :] = 1.0 / resp.shape[1]

    rng = np.random.default_rng(seed)
    draws = rng.random(e_tr.size)
    comp = np.sum(draws[:, None] > np.cumsum(resp, axis=1), axis=1)
    comp = np.minimum(comp, resp.shape[1] - 1)

    centered = x_scaled - gmm.mu_x[comp]
    mu_cond = gmm.mu_y[comp] + np.einsum("nij,nj->ni", gmm.A[comp], centered)
    z = rng.standard_normal((e_tr.size, int(gmm.D_y)))
    y_scaled = mu_cond + np.einsum("nij,nj->ni", gmm.L[comp], z)
    y_proc = (
        y_scaled * gmm.scaler_scale[gmm.D_x : gmm.D_x + gmm.D_y]
        + gmm.scaler_mean[gmm.D_x : gmm.D_x + gmm.D_y]
    )
    return 1.0 / (1.0 + np.exp(-y_proc))


def _sample_dsmc_exchange(
    *,
    models,
    alpha: float,
    theta: float,
    Etotal: np.ndarray,
    epsilon_tr_i: np.ndarray,
    epsilon_rot_1_i: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    epsilon_tr_f = epsilon_tr_i.copy()
    P_r = min(1.0 / Zr(theta, eta=1.0, alpha=alpha), 0.5)
    relax = rng.random(epsilon_tr_i.size) < 2.0 * P_r
    theta_for_gmm = prepare_theta(theta)

    np.random.seed(seed)
    relax_idx = np.where(relax)[0]
    vectorized = _sample_conditionals_vectorized(
        models.cond_gmm,
        theta_for_gmm,
        epsilon_tr_i[relax_idx],
        epsilon_rot_1_i[relax_idx],
        seed,
    )
    if vectorized is not None:
        epsilon_tr_f[relax_idx] = vectorized[:, 0]
    else:
        for idx in relax_idx:
            sample = models.cond_gmm.sample_conditionals(
                r=theta_for_gmm,
                e_tr=float(epsilon_tr_i[idx]),
                e_r1=float(epsilon_rot_1_i[idx]),
                n_samples=1,
            )
            epsilon_tr_f[idx] = float(sample[0, 0])
    return epsilon_tr_f


def fixed_point_residual(
    C0: float,
    *,
    theta: float,
    Etr_i: np.ndarray,
    Erot_i: np.ndarray,
    Etotal_i: np.ndarray,
    epsilon_tr_f: np.ndarray,
    delta_E: np.ndarray,
) -> float:
    """Return HCS relative-cooling residual for one trial C0."""
    f_tr = float(C0) * mode_neutral_factor(theta)
    Etrans_f = epsilon_tr_f * Etotal_i - f_tr * delta_E
    Erot_f = (1.0 - epsilon_tr_f) * Etotal_i - (1.0 - f_tr) * delta_E

    trans_neg = Etrans_f < 0.0
    if np.any(trans_neg):
        Erot_f = Erot_f.copy()
        Etrans_f = Etrans_f.copy()
        Erot_f[trans_neg] += Etrans_f[trans_neg]
        Etrans_f[trans_neg] = 1.0e-30
    rot_neg = Erot_f < 0.0
    if np.any(rot_neg):
        if not trans_neg.any():
            Etrans_f = Etrans_f.copy()
            Erot_f = Erot_f.copy()
        Etrans_f[rot_neg] += Erot_f[rot_neg]
        Erot_f[rot_neg] = 1.0e-30

    dEtr = Etrans_f - Etr_i
    dErot = Erot_f - Erot_i
    return float((2.0 / (3.0 * theta)) * np.mean(dEtr) - np.mean(dErot))


def solve_fixed_point_c0(
    *,
    theta: float,
    Etr_i: np.ndarray,
    Erot_i: np.ndarray,
    Etotal_i: np.ndarray,
    epsilon_tr_f: np.ndarray,
    delta_E: np.ndarray,
    C_min: float = -10.0,
    C_max: float = 5.0,
    scan_points: int = 161,
    tol: float = 1.0e-8,
    max_iter: int = 80,
) -> tuple[float, dict[str, object]]:
    grid = np.linspace(float(C_min), float(C_max), int(scan_points))
    residuals = np.array([
        fixed_point_residual(
            C,
            theta=theta,
            Etr_i=Etr_i,
            Erot_i=Erot_i,
            Etotal_i=Etotal_i,
            epsilon_tr_f=epsilon_tr_f,
            delta_E=delta_E,
        )
        for C in grid
    ])
    finite = np.isfinite(residuals)
    if not np.any(finite):
        return math.nan, {"status": "all_residuals_nonfinite"}

    idx_exact = np.where(finite & (np.abs(residuals) <= tol))[0]
    if idx_exact.size:
        i = int(idx_exact[0])
        return float(grid[i]), {
            "status": "ok",
            "residual": float(residuals[i]),
            "bracket_low": float(grid[i]),
            "bracket_high": float(grid[i]),
        }

    bracket = None
    for i in range(grid.size - 1):
        if not (finite[i] and finite[i + 1]):
            continue
        if residuals[i] * residuals[i + 1] < 0.0:
            bracket = (float(grid[i]), float(grid[i + 1]))
            break
    if bracket is None:
        i = int(np.nanargmin(np.abs(residuals)))
        return float(grid[i]), {
            "status": "no_bracket_min_abs_residual",
            "residual": float(residuals[i]),
            "bracket_low": float(C_min),
            "bracket_high": float(C_max),
        }

    lo, hi = bracket
    f_lo = fixed_point_residual(
        lo, theta=theta, Etr_i=Etr_i, Erot_i=Erot_i, Etotal_i=Etotal_i,
        epsilon_tr_f=epsilon_tr_f, delta_E=delta_E,
    )
    f_hi = fixed_point_residual(
        hi, theta=theta, Etr_i=Etr_i, Erot_i=Erot_i, Etotal_i=Etotal_i,
        epsilon_tr_f=epsilon_tr_f, delta_E=delta_E,
    )
    mid = 0.5 * (lo + hi)
    f_mid = math.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = fixed_point_residual(
            mid, theta=theta, Etr_i=Etr_i, Erot_i=Erot_i, Etotal_i=Etotal_i,
            epsilon_tr_f=epsilon_tr_f, delta_E=delta_E,
        )
        if abs(f_mid) <= tol or abs(hi - lo) <= tol:
            break
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return float(mid), {
        "status": "ok",
        "residual": float(f_mid),
        "bracket_low": float(lo),
        "bracket_high": float(hi),
        "residual_low": float(f_lo),
        "residual_high": float(f_hi),
    }


def estimate_case_hcs_fixed_point_c0(
    case: CTCCase,
    *,
    models,
    theta_target: float,
    beta_a: float,
    beta_b: float,
    n_samples: int = 20000,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 12345,
    C_min: float = -10.0,
    C_max: float = 5.0,
) -> tuple[float, dict[str, object]]:
    theta = float(theta_target)
    B = float(mode_neutral_factor(theta))
    if case.alpha >= 1.0 - 1.0e-12:
        return 0.0, {
            "alpha": float(case.alpha),
            "AR": float(case.AR),
            "theta_used": theta,
            "B_theta": B,
            "case_dir": str(case.path),
            "status": "elastic_alpha_by_convention",
            "estimator": "hcs-fixed-point",
            "n_used": 0,
            "C0": 0.0,
            "residual": 0.0,
        }

    _, Etr_i, Erot_i, Etotal_i, epsilon_tr_i, epsilon_rot_1_i = (
        _sample_hcs_pair_ensemble(theta, int(n_samples), bootstrap_seed)
    )
    epsilon_tr_f = _sample_dsmc_exchange(
        models=models,
        alpha=case.alpha,
        theta=theta,
        Etotal=Etotal_i,
        epsilon_tr_i=epsilon_tr_i,
        epsilon_rot_1_i=epsilon_rot_1_i,
        seed=bootstrap_seed,
    )
    gamma_factor = (
        float(models.get_gamma_max(case.alpha, case.AR))
        * float(models.get_one_hit(case.alpha, case.AR))
        * float(beta_a) / (float(beta_a) + float(beta_b))
    )
    delta_E = gamma_factor * Etotal_i

    C0, solve_diag = solve_fixed_point_c0(
        theta=theta,
        Etr_i=Etr_i,
        Erot_i=Erot_i,
        Etotal_i=Etotal_i,
        epsilon_tr_f=epsilon_tr_f,
        delta_E=delta_E,
        C_min=C_min,
        C_max=C_max,
    )
    A = (epsilon_tr_f - epsilon_tr_i) * Etotal_i
    diagnostics: dict[str, object] = {
        "alpha": float(case.alpha),
        "AR": float(case.AR),
        "theta_used": theta,
        "B_theta": B,
        "case_dir": str(case.path),
        "status": solve_diag.get("status", "unknown"),
        "estimator": "hcs-fixed-point",
        "n_used": int(n_samples),
        "gamma_factor": float(gamma_factor),
        "mean_A": float(np.mean(A)),
        "mean_delta_E_closure": float(np.mean(delta_E)),
        "mean_Etr_i": float(np.mean(Etr_i)),
        "mean_Erot_i": float(np.mean(Erot_i)),
        "mean_epsilon_tr_i": float(np.mean(epsilon_tr_i)),
        "mean_epsilon_tr_f": float(np.mean(epsilon_tr_f)),
        "C0": float(C0),
        "negative_due_to_exchange": bool(C0 < 0.0),
    }
    diagnostics.update(solve_diag)
    if bootstrap_samples > 0:
        values = []
        rng = np.random.default_rng(bootstrap_seed + 1729)
        n = Etr_i.size
        for _ in range(bootstrap_samples):
            idx = rng.integers(0, n, size=n)
            value, _ = solve_fixed_point_c0(
                theta=theta,
                Etr_i=Etr_i[idx],
                Erot_i=Erot_i[idx],
                Etotal_i=Etotal_i[idx],
                epsilon_tr_f=epsilon_tr_f[idx],
                delta_E=delta_E[idx],
                C_min=C_min,
                C_max=C_max,
            )
            values.append(value)
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        diagnostics["bootstrap_samples"] = int(bootstrap_samples)
        diagnostics["C0_bootstrap_stderr"] = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        if values.size:
            lo, hi = np.percentile(values, [2.5, 97.5])
            diagnostics["C0_ci95_low"] = float(lo)
            diagnostics["C0_ci95_high"] = float(hi)
    else:
        diagnostics["bootstrap_samples"] = int(bootstrap_samples)
        diagnostics["C0_bootstrap_stderr"] = None
        diagnostics["C0_ci95_low"] = None
        diagnostics["C0_ci95_high"] = None
    return float(C0), diagnostics


def _sample_fixed_point_arrays(theta, case, models, beta_a, beta_b, n_samples, seed):
    _, Etr_i, Erot_i, Etotal_i, epsilon_tr_i, epsilon_rot_1_i = (
        _sample_hcs_pair_ensemble(theta, int(n_samples), seed)
    )
    epsilon_tr_f = _sample_dsmc_exchange(
        models=models,
        alpha=case.alpha,
        theta=theta,
        Etotal=Etotal_i,
        epsilon_tr_i=epsilon_tr_i,
        epsilon_rot_1_i=epsilon_rot_1_i,
        seed=seed,
    )
    gamma_factor = (
        float(models.get_gamma_max(case.alpha, case.AR))
        * float(models.get_one_hit(case.alpha, case.AR))
        * float(beta_a) / (float(beta_a) + float(beta_b))
    )
    delta_E = gamma_factor * Etotal_i
    return {
        "Etr_i": Etr_i,
        "Erot_i": Erot_i,
        "Etotal_i": Etotal_i,
        "epsilon_tr_i": epsilon_tr_i,
        "epsilon_tr_f": epsilon_tr_f,
        "delta_E": delta_E,
        "gamma_factor": gamma_factor,
    }


def _theta_residual(theta, C0, case, models, beta_a, beta_b, n_samples, seed):
    arrays = _sample_fixed_point_arrays(
        theta, case, models, beta_a, beta_b, n_samples, seed
    )
    residual = fixed_point_residual(
        C0,
        theta=theta,
        Etr_i=arrays["Etr_i"],
        Erot_i=arrays["Erot_i"],
        Etotal_i=arrays["Etotal_i"],
        epsilon_tr_f=arrays["epsilon_tr_f"],
        delta_E=arrays["delta_E"],
    )
    return residual, arrays


def _interpolate_grid_c0(theta, theta_grid, C0_grid):
    theta_grid = np.asarray(theta_grid, dtype=float)
    C0_grid = np.asarray(C0_grid, dtype=float)
    if theta_grid.size == 0:
        return math.nan
    if theta < theta_grid[0] or theta > theta_grid[-1]:
        return math.nan
    return float(np.interp(float(theta), theta_grid, C0_grid))


def _theta_grid_residual(
    theta,
    theta_grid,
    C0_grid,
    case,
    models,
    beta_a,
    beta_b,
    n_samples,
    seed,
):
    C0 = _interpolate_grid_c0(theta, theta_grid, C0_grid)
    if not np.isfinite(C0):
        return math.nan, None
    arrays = _sample_fixed_point_arrays(
        theta, case, models, beta_a, beta_b, n_samples, seed
    )
    residual = fixed_point_residual(
        C0,
        theta=theta,
        Etr_i=arrays["Etr_i"],
        Erot_i=arrays["Erot_i"],
        Etotal_i=arrays["Etotal_i"],
        epsilon_tr_f=arrays["epsilon_tr_f"],
        delta_E=arrays["delta_E"],
    )
    return residual, arrays


def solve_grid_self_consistent_theta(
    *,
    theta_grid: np.ndarray,
    C0_grid: np.ndarray,
    case: CTCCase,
    models,
    beta_a: float,
    beta_b: float,
    n_samples: int = 20000,
    seed: int = 12345,
    scan_points: int = 31,
    tol: float = 1.0e-5,
    max_iter: int = 60,
) -> tuple[float, dict[str, object]]:
    theta_grid = np.asarray(theta_grid, dtype=float)
    C0_grid = np.asarray(C0_grid, dtype=float)
    order = np.argsort(theta_grid)
    theta_grid = theta_grid[order]
    C0_grid = C0_grid[order]
    valid = np.isfinite(theta_grid) & np.isfinite(C0_grid)
    theta_grid = theta_grid[valid]
    C0_grid = C0_grid[valid]
    if theta_grid.size < 2:
        return math.nan, {"status": "insufficient_theta_grid"}

    theta_min = float(theta_grid[0])
    theta_max = float(theta_grid[-1])
    cache: dict[float, tuple[float, dict[str, np.ndarray] | None]] = {}

    def residual_at(theta):
        key = round(float(theta), 12)
        if key not in cache:
            cache[key] = _theta_grid_residual(
                float(theta),
                theta_grid,
                C0_grid,
                case,
                models,
                beta_a,
                beta_b,
                n_samples,
                seed,
            )
        return cache[key]

    grid = np.linspace(theta_min, theta_max, int(scan_points))
    residuals = np.empty_like(grid)
    for i, theta in enumerate(grid):
        residuals[i], _ = residual_at(theta)

    finite = np.isfinite(residuals)
    bracket = None
    for i in range(grid.size - 1):
        if finite[i] and finite[i + 1] and residuals[i] * residuals[i + 1] < 0.0:
            bracket = (float(grid[i]), float(grid[i + 1]))
            break
    if bracket is None:
        i = int(np.nanargmin(np.abs(residuals)))
        return float(grid[i]), {
            "status": "no_bracket_min_abs_residual",
            "theta_residual": float(residuals[i]),
            "theta_bracket_low": theta_min,
            "theta_bracket_high": theta_max,
            "theta_scan_min_residual_abs": float(abs(residuals[i])),
            "residual_evaluations": int(len(cache)),
        }

    lo, hi = bracket
    f_lo, _ = residual_at(lo)
    f_hi, _ = residual_at(hi)
    mid = 0.5 * (lo + hi)
    f_mid = math.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid, _ = residual_at(mid)
        if abs(f_mid) <= tol or abs(hi - lo) <= tol:
            break
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return float(mid), {
        "status": "ok",
        "theta_residual": float(f_mid),
        "theta_bracket_low": float(lo),
        "theta_bracket_high": float(hi),
        "theta_residual_low": float(f_lo),
        "theta_residual_high": float(f_hi),
        "residual_evaluations": int(len(cache)),
    }


def solve_theta_from_ctc_residual(theta_grid, residual_grid):
    theta_grid = np.asarray(theta_grid, dtype=float)
    residual_grid = np.asarray(residual_grid, dtype=float)
    order = np.argsort(theta_grid)
    theta_grid = theta_grid[order]
    residual_grid = residual_grid[order]
    finite = np.isfinite(theta_grid) & np.isfinite(residual_grid)
    theta_grid = theta_grid[finite]
    residual_grid = residual_grid[finite]
    if theta_grid.size < 2:
        return math.nan, {"status": "insufficient_theta_grid"}

    for i in range(theta_grid.size - 1):
        r0 = residual_grid[i]
        r1 = residual_grid[i + 1]
        if r0 == 0.0:
            return float(theta_grid[i]), {
                "status": "ok",
                "theta_bracket_low": float(theta_grid[i]),
                "theta_bracket_high": float(theta_grid[i]),
                "theta_residual": 0.0,
            }
        if r0 * r1 < 0.0:
            theta = theta_grid[i] - r0 * (
                theta_grid[i + 1] - theta_grid[i]
            ) / (r1 - r0)
            return float(theta), {
                "status": "ok",
                "theta_bracket_low": float(theta_grid[i]),
                "theta_bracket_high": float(theta_grid[i + 1]),
                "theta_residual": 0.0,
                "theta_residual_low": float(r0),
                "theta_residual_high": float(r1),
            }

    idx = int(np.nanargmin(np.abs(residual_grid)))
    return float(theta_grid[idx]), {
        "status": "no_bracket_min_abs_residual",
        "theta_bracket_low": float(theta_grid[0]),
        "theta_bracket_high": float(theta_grid[-1]),
        "theta_residual": float(residual_grid[idx]),
    }


def solve_self_consistent_theta(
    *,
    C0: float,
    case: CTCCase,
    models,
    beta_a: float,
    beta_b: float,
    n_samples: int = 20000,
    seed: int = 12345,
    theta_min: float = 0.2,
    theta_max: float = 1.5,
    scan_points: int = 80,
    tol: float = 1.0e-5,
    max_iter: int = 60,
) -> tuple[float, dict[str, object]]:
    grid = np.linspace(float(theta_min), float(theta_max), int(scan_points))
    residuals = np.empty_like(grid)
    for i, theta in enumerate(grid):
        residuals[i], _ = _theta_residual(
            theta, C0, case, models, beta_a, beta_b, n_samples, seed
        )

    finite = np.isfinite(residuals)
    bracket = None
    for i in range(grid.size - 1):
        if finite[i] and finite[i + 1] and residuals[i] * residuals[i + 1] < 0.0:
            bracket = (float(grid[i]), float(grid[i + 1]))
            break
    if bracket is None:
        i = int(np.nanargmin(np.abs(residuals)))
        return float(grid[i]), {
            "status": "no_bracket_min_abs_residual",
            "theta_residual": float(residuals[i]),
            "theta_bracket_low": float(theta_min),
            "theta_bracket_high": float(theta_max),
        }

    lo, hi = bracket
    f_lo, _ = _theta_residual(
        lo, C0, case, models, beta_a, beta_b, n_samples, seed
    )
    f_hi, _ = _theta_residual(
        hi, C0, case, models, beta_a, beta_b, n_samples, seed
    )
    mid = 0.5 * (lo + hi)
    f_mid = math.nan
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid, _ = _theta_residual(
            mid, C0, case, models, beta_a, beta_b, n_samples, seed
        )
        if abs(f_mid) <= tol or abs(hi - lo) <= tol:
            break
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid

    return float(mid), {
        "status": "ok",
        "theta_residual": float(f_mid),
        "theta_bracket_low": float(lo),
        "theta_bracket_high": float(hi),
        "theta_residual_low": float(f_lo),
        "theta_residual_high": float(f_hi),
    }


def estimate_case_hcs_self_consistent_c0(
    case: CTCCase,
    *,
    C_mic: float,
    models,
    beta_a: float,
    beta_b: float,
    n_samples: int = 20000,
    bootstrap_seed: int = 12345,
    theta_min: float = 0.2,
    theta_max: float = 1.5,
    theta_scan_points: int = 28,
) -> tuple[float, dict[str, object]]:
    if case.alpha >= 1.0 - 1.0e-12:
        return 0.0, {
            "alpha": float(case.alpha),
            "AR": float(case.AR),
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "case_dir": str(case.path),
            "status": "elastic_alpha_by_convention",
            "estimator": "hcs-self-consistent",
            "n_used": 0,
            "C0": 0.0,
            "comparison_C_mic": float(C_mic),
        }

    theta, solve_diag = solve_self_consistent_theta(
        C0=float(C_mic),
        case=case,
        models=models,
        beta_a=beta_a,
        beta_b=beta_b,
        n_samples=n_samples,
        seed=bootstrap_seed,
        theta_min=theta_min,
        theta_max=theta_max,
        scan_points=theta_scan_points,
    )
    residual, arrays = _theta_residual(
        theta, float(C_mic), case, models, beta_a, beta_b, n_samples,
        bootstrap_seed
    )
    diagnostics: dict[str, object] = {
        "alpha": float(case.alpha),
        "AR": float(case.AR),
        "theta_pred": float(theta),
        "theta_used": float(theta),
        "B_theta": float(mode_neutral_factor(theta)),
        "case_dir": str(case.path),
        "status": solve_diag.get("status", "unknown"),
        "estimator": "hcs-self-consistent",
        "n_used": int(n_samples),
        "theta_scan_points": int(theta_scan_points),
        "C0": float(C_mic),
        "comparison_C_mic": float(C_mic),
        "theta_residual": float(residual),
        "gamma_factor": float(arrays["gamma_factor"]),
        "mean_Etr_i": float(np.mean(arrays["Etr_i"])),
        "mean_Erot_i": float(np.mean(arrays["Erot_i"])),
        "mean_epsilon_tr_i": float(np.mean(arrays["epsilon_tr_i"])),
        "mean_epsilon_tr_f": float(np.mean(arrays["epsilon_tr_f"])),
    }
    diagnostics.update(solve_diag)
    return float(C_mic), diagnostics


def estimate_group_hcs_grid_self_consistent_c0(
    cases: list[CTCCase],
    *,
    models,
    beta_a: float,
    beta_b: float,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
    n_samples: int = 20000,
    bootstrap_seed: int = 12345,
    theta_scan_points: int = 31,
) -> tuple[float, dict[str, object]]:
    if not cases:
        raise ValueError("No CTC cases supplied for grid self-consistent estimator")

    cases = sorted(cases, key=lambda item: item.theta)
    alpha = float(cases[0].alpha)
    AR = float(cases[0].AR)
    representative = cases[len(cases) // 2]

    grid_rows = []
    theta_values = []
    C_values = []
    for case in cases:
        C_mic, diag = estimate_case_c0_fast_stats(
            case,
            weight_mode=weight_mode,
            single_hit_only=single_hit_only,
            min_delta_E=min_delta_E,
        )
        grid_rows.append(diag)
        if np.isfinite(C_mic) and diag.get("status") in (
            "ok",
            "elastic_alpha_by_convention",
        ):
            theta_values.append(float(case.theta))
            C_values.append(float(C_mic))

    theta_grid = np.asarray(theta_values, dtype=float)
    C0_grid = np.asarray(C_values, dtype=float)
    order = np.argsort(theta_grid)
    theta_grid = theta_grid[order]
    C0_grid = C0_grid[order]

    if alpha >= 1.0 - 1.0e-12:
        return 0.0, {
            "alpha": alpha,
            "AR": AR,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "status": "elastic_alpha_by_convention",
            "estimator": "hcs-grid-self-consistent",
            "n_used": 0,
            "theta_grid_min": float(theta_grid[0]) if theta_grid.size else math.nan,
            "theta_grid_max": float(theta_grid[-1]) if theta_grid.size else math.nan,
            "theta_grid_count": int(theta_grid.size),
            "theta_grid_values": [float(x) for x in theta_grid],
            "C_mic_grid_values": [float(x) for x in C0_grid],
            "theta_grid_rows": grid_rows,
            "C0": 0.0,
        }

    if theta_grid.size < 2:
        return math.nan, {
            "alpha": alpha,
            "AR": AR,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "status": "insufficient_theta_grid",
            "estimator": "hcs-grid-self-consistent",
            "n_used": 0,
            "theta_grid_min": float(theta_grid[0]) if theta_grid.size else math.nan,
            "theta_grid_max": float(theta_grid[-1]) if theta_grid.size else math.nan,
            "theta_grid_count": int(theta_grid.size),
            "theta_grid_values": [float(x) for x in theta_grid],
            "C_mic_grid_values": [float(x) for x in C0_grid],
            "theta_grid_rows": grid_rows,
            "C0": math.nan,
        }

    theta, solve_diag = solve_grid_self_consistent_theta(
        theta_grid=theta_grid,
        C0_grid=C0_grid,
        case=representative,
        models=models,
        beta_a=beta_a,
        beta_b=beta_b,
        n_samples=n_samples,
        seed=bootstrap_seed,
        scan_points=theta_scan_points,
    )
    C0 = _interpolate_grid_c0(theta, theta_grid, C0_grid)
    residual, arrays = _theta_grid_residual(
        theta,
        theta_grid,
        C0_grid,
        representative,
        models,
        beta_a,
        beta_b,
        n_samples,
        bootstrap_seed,
    )
    if arrays is None:
        arrays = {
            "gamma_factor": math.nan,
            "Etr_i": np.array([], dtype=float),
            "Erot_i": np.array([], dtype=float),
            "epsilon_tr_i": np.array([], dtype=float),
            "epsilon_tr_f": np.array([], dtype=float),
        }

    diagnostics: dict[str, object] = {
        "alpha": alpha,
        "AR": AR,
        "theta_pred": float(theta),
        "theta_used": float(theta),
        "B_theta": float(mode_neutral_factor(theta)),
        "status": solve_diag.get("status", "unknown"),
        "estimator": "hcs-grid-self-consistent",
        "n_used": int(n_samples),
        "theta_scan_points": int(theta_scan_points),
        "theta_grid_min": float(theta_grid[0]),
        "theta_grid_max": float(theta_grid[-1]),
        "theta_grid_count": int(theta_grid.size),
        "theta_grid_values": [float(x) for x in theta_grid],
        "C_mic_grid_values": [float(x) for x in C0_grid],
        "theta_grid_n_used_values": [int(row["n_used"]) for row in grid_rows],
        "theta_grid_numerator_values": [float(row["numerator"]) for row in grid_rows],
        "theta_grid_denominator_values": [float(row["denominator"]) for row in grid_rows],
        "theta_grid_rows": grid_rows,
        "C0": float(C0),
        "theta_residual": float(residual),
        "gamma_factor": float(arrays["gamma_factor"]),
        "mean_Etr_i": (
            float(np.mean(arrays["Etr_i"])) if arrays["Etr_i"].size else math.nan
        ),
        "mean_Erot_i": (
            float(np.mean(arrays["Erot_i"])) if arrays["Erot_i"].size else math.nan
        ),
        "mean_epsilon_tr_i": (
            float(np.mean(arrays["epsilon_tr_i"]))
            if arrays["epsilon_tr_i"].size else math.nan
        ),
        "mean_epsilon_tr_f": (
            float(np.mean(arrays["epsilon_tr_f"]))
            if arrays["epsilon_tr_f"].size else math.nan
        ),
    }
    diagnostics.update(solve_diag)
    return float(C0), diagnostics


def estimate_group_hcs_grid_scaled_c0(
    cases: list[CTCCase],
    *,
    models,
    beta_a: float,
    beta_b: float,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
    n_samples: int = 20000,
    bootstrap_seed: int = 12345,
    theta_scan_points: int = 31,
    C_min: float = -10.0,
    C_max: float = 10.0,
) -> tuple[float, dict[str, object]]:
    """Project the CTC grid onto the DSMC effective HCS fixed-point branch.

    For each theta, compute the microscopic CTC curve C_mic(theta) and the
    DSMC effective fixed-point coefficient C_eff(theta).  Select the theta
    where the scale lambda=C_eff/C_mic is closest to unity, preferring properly
    bracketed fixed-point solves over boundary/min-residual fallbacks.
    """
    if not cases:
        raise ValueError("No CTC cases supplied for grid scaled estimator")

    cases = sorted(cases, key=lambda item: item.theta)
    alpha = float(cases[0].alpha)
    AR = float(cases[0].AR)
    representative = cases[len(cases) // 2]

    grid_rows = []
    theta_values = []
    C_values = []
    for case in cases:
        C_mic, diag = estimate_case_c0_fast_stats(
            case,
            weight_mode=weight_mode,
            single_hit_only=single_hit_only,
            min_delta_E=min_delta_E,
        )
        grid_rows.append(diag)
        if np.isfinite(C_mic) and diag.get("status") in (
            "ok",
            "elastic_alpha_by_convention",
        ):
            theta_values.append(float(case.theta))
            C_values.append(float(C_mic))

    theta_grid = np.asarray(theta_values, dtype=float)
    C0_grid = np.asarray(C_values, dtype=float)
    order = np.argsort(theta_grid)
    theta_grid = theta_grid[order]
    C0_grid = C0_grid[order]

    base_diag: dict[str, object] = {
        "alpha": alpha,
        "AR": AR,
        "estimator": "hcs-grid-scaled-self-consistent",
        "theta_grid_min": float(theta_grid[0]) if theta_grid.size else math.nan,
        "theta_grid_max": float(theta_grid[-1]) if theta_grid.size else math.nan,
        "theta_grid_count": int(theta_grid.size),
        "theta_grid_values": [float(x) for x in theta_grid],
        "C_mic_grid_values": [float(x) for x in C0_grid],
        "theta_grid_n_used_values": [int(row["n_used"]) for row in grid_rows],
        "theta_grid_numerator_values": [float(row["numerator"]) for row in grid_rows],
        "theta_grid_denominator_values": [float(row["denominator"]) for row in grid_rows],
        "theta_grid_rows": grid_rows,
    }

    if alpha >= 1.0 - 1.0e-12:
        diag = {
            **base_diag,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "status": "elastic_alpha_by_convention",
            "n_used": 0,
            "C0": 0.0,
            "lambda_scale": math.nan,
        }
        return 0.0, diag

    if theta_grid.size < 2:
        diag = {
            **base_diag,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "status": "insufficient_theta_grid",
            "n_used": 0,
            "C0": math.nan,
            "lambda_scale": math.nan,
        }
        return math.nan, diag

    scan = np.linspace(float(theta_grid[0]), float(theta_grid[-1]), int(theta_scan_points))
    scan_rows = []
    best = None
    best_fallback = None
    for idx, theta in enumerate(scan):
        C_mic = _interpolate_grid_c0(theta, theta_grid, C0_grid)
        if not np.isfinite(C_mic) or abs(C_mic) <= 1.0e-30:
            continue
        arrays = _sample_fixed_point_arrays(
            theta,
            representative,
            models,
            beta_a,
            beta_b,
            int(n_samples),
            bootstrap_seed + idx,
        )
        C_eff, solve_diag = solve_fixed_point_c0(
            theta=theta,
            Etr_i=arrays["Etr_i"],
            Erot_i=arrays["Erot_i"],
            Etotal_i=arrays["Etotal_i"],
            epsilon_tr_f=arrays["epsilon_tr_f"],
            delta_E=arrays["delta_E"],
            C_min=C_min,
            C_max=C_max,
        )
        lambda_scale = float(C_eff / C_mic) if np.isfinite(C_eff) else math.nan
        objective = (
            abs(math.log(abs(lambda_scale)))
            if np.isfinite(lambda_scale) and lambda_scale > 0.0
            else math.inf
        )
        row = {
            "theta": float(theta),
            "C_mic": float(C_mic),
            "C_eff": float(C_eff),
            "lambda_scale": float(lambda_scale),
            "objective": float(objective),
            "fixed_point_status": solve_diag.get("status", "unknown"),
            "fixed_point_residual": float(solve_diag.get("residual", math.nan)),
        }
        scan_rows.append(row)
        candidate = (objective, abs(float(solve_diag.get("residual", math.inf))), row, arrays)
        if np.isfinite(objective):
            if solve_diag.get("status") == "ok":
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
            if best_fallback is None or candidate[:2] < best_fallback[:2]:
                best_fallback = candidate

    chosen = best if best is not None else best_fallback
    if chosen is None:
        diag = {
            **base_diag,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "status": "no_valid_scaled_candidates",
            "n_used": int(n_samples),
            "C0": math.nan,
            "lambda_scale": math.nan,
            "scaled_scan_rows": scan_rows,
        }
        return math.nan, diag

    _, _, selected, arrays = chosen
    status = (
        "ok"
        if selected["fixed_point_status"] == "ok"
        else "no_bracket_min_scaled_objective"
    )
    diagnostics: dict[str, object] = {
        **base_diag,
        "theta_pred": float(selected["theta"]),
        "theta_used": float(selected["theta"]),
        "B_theta": float(mode_neutral_factor(selected["theta"])),
        "status": status,
        "n_used": int(n_samples),
        "theta_scan_points": int(theta_scan_points),
        "C0": float(selected["C_eff"]),
        "C_mic_at_theta": float(selected["C_mic"]),
        "lambda_scale": float(selected["lambda_scale"]),
        "scaled_objective": float(selected["objective"]),
        "theta_residual": float(selected["fixed_point_residual"]),
        "fixed_point_status": selected["fixed_point_status"],
        "scaled_scan_rows": scan_rows,
        "gamma_factor": float(arrays["gamma_factor"]),
        "mean_Etr_i": float(np.mean(arrays["Etr_i"])),
        "mean_Erot_i": float(np.mean(arrays["Erot_i"])),
        "mean_epsilon_tr_i": float(np.mean(arrays["epsilon_tr_i"])),
        "mean_epsilon_tr_f": float(np.mean(arrays["epsilon_tr_f"])),
    }
    return float(selected["C_eff"]), diagnostics


def estimate_group_hcs_grid_ctc_balance_c0(
    cases: list[CTCCase],
    *,
    models,
    beta_a: float,
    beta_b: float,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
    n_samples: int = 20000,
    bootstrap_seed: int = 12345,
    C_min: float = -10.0,
    C_max: float = 10.0,
) -> tuple[float, dict[str, object]]:
    """Use CTC reservoir balance for theta and DSMC fixed point for C0."""
    if not cases:
        raise ValueError("No CTC cases supplied for grid CTC-balance estimator")

    cases = sorted(cases, key=lambda item: item.theta)
    alpha = float(cases[0].alpha)
    AR = float(cases[0].AR)
    representative = cases[len(cases) // 2]

    grid_rows = []
    ctc_rows = []
    theta_values = []
    C_values = []
    residual_values = []
    for case in cases:
        C_mic, diag = estimate_case_c0_fast_stats(
            case,
            weight_mode=weight_mode,
            single_hit_only=single_hit_only,
            min_delta_E=min_delta_E,
        )
        ctc_diag = estimate_case_ctc_reservoir_stats(
            case,
            weight_mode=weight_mode,
            single_hit_only=single_hit_only,
            min_delta_E=min_delta_E,
        )
        grid_rows.append(diag)
        ctc_rows.append(ctc_diag)
        if (
            np.isfinite(C_mic)
            and np.isfinite(ctc_diag["ctc_theta_residual"])
            and diag.get("status") in ("ok", "elastic_alpha_by_convention")
            and ctc_diag.get("status") == "ok"
        ):
            theta_values.append(float(case.theta))
            C_values.append(float(C_mic))
            residual_values.append(float(ctc_diag["ctc_theta_residual"]))

    theta_grid = np.asarray(theta_values, dtype=float)
    C0_grid = np.asarray(C_values, dtype=float)
    residual_grid = np.asarray(residual_values, dtype=float)
    order = np.argsort(theta_grid)
    theta_grid = theta_grid[order]
    C0_grid = C0_grid[order]
    residual_grid = residual_grid[order]

    base_diag: dict[str, object] = {
        "alpha": alpha,
        "AR": AR,
        "estimator": "hcs-grid-ctc-balance",
        "theta_grid_min": float(theta_grid[0]) if theta_grid.size else math.nan,
        "theta_grid_max": float(theta_grid[-1]) if theta_grid.size else math.nan,
        "theta_grid_count": int(theta_grid.size),
        "theta_grid_values": [float(x) for x in theta_grid],
        "C_mic_grid_values": [float(x) for x in C0_grid],
        "ctc_theta_residual_values": [float(x) for x in residual_grid],
        "theta_grid_n_used_values": [int(row["n_used"]) for row in grid_rows],
        "theta_grid_numerator_values": [float(row["numerator"]) for row in grid_rows],
        "theta_grid_denominator_values": [float(row["denominator"]) for row in grid_rows],
        "theta_grid_rows": grid_rows,
        "ctc_reservoir_rows": ctc_rows,
    }

    if alpha >= 1.0 - 1.0e-12:
        diag = {
            **base_diag,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "status": "elastic_alpha_by_convention",
            "n_used": 0,
            "C0": 0.0,
            "C_mic_at_theta": math.nan,
            "lambda_scale": math.nan,
        }
        return 0.0, diag

    theta, theta_diag = solve_theta_from_ctc_residual(theta_grid, residual_grid)
    if not np.isfinite(theta):
        diag = {
            **base_diag,
            **theta_diag,
            "theta_pred": math.nan,
            "theta_used": math.nan,
            "B_theta": math.nan,
            "n_used": int(n_samples),
            "C0": math.nan,
            "C_mic_at_theta": math.nan,
            "lambda_scale": math.nan,
        }
        return math.nan, diag

    C_mic = _interpolate_grid_c0(theta, theta_grid, C0_grid)
    arrays = _sample_fixed_point_arrays(
        theta,
        representative,
        models,
        beta_a,
        beta_b,
        int(n_samples),
        bootstrap_seed,
    )
    C_eff, solve_diag = solve_fixed_point_c0(
        theta=theta,
        Etr_i=arrays["Etr_i"],
        Erot_i=arrays["Erot_i"],
        Etotal_i=arrays["Etotal_i"],
        epsilon_tr_f=arrays["epsilon_tr_f"],
        delta_E=arrays["delta_E"],
        C_min=C_min,
        C_max=C_max,
    )
    lambda_scale = float(C_eff / C_mic) if np.isfinite(C_eff) and C_mic else math.nan
    diagnostics: dict[str, object] = {
        **base_diag,
        "theta_pred": float(theta),
        "theta_used": float(theta),
        "B_theta": float(mode_neutral_factor(theta)),
        "status": solve_diag.get("status", "unknown"),
        "theta_solve_status": theta_diag.get("status", "unknown"),
        "n_used": int(n_samples),
        "C0": float(C_eff),
        "C_mic_at_theta": float(C_mic),
        "lambda_scale": float(lambda_scale),
        "theta_residual": float(solve_diag.get("residual", math.nan)),
        "ctc_theta_residual": float(theta_diag.get("theta_residual", math.nan)),
        "gamma_factor": float(arrays["gamma_factor"]),
        "mean_Etr_i": float(np.mean(arrays["Etr_i"])),
        "mean_Erot_i": float(np.mean(arrays["Erot_i"])),
        "mean_epsilon_tr_i": float(np.mean(arrays["epsilon_tr_i"])),
        "mean_epsilon_tr_f": float(np.mean(arrays["epsilon_tr_f"])),
    }
    diagnostics.update({f"ctc_{k}": v for k, v in theta_diag.items()})
    diagnostics.update({f"fixed_point_{k}": v for k, v in solve_diag.items()})
    return float(C_eff), diagnostics


def _bootstrap_hcs_balance(
    A: np.ndarray,
    delta_E: np.ndarray,
    theta: float,
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    if samples <= 0 or A.size < 2:
        return {
            "bootstrap_samples": int(samples),
            "C0_bootstrap_stderr": None,
            "C0_ci95_low": None,
            "C0_ci95_high": None,
        }
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    n = A.size
    for i in range(samples):
        idx = rng.integers(0, n, size=n)
        values[i] = hcs_balance_coefficient(
            float(np.mean(A[idx])),
            float(np.mean(delta_E[idx])),
            theta,
        )
    values = values[np.isfinite(values)]
    if values.size == 0:
        stderr = low = high = None
    else:
        stderr = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        low, high = [float(x) for x in np.percentile(values, [2.5, 97.5])]
    return {
        "bootstrap_samples": int(samples),
        "C0_bootstrap_stderr": stderr,
        "C0_ci95_low": low,
        "C0_ci95_high": high,
    }


def load_flat_table(path: str | Path | None) -> dict[str, float]:
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def _group_cases_by_alpha_ar(cases: Iterable[CTCCase]):
    groups: dict[tuple[float, float], list[CTCCase]] = {}
    for case in cases:
        key = (float(case.alpha), float(case.AR))
        groups.setdefault(key, []).append(case)
    return [
        group
        for _, group in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0]))
    ]


def _regularize_alpha_branch(
    rows: list[dict[str, object]],
    table: dict[str, float],
    *,
    strength: float = 10.0,
) -> None:
    candidates = [
        row
        for row in rows
        if row.get("status") == "ok"
        and np.isfinite(row.get("C0", math.nan))
        and float(row.get("alpha", 1.0)) < 1.0 - 1.0e-12
    ]
    candidates.sort(key=lambda row: float(row["alpha"]))
    n = len(candidates)
    if n < 3:
        for row in rows:
            row["regularization_status"] = "insufficient_alpha_rows"
        return

    raw = np.array([float(row["C0"]) for row in candidates], dtype=float)
    A = np.eye(n, dtype=float)
    if n >= 3 and strength > 0.0:
        D = np.zeros((n - 2, n), dtype=float)
        for i in range(n - 2):
            D[i, i : i + 3] = [1.0, -2.0, 1.0]
        A = A + float(strength) * (D.T @ D)
    smooth = np.linalg.solve(A, raw)

    for row, raw_value, smooth_value in zip(candidates, raw, smooth):
        key = table_key(float(row["alpha"]), float(row["AR"]))
        table[key] = float(smooth_value)
        row["C0_raw"] = float(raw_value)
        row["C0"] = float(smooth_value)
        row["regularization_strength"] = float(strength)
        row["regularization_delta"] = float(smooth_value - raw_value)
        row["regularization_status"] = "ok"
        if "comparison_C_alpha" in row:
            old = float(row["comparison_C_alpha"])
            row["comparison_delta"] = float(smooth_value - old)
            row["comparison_ratio"] = (
                float(smooth_value / old) if old != 0.0 else math.nan
            )
        if "comparison_C_fixed_point" in row:
            fixed = float(row["comparison_C_fixed_point"])
            row["comparison_delta_fixed_point"] = float(smooth_value - fixed)

    for row in rows:
        if row.get("regularization_status") is None:
            row["regularization_status"] = (
                "elastic_alpha_by_convention"
                if float(row.get("alpha", 0.0)) >= 1.0 - 1.0e-12
                else "not_regularized"
            )


def build_c0_table(
    root: str | Path = DEFAULT_CTC_ROOT,
    *,
    AR: float | None = None,
    alphas: Iterable[float] | None = None,
    weight_mode: str = "area",
    single_hit_only: bool = True,
    min_delta_E: float = 1.0e-12,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 12345,
    compare_table: str | Path | None = None,
    estimator: str = "microscopic-ftr",
    models=None,
    theta_table: dict[str, float] | None = None,
    beta_a: float = 1.21,
    beta_b: float = 3.67,
    microscopic_table: str | Path | None = None,
    fixed_point_table: str | Path | None = None,
    fixed_point_samples: int = 20000,
    theta_scan_points: int = 28,
    alpha_regularization_strength: float = 10.0,
) -> dict[str, object]:
    cases = discover_cases(root, AR=AR, alphas=alphas)
    if not cases:
        raise FileNotFoundError(f"No CTC cases found in {root}")

    if estimator not in (
        "microscopic-ftr",
        "hcs-balance",
        "hcs-fixed-point",
        "hcs-self-consistent",
        "hcs-grid-self-consistent",
        "hcs-grid-scaled-self-consistent",
        "hcs-grid-ctc-balance",
        "hcs-grid-ctc-balance-regularized",
    ):
        raise ValueError(
            "estimator must be 'microscopic-ftr', 'hcs-balance', "
            "'hcs-fixed-point', 'hcs-self-consistent', or "
            "'hcs-grid-self-consistent', or "
            "'hcs-grid-scaled-self-consistent', 'hcs-grid-ctc-balance', or "
            "'hcs-grid-ctc-balance-regularized'"
        )
    if estimator in ("hcs-balance", "hcs-fixed-point") and (
        models is None or theta_table is None
    ):
        raise ValueError(f"{estimator} estimator requires models and theta_table")
    if estimator == "hcs-self-consistent" and models is None:
        raise ValueError("hcs-self-consistent estimator requires models")
    if estimator == "hcs-grid-self-consistent" and models is None:
        raise ValueError("hcs-grid-self-consistent estimator requires models")
    if estimator == "hcs-grid-scaled-self-consistent" and models is None:
        raise ValueError("hcs-grid-scaled-self-consistent estimator requires models")
    if estimator == "hcs-grid-ctc-balance" and models is None:
        raise ValueError("hcs-grid-ctc-balance estimator requires models")
    if estimator == "hcs-grid-ctc-balance-regularized" and models is None:
        raise ValueError("hcs-grid-ctc-balance-regularized estimator requires models")

    compare = load_flat_table(compare_table)
    microscopic = load_flat_table(microscopic_table)
    fixed_point = load_flat_table(fixed_point_table)
    table: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    iterable = (
        _group_cases_by_alpha_ar(cases)
        if estimator in (
            "hcs-grid-self-consistent",
            "hcs-grid-scaled-self-consistent",
            "hcs-grid-ctc-balance",
            "hcs-grid-ctc-balance-regularized",
        )
        else cases
    )
    for i, item in enumerate(iterable):
        case = (
            item[0]
            if estimator in ("hcs-grid-self-consistent", "hcs-grid-scaled-self-consistent")
            or estimator in (
                "hcs-grid-ctc-balance",
                "hcs-grid-ctc-balance-regularized",
            )
            else item
        )
        if estimator == "hcs-balance":
            theta_target = _load_theta_target(theta_table, case.alpha, case.AR)
            C0, diag = estimate_case_hcs_balance_c0(
                case,
                models=models,
                theta_target=theta_target,
                beta_a=beta_a,
                beta_b=beta_b,
                single_hit_only=single_hit_only,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + i,
                exchange_seed=bootstrap_seed + i,
            )
        elif estimator == "hcs-fixed-point":
            theta_target = _load_theta_target(theta_table, case.alpha, case.AR)
            C0, diag = estimate_case_hcs_fixed_point_c0(
                case,
                models=models,
                theta_target=theta_target,
                beta_a=beta_a,
                beta_b=beta_b,
                n_samples=int(fixed_point_samples),
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + i,
            )
        elif estimator == "hcs-self-consistent":
            key = table_key(case.alpha, case.AR)
            if key in microscopic:
                C_mic = float(microscopic[key])
            else:
                C_mic, _ = estimate_case_c0(
                    case,
                    weight_mode=weight_mode,
                    single_hit_only=single_hit_only,
                    min_delta_E=min_delta_E,
                    bootstrap_samples=0,
                    bootstrap_seed=bootstrap_seed + i,
                )
            C0, diag = estimate_case_hcs_self_consistent_c0(
                case,
                C_mic=C_mic,
                models=models,
                beta_a=beta_a,
                beta_b=beta_b,
                n_samples=int(fixed_point_samples),
                bootstrap_seed=bootstrap_seed + i,
                theta_scan_points=int(theta_scan_points),
            )
        elif estimator == "hcs-grid-self-consistent":
            C0, diag = estimate_group_hcs_grid_self_consistent_c0(
                item,
                models=models,
                beta_a=beta_a,
                beta_b=beta_b,
                weight_mode=weight_mode,
                single_hit_only=single_hit_only,
                min_delta_E=min_delta_E,
                n_samples=int(fixed_point_samples),
                bootstrap_seed=bootstrap_seed + i,
                theta_scan_points=int(theta_scan_points),
            )
        elif estimator == "hcs-grid-scaled-self-consistent":
            C0, diag = estimate_group_hcs_grid_scaled_c0(
                item,
                models=models,
                beta_a=beta_a,
                beta_b=beta_b,
                weight_mode=weight_mode,
                single_hit_only=single_hit_only,
                min_delta_E=min_delta_E,
                n_samples=int(fixed_point_samples),
                bootstrap_seed=bootstrap_seed + i,
                theta_scan_points=int(theta_scan_points),
            )
        elif estimator in ("hcs-grid-ctc-balance", "hcs-grid-ctc-balance-regularized"):
            C0, diag = estimate_group_hcs_grid_ctc_balance_c0(
                item,
                models=models,
                beta_a=beta_a,
                beta_b=beta_b,
                weight_mode=weight_mode,
                single_hit_only=single_hit_only,
                min_delta_E=min_delta_E,
                n_samples=int(fixed_point_samples),
                bootstrap_seed=bootstrap_seed + i,
            )
        else:
            C0, diag = estimate_case_c0(
                case,
                weight_mode=weight_mode,
                single_hit_only=single_hit_only,
                min_delta_E=min_delta_E,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed + i,
            )
            diag["estimator"] = "microscopic-ftr"
        key = table_key(case.alpha, case.AR)
        table[key] = float(C0)
        if key in compare:
            old = float(compare[key])
            diag["comparison_C_alpha"] = old
            diag["comparison_delta"] = float(C0 - old)
            diag["comparison_ratio"] = float(C0 / old) if old != 0.0 else math.nan
        if estimator in (
            "hcs-grid-self-consistent",
            "hcs-grid-scaled-self-consistent",
            "hcs-grid-ctc-balance",
            "hcs-grid-ctc-balance-regularized",
        ) and theta_table is not None:
            try:
                theta_old = _load_theta_target(theta_table, case.alpha, case.AR)
            except KeyError:
                theta_old = math.nan
            diag["comparison_theta_target"] = float(theta_old)
            if np.isfinite(theta_old) and np.isfinite(diag.get("theta_pred", math.nan)):
                diag["comparison_theta_delta"] = float(diag["theta_pred"] - theta_old)
        if estimator in (
            "hcs-grid-self-consistent",
            "hcs-grid-scaled-self-consistent",
            "hcs-grid-ctc-balance",
            "hcs-grid-ctc-balance-regularized",
        ) and key in fixed_point:
            fixed = float(fixed_point[key])
            diag["comparison_C_fixed_point"] = fixed
            diag["comparison_delta_fixed_point"] = float(C0 - fixed)
        if estimator in ("hcs-balance", "hcs-fixed-point") and key in microscopic:
            mic = float(microscopic[key])
            diag["comparison_C_mic"] = mic
            diag["comparison_delta_mic"] = float(C0 - mic)
        rows.append(diag)

    if estimator == "hcs-grid-ctc-balance-regularized":
        _regularize_alpha_branch(
            rows,
            table,
            strength=float(alpha_regularization_strength),
        )

    if estimator == "hcs-balance":
        model_name = HCS_BALANCE_MODEL_NAME
    elif estimator == "hcs-fixed-point":
        model_name = HCS_FIXED_POINT_MODEL_NAME
    elif estimator == "hcs-self-consistent":
        model_name = HCS_SELF_CONSISTENT_MODEL_NAME
    elif estimator == "hcs-grid-self-consistent":
        model_name = HCS_GRID_SELF_CONSISTENT_MODEL_NAME
    elif estimator == "hcs-grid-scaled-self-consistent":
        model_name = HCS_GRID_SCALED_MODEL_NAME
    elif estimator == "hcs-grid-ctc-balance":
        model_name = HCS_GRID_CTC_BALANCE_MODEL_NAME
    elif estimator == "hcs-grid-ctc-balance-regularized":
        model_name = HCS_GRID_CTC_BALANCE_REG_MODEL_NAME
    else:
        model_name = TABLE_MODEL_NAME
    formula = (
        "C0 = 1 + mean(A) / (B(theta*) * mean(delta_E_closure))"
        if estimator == "hcs-balance"
        else "C0 solves relative-cooling residual R(C0; theta*) = 0"
        if estimator == "hcs-fixed-point"
        else "theta solves R(C_mic; theta) = 0 using analytical DSMC pair ensemble"
        if estimator == "hcs-self-consistent"
        else "theta solves R(C_mic(theta); theta) = 0 using CTC theta-grid interpolation"
        if estimator == "hcs-grid-self-consistent"
        else "theta minimizes |log(C_eff(theta)/C_mic(theta))|; C_eff solves R(C; theta)=0"
        if estimator == "hcs-grid-scaled-self-consistent"
        else "theta solves CTC reservoir balance; C0 solves DSMC R(C0; theta)=0"
        if estimator == "hcs-grid-ctc-balance"
        else "hcs-grid-ctc-balance with penalized second-difference smoothing over alpha"
        if estimator == "hcs-grid-ctc-balance-regularized"
        else "C0 = sum(w * f_req * delta_E) / sum(w * B(theta) * delta_E)"
    )
    return {
        "table": table,
        "diagnostics": {
            "model": model_name,
            "root": str(root),
            "estimator": estimator,
            "formula": formula,
            "B(theta)": "3*theta/(3*theta+2)",
            "f_req_convention": "f_req = -ftr_data[:,0]",
            "theta_limitation": (
                "microscopic-ftr uses CTC r=theta; hcs-balance and "
                "hcs-fixed-point use theta_target_table theta*; "
                "hcs-grid-self-consistent solves theta from the CTC grid."
            ),
            "weight_mode": weight_mode,
            "single_hit_only": bool(single_hit_only),
            "beta_a": float(beta_a),
            "beta_b": float(beta_b),
            "fixed_point_samples": int(fixed_point_samples),
            "theta_scan_points": int(theta_scan_points),
            "alpha_regularization_strength": float(alpha_regularization_strength),
            "rows": rows,
        },
    }


def write_outputs(
    payload: dict[str, object],
    table_path: str | Path,
    diagnostics_json_path: str | Path | None = None,
    diagnostics_csv_path: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    table_path = Path(table_path)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_json_path = (
        Path(diagnostics_json_path)
        if diagnostics_json_path is not None
        else table_path.with_name(table_path.stem.replace("table", "diagnostics") + ".json")
    )
    diagnostics_csv_path = (
        Path(diagnostics_csv_path)
        if diagnostics_csv_path is not None
        else table_path.with_name(table_path.stem.replace("table", "diagnostics") + ".csv")
    )
    diagnostics_json_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(table_path, "w") as f:
        json.dump(payload["table"], f, indent=2, sort_keys=True)
        f.write("\n")
    with open(diagnostics_json_path, "w") as f:
        json.dump(payload["diagnostics"], f, indent=2, sort_keys=True)
        f.write("\n")

    rows = payload["diagnostics"]["rows"]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(diagnostics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return table_path, diagnostics_json_path, diagnostics_csv_path
