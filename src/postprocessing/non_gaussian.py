import glob
import json
import os

import numpy as np


SUMMARY_PATTERN = "*_ng_summary.json"


def load_non_gaussian_summary(path):
    """Load one per-realization non-Gaussian summary JSON."""
    with open(path, "r") as f:
        return json.load(f)


def load_non_gaussian_moments(path):
    """Load one *_ng_moments.txt file as a named column dictionary."""
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    new_cols = [
        "t", "tau", "Ttrans", "Trot", "theta", "a2_tr", "a3_tr",
        "a2_rot", "a11", "c2", "c4", "c6", "w2", "w4", "c2w2",
        "sample_index", "n_samples_particles",
    ]
    old_cols = [
        "t", "tau", "Ttrans", "Trot", "theta", "a2_tr", "a3_tr",
        "a2_rot", "a11", "c2", "c4", "c6", "w2", "w4", "c2w2",
        "n_samples_particles",
    ]
    cols = new_cols if data.shape[1] >= len(new_cols) else old_cols
    if data.shape[1] < len(cols):
        raise ValueError(
            f"Moment file has {data.shape[1]} columns, expected {len(cols)}: {path}"
        )
    return {name: data[:, i] for i, name in enumerate(cols)}


def load_histogram(path):
    """Load a two-column histogram file."""
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"Histogram file has fewer than two columns: {path}")
    return data[:, 0], data[:, 1]


def aggregate_non_gaussian_summaries(results_dir):
    """Aggregate all *_ng_summary.json files in a results directory.

    Returns mean and standard error for cumulants, moments, and collision
    frequency across available realizations.
    """
    paths = sorted(glob.glob(os.path.join(results_dir, SUMMARY_PATTERN)))
    if not paths:
        raise FileNotFoundError(f"No {SUMMARY_PATTERN} files found in {results_dir}")

    summaries = [load_non_gaussian_summary(path) for path in paths]

    def _mean_stderr(values):
        arr = np.array(list(values), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan, np.nan
        stderr = np.std(arr, ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        return float(np.mean(arr)), float(stderr)

    cumulant_names = ["a2_tr", "a3_tr", "a2_rot", "a11"]
    moment_names = ["c2", "c4", "c6", "w2", "w4", "c2w2"]
    result = {
        "n_realizations": len(summaries),
        "summary_paths": paths,
        "cumulants": {},
        "moments": {},
    }
    for name in cumulant_names:
        result["cumulants"][name] = dict(zip(
            ["mean", "stderr"],
            _mean_stderr(s["cumulants"].get(name) for s in summaries)
        ))
    for name in moment_names:
        result["moments"][name] = dict(zip(
            ["mean", "stderr"],
            _mean_stderr(s["moments"].get(name) for s in summaries)
        ))
    result["collision_frequency"] = dict(zip(
        ["mean", "stderr"],
        _mean_stderr(s.get("collision_frequency") for s in summaries)
    ))
    result["n_particle_samples"] = int(sum(
        s.get("n_particle_samples", 0) for s in summaries
    ))
    result["expected_output_samples"] = int(max(
        (s.get("expected_output_samples") or 0) for s in summaries
    ))
    result["complete_realizations"] = int(sum(
        bool(s.get("sampling_complete", False)) for s in summaries
    ))
    return result


def aggregate_histograms(results_dir, suffix):
    """Average matching per-realization histogram files by suffix."""
    paths = sorted(glob.glob(os.path.join(results_dir, f"*{suffix}")))
    if not paths:
        raise FileNotFoundError(f"No *{suffix} files found in {results_dir}")

    centers_ref = None
    densities = []
    for path in paths:
        centers, density = load_histogram(path)
        if not np.any(density > 0.0):
            continue
        if centers_ref is None:
            centers_ref = centers
        elif centers.shape != centers_ref.shape or not np.allclose(centers, centers_ref):
            raise ValueError(f"Histogram bin centers do not match: {path}")
        densities.append(density)
    if not densities:
        raise FileNotFoundError(f"No populated *{suffix} files found in {results_dir}")

    arr = np.vstack(densities)
    stderr = (
        np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    )
    return {
        "paths": paths,
        "centers": centers_ref,
        "density_mean": np.mean(arr, axis=0),
        "density_stderr": stderr,
    }


def maxwell_speed_pdf(c):
    """Reference PDF for c=|v|/sqrt(2T/m) in 3D."""
    return (4.0 / np.sqrt(np.pi)) * c * c * np.exp(-c * c)


def rayleigh_rot_speed_pdf(w):
    """Reference PDF for w = sqrt(w_y^2 + w_z^2), with <w^2> = 1."""
    return 2.0 * w * np.exp(-w * w)


def maxwell_energy_pdf(epsilon):
    """Reference PDF for epsilon=c^2 in 3D."""
    return (2.0 / np.sqrt(np.pi)) * np.sqrt(epsilon) * np.exp(-epsilon)


def exponential_energy_pdf(epsilon):
    """Reference PDF for epsilon_r=w^2 with two rotational DOF."""
    return np.exp(-epsilon)


def maxwell_energy_coupling_pdf(x):
    """Reference PDF for x=epsilon_t*epsilon_r.

    Here epsilon_t~Gamma(3/2,1) and epsilon_r~Gamma(1,1), independent.
    The product density reduces to 2 exp(-2 sqrt(x)).
    """
    return 2.0 * np.exp(-2.0 * np.sqrt(x))


def histogram_ratio_to_reference(hist, reference):
    """Return density/reference with zeros where the reference is unavailable."""
    ref = reference(hist["centers"])
    ratio = np.full_like(hist["density_mean"], np.nan, dtype=float)
    mask = ref > 0.0
    ratio[mask] = hist["density_mean"][mask] / ref[mask]
    out = dict(hist)
    out["reference"] = ref
    out["ratio"] = ratio
    return out


def aggregate_non_gaussian_results(results_dir):
    """Aggregate summaries plus standard paper histogram ratios for one case."""
    result = aggregate_non_gaussian_summaries(results_dir)
    speed = aggregate_histograms(results_dir, "_ng_hist_speed.txt")
    rot_speed = aggregate_histograms(results_dir, "_ng_hist_rot_speed.txt")
    energy_tr = aggregate_histograms(results_dir, "_ng_hist_energy_tr.txt")
    energy_rot = aggregate_histograms(results_dir, "_ng_hist_energy_rot.txt")
    energy_coupling = aggregate_histograms(
        results_dir, "_ng_hist_energy_coupling.txt"
    )
    result["histograms"] = {
        "speed": histogram_ratio_to_reference(speed, maxwell_speed_pdf),
        "rot_speed": histogram_ratio_to_reference(rot_speed, rayleigh_rot_speed_pdf),
        "energy_tr": histogram_ratio_to_reference(energy_tr, maxwell_energy_pdf),
        "energy_rot": histogram_ratio_to_reference(
            energy_rot, exponential_energy_pdf
        ),
        "energy_coupling": histogram_ratio_to_reference(
            energy_coupling, maxwell_energy_coupling_pdf
        ),
    }
    return result


def normalize_collision_frequency_by_sphere(case_results, sphere_key):
    """Add nu_over_sphere to a dict of aggregated case results.

    case_results is a mapping such as {(AR, alpha): aggregate_result}.
    sphere_key selects the reference case, typically (1.0, alpha).
    """
    nu_sphere = case_results[sphere_key]["collision_frequency"]["mean"]
    for result in case_results.values():
        nu = result["collision_frequency"]["mean"]
        result["collision_frequency"]["nu_over_sphere"] = (
            float(nu / nu_sphere) if nu_sphere else np.nan
        )
    return case_results


# ---------------------------------------------------------------------------
# Paper postprocessing extensions (HCS non-Gaussian campaign)
# ---------------------------------------------------------------------------

def load_base_timeseries(path):
    """Load base DSMC output (no header, 5 columns: t tau Ttrans Trot T_total)."""
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 5:
        raise ValueError(f"Expected >=5 columns in {path}, got {data.shape[1]}")
    return {
        "t": data[:, 0],
        "tau": data[:, 1],
        "Ttrans": data[:, 2],
        "Trot": data[:, 3],
        "T_total": data[:, 4],
    }


def _theta_from_base(ts):
    """Compute theta = Ttrans / Trot, returning nan where Trot==0."""
    Trot = ts["Trot"]
    theta = np.where(Trot > 0.0, ts["Ttrans"] / Trot, np.nan)
    return theta


def load_base_ensemble(results_dir, tau_max=120.0, n_grid=200):
    """Load theta(tau) across all seeds from base .txt files, tau in [0, tau_max].

    Returns tau_grid, theta_mean, theta_stderr (all shape (n_grid,)).
    Raises FileNotFoundError if no base files are found.
    """
    pattern = os.path.join(results_dir, "*.txt")
    paths = sorted(
        p for p in glob.glob(pattern)
        if "_ng_" not in os.path.basename(p) and "_pressure" not in os.path.basename(p)
    )
    if not paths:
        raise FileNotFoundError(f"No base .txt files found in {results_dir}")

    tau_grid = np.linspace(0.0, tau_max, n_grid)
    theta_traces = []
    for path in paths:
        try:
            ts = load_base_timeseries(path)
        except Exception:
            continue
        tau = ts["tau"]
        theta = _theta_from_base(ts)
        mask = np.isfinite(theta) & (tau <= tau_max * 1.05)
        if mask.sum() < 2:
            continue
        interp = np.interp(tau_grid, tau[mask], theta[mask],
                           left=np.nan, right=np.nan)
        theta_traces.append(interp)

    if not theta_traces:
        raise FileNotFoundError(f"No valid base timeseries data in {results_dir}")

    arr = np.vstack(theta_traces)
    mean = np.nanmean(arr, axis=0)
    n_valid = np.sum(np.isfinite(arr), axis=0)
    stderr = np.where(
        n_valid > 1,
        np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_valid),
        np.nan,
    )
    return tau_grid, mean, stderr


def load_theta_star_table(models_dir, ar_label):
    """Load theta_target_table_AR{ar_label}.json → dict {alpha: theta_star}.

    Keys in the JSON are "(alpha, AR)" strings; the AR part is ignored.
    """
    path = os.path.join(models_dir, f"theta_target_table_AR{ar_label}.json")
    with open(path) as f:
        raw = json.load(f)
    result = {}
    for key, val in raw.items():
        # key format: "(0.700, 2.0)"
        alpha_str = key.strip("()").split(",")[0].strip()
        result[round(float(alpha_str), 4)] = float(val)
    return result


def find_theta_divergence_mask(theta_col, ref_samples=10, diverge_factor=3.0,
                               theta_abs_max=5.0):
    """Return a boolean mask of pre-divergence valid samples.

    Reference theta = median of first ref_samples values.
    Divergence: theta > max(theta_ref * diverge_factor, theta_abs_max).
    Mask is True up to (not including) the first diverging index.
    All samples are valid if no divergence is detected.
    """
    theta = np.asarray(theta_col, dtype=float)
    if theta.size == 0:
        return np.ones(0, dtype=bool)
    n_ref = min(ref_samples, theta.size)
    theta_ref = float(np.nanmedian(theta[:n_ref]))
    threshold = max(theta_ref * diverge_factor, theta_abs_max)
    diverge_indices = np.where(theta > threshold)[0]
    if diverge_indices.size == 0:
        return np.ones(theta.size, dtype=bool)
    first_bad = int(diverge_indices[0])
    mask = np.zeros(theta.size, dtype=bool)
    mask[:first_bad] = True
    return mask


_CUMULANT_COLS = ["a2_tr", "a3_tr", "a2_rot", "a11"]
_MOMENT_COLS = ["c2", "c4", "c6", "w2", "w4", "c2w2"]


def aggregate_moments_timeseries(results_dir, diverge_factor=3.0, theta_abs_max=5.0,
                                 n_tau_grid=300):
    """Load ng_moments.txt per seed with theta-divergence masking.

    Returns a dict with keys:
      tau_grid   : 1D array, common tau axis (linear from first to last valid)
      per_col    : {col_name: {'mean': array, 'stderr': array}} for cumulant/moment cols
      n_valid_per_seed: 1D int array — valid sample count per seed
      n_seeds    : int
    Theta column is never returned (rescaling artifact).
    """
    pattern = os.path.join(results_dir, "*_ng_moments.txt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No *_ng_moments.txt files in {results_dir}")

    cols_of_interest = _CUMULANT_COLS + _MOMENT_COLS
    per_seed_avgs = {col: [] for col in cols_of_interest}
    per_seed_tau_traces = []
    per_seed_col_traces = {col: [] for col in cols_of_interest}
    n_valid_per_seed = []

    for path in paths:
        try:
            m = load_non_gaussian_moments(path)
        except Exception:
            continue
        theta = m.get("theta", np.ones(1) * np.nan)
        mask = find_theta_divergence_mask(
            theta, diverge_factor=diverge_factor, theta_abs_max=theta_abs_max
        )
        n_valid = int(mask.sum())
        n_valid_per_seed.append(n_valid)
        if n_valid == 0:
            continue
        tau_valid = m["tau"][mask]
        per_seed_tau_traces.append(tau_valid)
        for col in cols_of_interest:
            vals = m.get(col, np.full(mask.size, np.nan))
            per_seed_col_traces[col].append(vals[mask])
            per_seed_avgs[col].append(float(np.nanmean(vals[mask])))

    if not per_seed_tau_traces:
        raise FileNotFoundError(f"No valid (non-empty) ng_moments in {results_dir}")

    # Build a common tau grid spanning the typical valid range
    tau_min = float(np.median([t[0] for t in per_seed_tau_traces]))
    tau_max_val = float(np.median([t[-1] for t in per_seed_tau_traces]))
    tau_grid = np.linspace(tau_min, tau_max_val, n_tau_grid)

    per_col_interp = {col: [] for col in cols_of_interest}
    for i, tau_valid in enumerate(per_seed_tau_traces):
        for col in cols_of_interest:
            vals = per_seed_col_traces[col][i]
            if tau_valid.size < 2:
                interp = np.full(tau_grid.size, np.nan)
            else:
                interp = np.interp(tau_grid, tau_valid, vals,
                                   left=np.nan, right=np.nan)
            per_col_interp[col].append(interp)

    result_per_col = {}
    for col in cols_of_interest:
        arr = np.vstack(per_col_interp[col])
        mean = np.nanmean(arr, axis=0)
        n_v = np.sum(np.isfinite(arr), axis=0)
        stderr = np.where(
            n_v > 1,
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(n_v),
            np.nan,
        )
        result_per_col[col] = {"mean": mean, "stderr": stderr,
                                "traces": arr}

    return {
        "tau_grid": tau_grid,
        "per_col": result_per_col,
        "n_valid_per_seed": np.array(n_valid_per_seed, dtype=int),
        "n_seeds": len(n_valid_per_seed),
        "per_seed_avgs": per_seed_avgs,
    }


def aggregate_ng_summaries_from_moments(results_dir, diverge_factor=3.0,
                                        theta_abs_max=5.0):
    """Re-compute per-case cumulant/moment stats from theta-truncated ng_moments.

    Uses only samples before theta diverges (rescaling artifact) per seed.
    Returns the same structure as aggregate_non_gaussian_summaries() for
    cumulants, moments, and n_particle_samples so it can be used as a drop-in
    replacement for those fields.

    Also returns collision_frequency and complete_realizations from ng_summary
    (collision_frequency is unaffected by the rescaling artifact).
    """
    ts_result = aggregate_moments_timeseries(
        results_dir, diverge_factor=diverge_factor, theta_abs_max=theta_abs_max
    )
    avgs = ts_result["per_seed_avgs"]

    def _mean_stderr(vals):
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        stderr = np.std(arr, ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        return float(np.mean(arr)), float(stderr)

    cumulants = {}
    for col in _CUMULANT_COLS:
        cumulants[col] = dict(zip(["mean", "stderr"], _mean_stderr(avgs[col])))
    moments = {}
    for col in _MOMENT_COLS:
        moments[col] = dict(zip(["mean", "stderr"], _mean_stderr(avgs[col])))

    # Collision frequency and completion from ng_summary (unaffected by artifact)
    try:
        summary_result = aggregate_non_gaussian_summaries(results_dir)
    except FileNotFoundError:
        summary_result = {"collision_frequency": {"mean": np.nan, "stderr": np.nan},
                          "complete_realizations": 0, "n_realizations": 0,
                          "n_particle_samples": 0, "expected_output_samples": 0}

    return {
        "n_realizations": summary_result["n_realizations"],
        "complete_realizations": summary_result["complete_realizations"],
        "expected_output_samples": summary_result["expected_output_samples"],
        "n_particle_samples": int(sum(ts_result["n_valid_per_seed"])),
        "cumulants": cumulants,
        "moments": moments,
        "collision_frequency": summary_result["collision_frequency"],
        "n_valid_per_seed": ts_result["n_valid_per_seed"],
    }


def brey_a2_tr(alpha):
    """Brey first-Sonine prediction for a₂ᵗʳ of a smooth-sphere granular gas.

    Brey et al. (1998) Eq. 18, valid for d=3, α in [0,1].
    """
    alpha = np.asarray(alpha, dtype=float)
    num = 16.0 * (1.0 - alpha) * (1.0 - 2.0 * alpha ** 2)
    den = 81.0 - 17.0 * alpha + 30.0 * alpha ** 2 - 30.0 * alpha ** 3
    return num / den


def sonine_speed_ratio(c, a2_tr):
    """Second-Sonine correction ratio φ_c,S / φ_{c,M} (Megías-Santos Eq. 5.4a, d_t=3).

    φ_c,S / φ_{c,M} = 1 + a2_tr * S_2^{1/2}(c²)
    where S_2^{1/2}(c²) = (4c⁴ - 20c² + 15) / 8
    """
    c = np.asarray(c, dtype=float)
    return 1.0 + a2_tr * (4.0 * c ** 4 - 20.0 * c ** 2 + 15.0) / 8.0


def sonine_rot_speed_ratio(w, a2_rot):
    """Second-Sonine correction ratio φ_w,S / φ_{w,M} (Megías-Santos Eq. 5.4b, d_r=2).

    φ_w,S / φ_{w,M} = 1 + a2_rot * S_2^1(w²)
    where S_2^1(w²) = (4w⁴ - 16w² + 8) / 8
    """
    w = np.asarray(w, dtype=float)
    return 1.0 + a2_rot * (4.0 * w ** 4 - 16.0 * w ** 2 + 8.0) / 8.0


def aggregate_histograms_with_validity_filter(results_dir, suffix,
                                              min_valid_fraction=0.85,
                                              diverge_factor=3.0,
                                              theta_abs_max=2.0):
    """Average matching histogram files, excluding seeds with excessive theta drift.

    For each seed:
      1. Load its *_ng_moments.txt → compute valid_fraction from theta divergence mask.
      2. Include its histogram only if valid_fraction >= min_valid_fraction.

    This filters out realizations where the hcs_rescale artifact (Trot → 0 in
    rescaled frame) has contaminated a significant fraction of production-window
    samples, which would distort the accumulated histogram shape.

    Returns the same dict structure as aggregate_histograms(), plus
    'n_included' and 'n_total' counts.
    """
    moments_paths = sorted(glob.glob(os.path.join(results_dir, "*_ng_moments.txt")))
    valid_fractions = {}
    for mpath in moments_paths:
        stem = os.path.basename(mpath).replace("_ng_moments.txt", "")
        try:
            m = load_non_gaussian_moments(mpath)
            theta = m.get("theta", np.full(1, np.nan))
            mask = find_theta_divergence_mask(
                theta, diverge_factor=diverge_factor, theta_abs_max=theta_abs_max
            )
            valid_fractions[stem] = float(mask.sum()) / max(float(mask.size), 1.0)
        except Exception:
            valid_fractions[stem] = 1.0  # no moments file → assume valid

    hist_paths = sorted(glob.glob(os.path.join(results_dir, f"*{suffix}")))
    if not hist_paths:
        raise FileNotFoundError(f"No *{suffix} files in {results_dir}")

    centers_ref = None
    densities = []
    n_total = len(hist_paths)

    for path in hist_paths:
        stem = os.path.basename(path).replace(suffix, "")
        if valid_fractions.get(stem, 1.0) < min_valid_fraction:
            continue
        try:
            centers, density = load_histogram(path)
        except Exception:
            continue
        if not np.any(density > 0.0):
            continue
        if centers_ref is None:
            centers_ref = centers
        elif centers.shape != centers_ref.shape or not np.allclose(centers, centers_ref):
            raise ValueError(f"Histogram bin mismatch: {path}")
        densities.append(density)

    if not densities:
        # Fall back to unfiltered if nothing passes
        return aggregate_histograms(results_dir, suffix)

    arr = np.vstack(densities)
    stderr = (
        np.std(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1 else np.zeros(arr.shape[1])
    )
    return {
        "paths": hist_paths,
        "centers": centers_ref,
        "density_mean": np.mean(arr, axis=0),
        "density_stderr": stderr,
        "n_included": len(densities),
        "n_total": n_total,
    }
