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
