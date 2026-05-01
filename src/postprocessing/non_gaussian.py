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
    cols = [
        "t", "tau", "Ttrans", "Trot", "theta", "a2_tr", "a3_tr",
        "a2_rot", "a11", "c4", "c6", "w4", "c2w2",
        "n_samples_particles",
    ]
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
    moment_names = ["c4", "c6", "w4", "c2w2"]
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
        if centers_ref is None:
            centers_ref = centers
        elif centers.shape != centers_ref.shape or not np.allclose(centers, centers_ref):
            raise ValueError(f"Histogram bin centers do not match: {path}")
        densities.append(density)

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


def maxwell_component_pdf(c):
    """Reference PDF for one reduced component c_i = v_i / sqrt(2T/m)."""
    return np.exp(-c * c) / np.sqrt(np.pi)


def rayleigh_rot_speed_pdf(w):
    """Reference PDF for w = sqrt(w_y^2 + w_z^2), with <w^2> = 1."""
    return 2.0 * w * np.exp(-w * w)


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
    tr = aggregate_histograms(results_dir, "_ng_hist_tr.txt")
    rot_comp = aggregate_histograms(results_dir, "_ng_hist_rot_component.txt")
    rot_speed = aggregate_histograms(results_dir, "_ng_hist_rot_speed.txt")
    result["histograms"] = {
        "tr_component": histogram_ratio_to_reference(tr, maxwell_component_pdf),
        "rot_component": histogram_ratio_to_reference(rot_comp, maxwell_component_pdf),
        "rot_speed": histogram_ratio_to_reference(rot_speed, rayleigh_rot_speed_pdf),
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
