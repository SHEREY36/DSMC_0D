import os
import json
import numpy as np
from scipy.stats import laplace as laplace_dist


def load_ftr_data(results_root, alpha, r, AR):
    """Load f_tr column from ftr_data.txt for a given (alpha, r, AR) case.

    ftr_data.txt columns: [f_tr, delta_Et_el, delta_E_diss]
    Returns 1D array of f_tr values.
    """
    folder = f"alpha_{alpha:.3f}_r{r:.2f}_AR{AR:.1f}"
    path = os.path.join(results_root, folder, "ftr_data.txt")
    data = np.loadtxt(path)
    return data[:, 0]


def fit_ftr_laplace(f_tr_data, trim_percentile=1.0):
    """Fit a Laplace distribution to f_tr samples via MLE.

    Trims the top and bottom `trim_percentile` percent to exclude extreme
    outliers (caused by near-elastic collisions with tiny delta_E_diss)
    before fitting.

    Returns (loc, scale).
    """
    lo = np.percentile(f_tr_data, trim_percentile)
    hi = np.percentile(f_tr_data, 100.0 - trim_percentile)
    core = f_tr_data[(f_tr_data >= lo) & (f_tr_data <= hi)]

    loc, scale = laplace_dist.fit(core)
    return float(loc), float(scale)


def build_ftr_table(results_root, alpha_values, r, AR, trim_percentile=1.0):
    """Fit Laplace f_tr parameters for each alpha at fixed r and AR.

    Returns dict keyed by "(alpha, AR)" -> {loc, scale}.
    """
    table = {}
    for alpha in alpha_values:
        folder = f"alpha_{alpha:.3f}_r{r:.2f}_AR{AR:.1f}"
        path = os.path.join(results_root, folder, "ftr_data.txt")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping alpha={alpha}")
            continue
        print(f"  Fitting f_tr for alpha={alpha:.3f}...")
        f_tr_data = load_ftr_data(results_root, alpha, r, AR)
        loc, scale = fit_ftr_laplace(f_tr_data, trim_percentile=trim_percentile)
        key = f"({alpha:.3f}, {AR:.1f})"
        table[key] = {"loc": loc, "scale": scale}
        print(f"    loc={loc:.4f}, scale={scale:.4f}")
    return table


def save_ftr_table(table, filepath):
    """Save f_tr parameter table to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(table, f, indent=2)


def load_ftr_table(filepath):
    """Load f_tr parameter table from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def lookup_ftr_params(table, alpha, AR):
    """Look up Laplace f_tr parameters (loc, scale) for (alpha, AR).

    Tries exact match first, then interpolates linearly in alpha at fixed AR.
    Returns tuple (loc, scale), or raises KeyError if AR not available.
    """
    alpha = float(alpha)
    AR = float(AR)

    key = f"({alpha:.3f}, {AR:.1f})"
    if key in table:
        entry = table[key]
        return entry["loc"], entry["scale"]

    # Collect all entries for this AR
    pairs = []
    for k, v in table.items():
        a_str, ar_str = k.strip()[1:-1].split(",")
        ar_i = float(ar_str.strip())
        if np.isclose(ar_i, AR, atol=1e-9):
            pairs.append((float(a_str.strip()), v))

    if not pairs:
        available = sorted({float(k.strip()[1:-1].split(",")[1]) for k in table})
        raise KeyError(
            f"f_tr params not available for AR={AR}. Available: {available}"
        )

    pairs.sort(key=lambda x: x[0])
    alphas = np.array([p[0] for p in pairs])
    fields = ["loc", "scale"]
    arrays = {f: np.array([p[1][f] for p in pairs]) for f in fields}

    return tuple(float(np.interp(alpha, alphas, arrays[f])) for f in fields)
