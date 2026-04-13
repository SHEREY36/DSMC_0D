import json
import numpy as np


def save_zr_eff_table(table, filepath):
    """Save the Z_R_eff lookup table to a JSON file.

    Table format: {"(alpha, AR)": {"theta_star": ..., "Z_R_eff": ..., ...}}
    """
    with open(filepath, 'w') as f:
        json.dump(table, f, indent=2)


def load_zr_eff_table(filepath):
    """Load the Z_R_eff lookup table from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def lookup_zr_eff(table, alpha, AR):
    """Look up (theta_star, Z_R_eff) for a given (alpha, AR) pair.

    Tries exact key match first, then linearly interpolates in alpha for the
    given AR. Raises KeyError if AR is not represented in the table.

    Returns (theta_star, Z_R_eff).
    """
    key = f"({alpha:.3f}, {float(AR):.1f})"
    if key in table:
        entry = table[key]
        return float(entry['theta_star']), float(entry['Z_R_eff'])

    return _interpolate_alpha_for_AR(table, alpha, AR)


def _parse_table_key(key):
    alpha_str, ar_str = key.strip()[1:-1].split(",")
    return float(alpha_str.strip()), float(ar_str.strip())


def _interpolate_alpha_for_AR(table, alpha, AR):
    alpha = float(alpha)
    AR = float(AR)

    pairs = []
    for key, entry in table.items():
        alpha_i, ar_i = _parse_table_key(key)
        if np.isclose(ar_i, AR, atol=1e-12):
            pairs.append((alpha_i, float(entry['theta_star']), float(entry['Z_R_eff'])))

    if not pairs:
        available = sorted({_parse_table_key(k)[1] for k in table.keys()})
        raise KeyError(
            f"Z_R_eff table has no entries for AR={AR}. "
            f"Available AR values: {available}"
        )

    pairs.sort(key=lambda x: x[0])
    alphas = np.array([p[0] for p in pairs])
    theta_stars = np.array([p[1] for p in pairs])
    zr_effs = np.array([p[2] for p in pairs])

    theta_star = float(np.interp(alpha, alphas, theta_stars))
    Z_R_eff = float(np.interp(alpha, alphas, zr_effs))

    return theta_star, Z_R_eff
