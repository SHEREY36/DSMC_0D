import os
import json
import numpy as np
from scipy.stats import beta as beta_dist

from .data_loader import load_max_dissipation, compute_one_hit_ratio


def build_gamma_max_table(root_dir, alpha_dirs, ar_dirs):
    """Build a lookup table of maximum fractional dissipation.

    Computes gamma_max from Ef.txt for each (alpha, AR) pair in the
    CTC_data/Alpha/ directory tree.

    Returns dict with string keys "(alpha, AR)" -> float.
    """
    table = {}
    for alpha in alpha_dirs:
        if alpha == 100:
            continue  # elastic case has no dissipation
        for ar in ar_dirs:
            ef_path = os.path.join(root_dir, str(alpha), str(ar), "Ef.txt")
            if not os.path.exists(ef_path):
                print(f"Warning: {ef_path} not found, skipping")
                continue
            gamma_max = load_max_dissipation(ef_path)
            alpha_val = alpha / 100.0
            AR_val = ar / 10.0
            key = f"({alpha_val}, {AR_val})"
            table[key] = float(gamma_max)
    return table


def build_one_hit_table(root_dir, alpha_dirs, ar_dirs):
    """Build a lookup table of single-contact-point collision probability.

    Computes one_hit_ratio from NPhit.txt for each (alpha, AR) pair.

    Returns dict with string keys "(alpha, AR)" -> float.
    """
    table = {}
    for alpha in alpha_dirs:
        for ar in ar_dirs:
            nphit_path = os.path.join(
                root_dir, str(alpha), str(ar), "NPhit.txt"
            )
            if not os.path.exists(nphit_path):
                print(f"Warning: {nphit_path} not found, skipping")
                continue
            ratio = compute_one_hit_ratio(nphit_path)
            alpha_val = alpha / 100.0
            AR_val = ar / 10.0
            key = f"({alpha_val}, {AR_val})"
            table[key] = float(ratio)
    return table


def lookup_gamma_max(table, alpha, AR):
    """Look up gamma_max for a given (alpha, AR) pair.

    Raises KeyError with an informative message if the pair is not found.
    """
    key = f"({alpha}, {AR})"
    if key not in table:
        available = sorted(table.keys())
        raise KeyError(
            f"gamma_max not available for (alpha={alpha}, AR={AR}). "
            f"Available pairs: {available}"
        )
    return table[key]


def lookup_one_hit(table, alpha, AR):
    """Look up one-hit probability for a given (alpha, AR) pair.

    Raises KeyError with an informative message if the pair is not found.
    """
    key = f"({alpha}, {AR})"
    if key not in table:
        available = sorted(table.keys())
        raise KeyError(
            f"one_hit probability not available for (alpha={alpha}, AR={AR}). "
            f"Available pairs: {available}"
        )
    return table[key]


def save_table(table, filepath):
    """Save a lookup table to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(table, f, indent=2)


def load_table(filepath):
    """Load a lookup table from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def sample_dissp(a, b):
    """Sample a single value from the Beta dissipation distribution."""
    return beta_dist.rvs(a, b, size=1)[0]
