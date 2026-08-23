"""Compatibility wrapper for archived mu-chi preprocessing utilities."""

import os

from src.archive.preprocessing.mu_chi_model import *  # noqa: F401,F403
from src.archive.preprocessing import mu_chi_model as _archived


def _archive_path(path):
    if path and not os.path.exists(path):
        candidate = os.path.join("models", "archive", os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
    return path


def load_mu_chi_model(path):
    return _archived.load_mu_chi_model(_archive_path(path))
