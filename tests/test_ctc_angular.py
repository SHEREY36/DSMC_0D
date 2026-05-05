import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.ctc_angular import (
    CTCAngularModel,
    build_ctc_angular_lookup,
    load_case_rows,
)
from src.simulation.mu_joint import mu_plane_post_relative_with_eps


def _ctc_row(mu, chi_rad, eps_rad):
    ghat0 = np.array([-1.0, 0.0, 0.0])
    eij = np.array([-mu, np.sqrt(max(1.0 - mu * mu, 0.0)), 0.0])
    gpost = mu_plane_post_relative_with_eps(
        ghat0, eij, chi_rad, 1.0, eps_rad, rng=np.random.default_rng(1)
    )
    row = np.zeros(10)
    row[1] = chi_rad
    row[3] = mu
    row[4:7] = eij
    row[7:10] = gpost
    return row


def test_load_case_rows_recovers_chi_and_eps(tmp_path):
    case_dir = tmp_path / "alpha_0.80_r1.00_AR2.0"
    case_dir.mkdir()
    rows = np.array([
        _ctc_row(0.4, 0.7, 0.3),
        _ctc_row(0.8, 1.1, -0.5),
    ])
    np.savetxt(case_dir / "chi.txt", rows)

    actual, _ = load_case_rows(str(case_dir))

    assert np.allclose(actual[:, 0], [0.4, 0.8])
    assert np.allclose(actual[:, 1], [0.7, 1.1])
    assert np.allclose(actual[:, 2], [0.3, -0.5])


def test_ctc_angular_lookup_builds_and_samples(tmp_path):
    source = tmp_path / "source"
    case_dir = source / "alpha_0.80_r1.00_AR2.0"
    case_dir.mkdir(parents=True)
    rows = np.array([
        _ctc_row(0.20, 0.40, 0.10),
        _ctc_row(0.45, 0.70, -0.20),
        _ctc_row(0.70, 1.00, 0.30),
        _ctc_row(0.95, 1.30, -0.40),
    ])
    np.savetxt(case_dir / "chi.txt", rows)

    lookup_path = tmp_path / "ctc_angular.npz"
    build_ctc_angular_lookup(
        str(source), str(lookup_path), n_mu_bins=4, min_AR=1.5
    )
    model = CTCAngularModel(str(lookup_path))

    chi_rad, eps_rad = model.sample(0.80, 2.0, 0.72, rng=np.random.default_rng(2))
    assert np.isfinite(chi_rad)
    assert np.isfinite(eps_rad)

    lambda2 = model.lambda2_by_bin(0.80, 2.0)
    assert np.isfinite(lambda2).any()
