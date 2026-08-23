import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.archive.tools.diagnose_stress_transport_weight import (
    compute_case,
    lambda2_dsmc_hs,
    p2,
)
from src.simulation.collision import lookup_stress_transport_weight


def test_lambda2_dsmc_elastic_hard_sphere_is_zero():
    lambda2 = lambda2_dsmc_hs(1.0, n_grid=20001)
    assert abs(lambda2) < 1.0e-8


def test_compute_case_uses_fixed_incoming_vrel_and_q_weight(tmp_path):
    case_dir = tmp_path / "alpha_0.80_r1.00_AR2.0"
    case_dir.mkdir()

    ghat_post = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    chi = np.zeros((3, 10))
    chi[:, 7:10] = ghat_post
    np.savetxt(case_dir / "chi.txt", chi)

    ef = np.array([
        [2.0, 0.5, 0.5, 4.0, 0.2, 0.3],
        [2.0, 0.5, 0.5, 2.0, 0.2, 0.3],
        [2.0, 0.5, 0.5, 1.0, 0.2, 0.3],
    ])
    np.savetxt(case_dir / "Ef.txt", ef)

    row = compute_case(str(case_dir), n_grid=20001)

    c = -ghat_post[:, 0]
    expected_lambda = float(np.mean(p2(c)))
    q = ef[:, 3] / ef[:, 0]
    expected_lambda_q = float(np.sum(q * p2(c)) / np.sum(q))

    assert row.n_events == 3
    assert np.isclose(row.lambda2_ctc, expected_lambda)
    assert np.isclose(row.lambda2_ctc_q, expected_lambda_q)
    assert np.isfinite(row.w_eta)


def test_lookup_stress_transport_weight_interpolates_and_rejects_gt_one():
    table = {
        (0.8, 2.0): 0.98,
        (1.0, 2.0): 0.99,
        (0.8, 3.0): 0.94,
        (1.0, 3.0): 0.96,
    }

    assert np.isclose(lookup_stress_transport_weight(table, 0.9, 2.0), 0.985)
    assert np.isclose(lookup_stress_transport_weight(table, 0.9, 2.5), 0.9675)

    try:
        lookup_stress_transport_weight(table, 0.9, 1.4)
    except KeyError as exc:
        assert "AR in" in str(exc)
    else:
        raise AssertionError("Expected AR outside table range to fail")

    bad_table = {(0.8, 2.0): 1.01}
    try:
        lookup_stress_transport_weight(bad_table, 0.8, 2.0)
    except ValueError as exc:
        assert "w_eta" in str(exc)
    else:
        raise AssertionError("Expected w_eta > 1 to fail")
