import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.vss_rank2 import (
    alpha_eff_from_beta,
    build_vss_alpha_eff_table,
    load_vss_alpha_eff_table,
    lookup_vss_alpha_eff,
    p2,
    sample_vss_chi,
    vss_rank2_moment,
)


def test_vss_rank2_moment_matches_monte_carlo():
    rng = np.random.default_rng(123)
    alpha_eff = 6.8
    chis = np.array([sample_vss_chi(alpha_eff, rng=rng) for _ in range(200000)])
    actual = float(np.mean(1.0 - p2(np.cos(chis))))
    expected = vss_rank2_moment(alpha_eff)

    assert np.isclose(actual, expected, atol=5.0e-3)


def test_alpha_eff_forward_root_inverts_beta():
    beta = 0.6
    alpha_eff = alpha_eff_from_beta(beta, branch="forward")

    assert alpha_eff > 1.0
    assert np.isclose(vss_rank2_moment(alpha_eff), beta)


def test_build_vss_alpha_eff_table_from_chi_fixture(tmp_path):
    source = tmp_path / "source"
    case_dir = source / "alpha_0.80_r1.00_AR2.0"
    case_dir.mkdir(parents=True)

    chi_rad = np.array([0.0, np.pi / 2.0, np.pi / 3.0, np.pi / 4.0])
    chi = np.zeros((chi_rad.size, 2))
    chi[:, 1] = chi_rad
    np.savetxt(case_dir / "chi.txt", chi)

    output = tmp_path / "vss_alpha_eff.json"
    payload = build_vss_alpha_eff_table(
        str(source), AR=2.0, p_eta=0.5, output_path=str(output)
    )
    row = payload["rows"][0]

    beta_ctc = float(np.mean(1.0 - p2(np.cos(chi_rad))))
    beta_target = 0.5 * beta_ctc
    expected_alpha_eff = alpha_eff_from_beta(beta_target, branch="forward")

    assert np.isclose(row["beta_ctc"], beta_ctc)
    assert np.isclose(row["beta_target"], beta_target)
    assert np.isclose(row["alpha_eff"], expected_alpha_eff)

    table = load_vss_alpha_eff_table(str(output))
    assert np.isclose(lookup_vss_alpha_eff(table, 0.8, 2.0), expected_alpha_eff)


def test_lookup_vss_alpha_eff_interpolates_for_ar_specific_table(tmp_path):
    payload = {
        "rows": [
            {"alpha": 0.8, "AR": 1.5, "alpha_eff": 8.0},
            {"alpha": 0.9, "AR": 1.5, "alpha_eff": 10.0},
            {"alpha": 0.8, "AR": 2.5, "alpha_eff": 12.0},
        ]
    }
    output = tmp_path / "vss_alpha_eff_AR15.json"
    import json
    output.write_text(json.dumps(payload))

    table = load_vss_alpha_eff_table(str(output))

    assert np.isclose(lookup_vss_alpha_eff(table, 0.85, 1.5), 9.0)
    assert np.isclose(lookup_vss_alpha_eff(table, 0.8, 2.5), 12.0)
