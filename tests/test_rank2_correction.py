import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.rank2_correction import (
    apply_rank0_ftr_probe,
    apply_rank2_ftr_correction,
    build_C2_table,
    compute_rank2_a2,
    load_C2_table,
    lookup_C2,
    p2,
)


def test_compute_rank2_a2_is_zero_for_isotropic_second_moment():
    vel = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])

    assert compute_rank2_a2(vel, mass=2.0) < 1.0e-14


def test_compute_rank2_a2_matches_uniaxial_covariance_and_removes_bulk():
    vel = np.array([
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    shifted = vel + np.array([10.0, -7.0, 3.0])

    K = vel.T @ vel / vel.shape[0]
    T = np.trace(K) / 3.0
    dev = K - T * np.eye(3)
    expected = np.trace(dev @ dev) / (8.0 * T * T)

    assert np.isclose(compute_rank2_a2(vel), expected)
    assert np.isclose(compute_rank2_a2(shifted), expected)


def test_apply_rank2_ftr_correction_preserves_old_formula_when_disabled():
    old = 1.2 * 3.0 * 2.0 / (3.0 * 2.0 + 2.0)

    assert np.isclose(apply_rank2_ftr_correction(1.2, 2.0), old)
    assert np.isclose(
        apply_rank2_ftr_correction(1.2, 2.0, C2=0.5, a2=0.25),
        old * (1.0 + 0.5 * 0.25),
    )
    assert np.isclose(
        apply_rank2_ftr_correction(1.2, 2.0, C2=-0.5, a2=0.25),
        old * (1.0 - 0.5 * 0.25),
    )


def test_apply_rank0_ftr_probe_scales_rank0_formula():
    old = 1.2 * 3.0 * 2.0 / (3.0 * 2.0 + 2.0)

    assert np.isclose(apply_rank0_ftr_probe(1.2, 2.0, delta=0.0), old)
    assert np.isclose(apply_rank0_ftr_probe(1.2, 2.0, delta=0.1), old * 1.1)
    assert np.isclose(apply_rank0_ftr_probe(1.2, 2.0, delta=-0.1), old * 0.9)


def test_C2_table_loader_interpolates_and_rejects_nonfinite(tmp_path):
    payload = {
        "rows": [
            {"alpha": 0.8, "AR": 2.0, "C2": 1.0},
            {"alpha": 0.9, "AR": 2.0, "C2": 3.0},
            {"alpha": 0.8, "AR": 2.5, "C2": 5.0},
        ]
    }
    path = tmp_path / "C2_table.json"
    path.write_text(json.dumps(payload))

    table = load_C2_table(str(path))

    assert np.isclose(lookup_C2(table, 0.8, 2.0), 1.0)
    assert np.isclose(lookup_C2(table, 0.85, 2.0), 2.0)
    assert np.isclose(lookup_C2(table, 0.8, 2.5), 5.0)

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows": [{"alpha": 0.8, "AR": 2.0, "C2": float("nan")}]}))
    try:
        load_C2_table(str(bad))
    except ValueError as exc:
        assert "Invalid C2" in str(exc)
    else:
        raise AssertionError("Expected non-finite C2 to fail")


def test_build_C2_table_uses_pre_collision_uvec_columns(tmp_path):
    source = tmp_path / "ctc"
    case_dir = source / "alpha_0.80_r1.00_AR2.0"
    case_dir.mkdir(parents=True)

    rng = np.random.default_rng(123)
    u1 = rng.normal(size=(2000, 3))
    u1 /= np.linalg.norm(u1, axis=1)[:, None]
    u2 = rng.normal(size=(2000, 3))
    u2 /= np.linalg.norm(u2, axis=1)[:, None]
    post1 = rng.normal(size=(2000, 3))
    post1 /= np.linalg.norm(post1, axis=1)[:, None]
    post2 = rng.normal(size=(2000, 3))
    post2 /= np.linalg.norm(post2, axis=1)[:, None]

    response = p2(u1[:, 1]) * p2(u2[:, 1])
    ftr = 1.0 + 0.7 * response
    ftr_data = np.column_stack([ftr, np.ones_like(ftr), np.ones_like(ftr)])
    uvec = np.column_stack([u1, u2, post1, post2])
    np.savetxt(case_dir / "ftr_data.txt", ftr_data)
    np.savetxt(case_dir / "uvec.dat", uvec)

    output = tmp_path / "C2_table_AR20.json"
    payload = build_C2_table(
        str(source), AR=2.0, output_path=str(output),
        epsilon_values=[0.02, 0.04], bootstrap_samples=0,
    )
    row = payload["rows"][0]

    assert output.exists()
    assert row["alpha"] == 0.8
    assert row["AR"] == 2.0
    assert row["n_valid"] == 2000
    assert row["direction_results"]["y"]["C2"] > 0.0
    assert np.isfinite(row["direction_results"]["x"]["C2"])
    assert np.isclose(row["C2"], row["direction_results"]["yz_mean"]["C2"])

    payload_again = build_C2_table(
        str(source), AR=2.0, epsilon_values=[0.02, 0.04],
        bootstrap_samples=0,
    )

    altered_post = np.column_stack([u1, u2, -post1, -post2])
    np.savetxt(case_dir / "uvec.dat", altered_post)
    payload_post_changed = build_C2_table(
        str(source), AR=2.0, epsilon_values=[0.02, 0.04],
        bootstrap_samples=0,
    )

    assert np.isclose(
        payload_again["rows"][0]["C2"],
        payload_post_changed["rows"][0]["C2"],
    )
