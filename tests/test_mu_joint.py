import numpy as np

from src.simulation.mu_joint import (
    MuJointAR2Model,
    build_mu_joint_ar2_lookup,
    hard_sphere_chi_and_gmag,
    mu_plane_post_relative,
)


def test_mu_joint_lookup_builds_and_samples(tmp_path):
    case_dir = tmp_path / "source" / "alpha_0.80_r1.00_AR2.0"
    case_dir.mkdir(parents=True)

    chi_rows = np.array([
        [0.10, 0.20, 0.0, 0.10],
        [0.20, 0.40, 0.0, 0.35],
        [0.30, 0.70, 0.0, 0.70],
        [0.40, 0.90, 0.0, 0.95],
    ])
    ef_rows = np.array([
        [2.0, 0.5, 0.5, 1.5, 0.7, 0.3, 0.0],
        [2.0, 0.5, 0.5, 1.4, 0.6, 0.4, 0.0],
        [2.0, 0.5, 0.5, 1.3, 0.5, 0.5, 0.0],
        [2.0, 0.5, 0.5, 1.2, 0.4, 0.6, 0.0],
    ])
    np.savetxt(case_dir / "chi.txt", chi_rows)
    np.savetxt(case_dir / "Ef.txt", ef_rows)

    lookup_path = tmp_path / "lookup.npz"
    build_mu_joint_ar2_lookup(str(tmp_path / "source"), str(lookup_path), n_mu_bins=2)
    model = MuJointAR2Model(str(lookup_path), auto_build=False)
    sample = model.sample(0.80, 0.8, rng=np.random.default_rng(1))

    assert np.isfinite(sample["chi_rad"])
    assert sample["q_total"] > 0.0
    assert 0.0 <= sample["eps_tr_post"] <= 1.0
    assert 0.0 <= sample["eps_rot1_post"] <= 1.0


def test_mu_plane_update_reproduces_hard_sphere_geometry():
    rng = np.random.default_rng(123)
    alpha = 0.8
    for _ in range(200):
        g = rng.normal(size=3)
        eij = rng.normal(size=3)
        eij /= np.linalg.norm(eij)
        if np.dot(eij, g) < 0.0:
            eij = -eij

        chi, gpost_mag = hard_sphere_chi_and_gmag(g, eij, alpha)
        actual = mu_plane_post_relative(g, eij, chi, gpost_mag, rng=rng)
        expected = g - (1.0 + alpha) * np.dot(g, eij) * eij

        assert np.allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
