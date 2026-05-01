"""Tests for the conditional chi Beta model (src/preprocessing/mu_chi_model.py)."""
import numpy as np
import pytest

from src.preprocessing.mu_chi_model import (
    eval_beta_params,
    fit_mu_chi_model,
    load_mu_chi_model,
    save_mu_chi_model,
    sample_chi_given_mu,
)


MODEL_PATH = "models/mu_chi_beta_coeffs.npz"


@pytest.fixture(scope="module")
def loaded_model():
    return load_mu_chi_model(MODEL_PATH)


def test_eval_beta_params_positive(loaded_model):
    c_a, c_b, M, N, J, beta_exp = loaded_model
    for mu in np.linspace(0.05, 0.95, 10):
        for alpha in [0.6, 0.8, 1.0]:
            for AR in [1.0, 1.5, 2.0, 3.0]:
                a, b = eval_beta_params(mu, alpha, AR, c_a, c_b, M, N, J, beta_exp)
                assert a > 0.0, f"a <= 0 at mu={mu}, alpha={alpha}, AR={AR}: a={a}"
                assert b > 0.0, f"b <= 0 at mu={mu}, alpha={alpha}, AR={AR}: b={b}"
                assert np.isfinite(a) and np.isfinite(b), "Non-finite Beta params"


def test_sample_chi_in_range(loaded_model):
    c_a, c_b, M, N, J, beta_exp = loaded_model
    rng = np.random.default_rng(42)
    for mu in [0.1, 0.5, 0.9]:
        for alpha in [0.6, 1.0]:
            for AR in [1.0, 2.0]:
                for _ in range(50):
                    chi = sample_chi_given_mu(mu, alpha, AR, c_a, c_b, M, N, J,
                                              beta_exp=beta_exp, rng=rng)
                    assert 0.0 <= chi <= np.pi, (
                        f"chi={chi:.4f} out of [0, pi] for mu={mu}, alpha={alpha}, AR={AR}"
                    )


def test_sphere_limit_concentrated(loaded_model):
    """For AR=1, alpha=1 (elastic sphere), Beta params should be large — tight distribution."""
    c_a, c_b, M, N, J, beta_exp = loaded_model
    for mu in np.linspace(0.1, 0.9, 5):
        a, b = eval_beta_params(mu, 1.0, 1.0, c_a, c_b, M, N, J, beta_exp)
        # Large a+b means low variance (concentrated distribution)
        concentration = a + b
        assert concentration > 10.0, (
            f"AR=1 alpha=1 should have concentrated Beta; got a={a:.2f}, b={b:.2f} at mu={mu:.1f}"
        )


def test_chi_mean_tracks_hs_for_ar1(loaded_model):
    """For AR=1, alpha=1: mean chi from model ≈ chi_hs(mu) within tolerance."""
    c_a, c_b, M, N, J, beta_exp = loaded_model
    rng = np.random.default_rng(99)
    n_samples = 5000
    tolerance = 0.15  # radians

    for mu in [0.2, 0.4, 0.6, 0.8]:
        # Hard-sphere elastic chi
        chi_hs = np.arccos(np.clip(1.0 - 2.0 * mu**2, -1.0, 1.0))
        # Sample from model
        samples = [sample_chi_given_mu(mu, 1.0, 1.0, c_a, c_b, M, N, J,
                                       beta_exp=beta_exp, rng=rng)
                   for _ in range(n_samples)]
        mean_chi = float(np.mean(samples))
        assert abs(mean_chi - chi_hs) < tolerance, (
            f"AR=1 alpha=1 mu={mu:.1f}: model mean={mean_chi:.3f}, chi_hs={chi_hs:.3f}, "
            f"diff={abs(mean_chi - chi_hs):.3f} > tol={tolerance}"
        )


def test_inelastic_chi_decreases(loaded_model):
    """At fixed mu and AR, mean chi should decrease as alpha decreases (more inelastic)."""
    c_a, c_b, M, N, J, beta_exp = loaded_model
    mu = 0.6
    AR = 2.0
    alphas = [1.0, 0.8, 0.6]
    means = []
    rng = np.random.default_rng(7)
    for alpha in alphas:
        samples = [sample_chi_given_mu(mu, alpha, AR, c_a, c_b, M, N, J,
                                       beta_exp=beta_exp, rng=rng)
                   for _ in range(3000)]
        means.append(float(np.mean(samples)))
    # Mean chi should be non-increasing as alpha decreases (more inelastic → smaller chi generally)
    # Allow some tolerance for statistical noise
    assert means[0] >= means[2] - 0.1, (
        f"Expected chi(alpha=1.0) >= chi(alpha=0.6), got {means[0]:.3f} vs {means[2]:.3f}"
    )


def test_load_save_roundtrip(tmp_path, loaded_model):
    c_a, c_b, M, N, J, beta_exp = loaded_model
    path = str(tmp_path / "test_mu_chi.npz")
    save_mu_chi_model(path, c_a, c_b, M, N, J, beta_exp)
    c_a2, c_b2, M2, N2, J2, beta2 = load_mu_chi_model(path)

    assert M == M2 and N == N2 and J == J2
    assert abs(beta_exp - beta2) < 1e-12
    np.testing.assert_array_equal(c_a, c_a2)
    np.testing.assert_array_equal(c_b, c_b2)
