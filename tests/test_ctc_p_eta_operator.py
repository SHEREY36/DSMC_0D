import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.archive.tools.diagnose_ctc_p_eta_operator import (  # noqa: E402
    GHAT0,
    compute_case,
)
from src.simulation.dsmc import _chi_hs  # noqa: E402
from src.simulation.mu_joint import mu_plane_post_relative_with_eps  # noqa: E402


def _write_case(tmp_path, ghat_posts, alpha=1.0, AR=2.0):
    case_dir = tmp_path / f"alpha_{alpha:.2f}_r1.00_AR{AR:.1f}"
    case_dir.mkdir()

    mus = np.array([0.25, 0.45, 0.65, 0.85], dtype=float)
    eij = np.column_stack([
        mus,
        np.sqrt(1.0 - mus**2),
        np.zeros_like(mus),
    ])

    chi = np.zeros((mus.size, 10))
    chi[:, 3] = mus
    chi[:, 4:7] = eij
    chi[:, 7:10] = ghat_posts
    np.savetxt(case_dir / "chi.txt", chi)

    ef = np.zeros((mus.size, 6))
    ef[:, 0] = 1.0
    ef[:, 1] = 0.1
    ef[:, 2] = 0.2
    ef[:, 3] = 1.0
    ef[:, 4] = 0.1
    ef[:, 5] = 0.2
    np.savetxt(case_dir / "Ef.txt", ef)
    return case_dir


def test_operator_p_eta_is_one_when_ctc_equals_full_branch(tmp_path):
    alpha = 1.0
    mus = np.array([0.25, 0.45, 0.65, 0.85], dtype=float)
    eij = np.column_stack([
        mus,
        np.sqrt(1.0 - mus**2),
        np.zeros_like(mus),
    ])
    # CTC stores eij opposite to the DSMC-aligned ghat0 convention, so the
    # diagnostic flips eij internally before constructing the full branch.
    aligned_eij = -eij
    ghat_posts = []
    for mu, normal in zip(mus, aligned_eij):
        gpost = mu_plane_post_relative_with_eps(
            GHAT0, normal, _chi_hs(mu, alpha), 1.0, 0.0
        )
        ghat_posts.append(gpost / np.linalg.norm(gpost))
    case_dir = _write_case(tmp_path, np.array(ghat_posts), alpha=alpha)

    row = compute_case(str(case_dir), eps_model=None, n_rotations=3, seed=7)

    assert row.finite
    assert np.isclose(row.p_eta_lsq, 1.0, atol=1.0e-12)
    assert row.p_eta_std < 1.0e-10


def test_operator_p_eta_is_zero_when_ctc_equals_scalar_branch(tmp_path):
    ghat_posts = np.broadcast_to(GHAT0, (4, 3))
    case_dir = _write_case(tmp_path, ghat_posts, alpha=1.0)

    row = compute_case(str(case_dir), eps_model=None, n_rotations=3, seed=11)

    assert row.finite
    assert np.isclose(row.p_eta_lsq, 0.0, atol=1.0e-12)
    assert row.p_eta_std < 1.0e-10
