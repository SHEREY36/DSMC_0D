import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.ctc_c0 import (
    CTCCase,
    build_c0_table,
    estimate_case_c0,
    fixed_point_residual,
    hcs_balance_coefficient,
    mode_neutral_factor,
    solve_grid_self_consistent_theta,
    solve_fixed_point_c0,
    solve_self_consistent_theta,
)
import src.calibration.ctc_c0 as ctc_c0
from src.simulation.collision import CollisionModels


def _write_case(root, alpha, AR, f_req, delta_E, nphit=None, theta=1.0, ef=None):
    case = root / f"alpha_{alpha:.2f}_r{theta:.2f}_AR{AR:.1f}"
    case.mkdir(parents=True)
    f_req = np.asarray(f_req, dtype=float)
    delta_E = np.asarray(delta_E, dtype=float)
    if nphit is None:
        nphit = np.ones(f_req.size, dtype=int)
    nphit = np.asarray(nphit, dtype=int)
    ftr_signed = -f_req
    delta_Et_el = np.zeros_like(f_req)
    np.savetxt(case / "ftr_data.txt", np.column_stack([ftr_signed, delta_Et_el, delta_E]))
    np.savetxt(case / "NPhit.txt", nphit, fmt="%d")
    np.savetxt(case / "chi.txt", np.column_stack([np.linspace(0.1, 1.0, f_req.size)]))
    if ef is not None:
        np.savetxt(case / "Ef.txt", np.asarray(ef, dtype=float))
    return case


class _ConstantExchangeGMM:
    def __init__(self, epsilon_tr_f):
        self.epsilon_tr_f = float(epsilon_tr_f)

    def sample_conditionals(self, r, e_tr, e_r1, n_samples=1):
        return np.tile([self.epsilon_tr_f, e_r1], (n_samples, 1))


class _FakeHCSModels:
    def __init__(self, epsilon_tr_f, gamma_max=1.0, one_hit=1.0):
        self.cond_gmm = _ConstantExchangeGMM(epsilon_tr_f)
        self.gamma_max = float(gamma_max)
        self.one_hit = float(one_hit)

    def get_gamma_max(self, alpha, AR):
        return self.gamma_max

    def get_one_hit(self, alpha, AR):
        return self.one_hit


def test_ctc_c0_estimator_uses_positive_dsmc_sign(tmp_path):
    root = tmp_path / "ctc"
    case_dir = _write_case(root, 0.90, 2.0, f_req=[0.6, 0.6], delta_E=[1.0, 1.0])
    case = CTCCase(alpha=0.90, theta=1.0, AR=2.0, path=case_dir)

    C0, diag = estimate_case_c0(case, bootstrap_samples=0)

    assert C0 > 0.0
    assert np.isclose(C0, 1.0)
    assert np.isclose(diag["mean_f_req"], 0.6)


def test_ctc_c0_estimator_matches_weighted_moment_formula_and_filters_hits(tmp_path):
    f_req = np.array([0.3, 0.9, 99.0])
    delta_E = np.array([2.0, 4.0, 100.0])
    nphit = np.array([1, 1, 2])
    case_dir = _write_case(tmp_path, 0.80, 1.5, f_req=f_req, delta_E=delta_E, nphit=nphit)
    case = CTCCase(alpha=0.80, theta=1.0, AR=1.5, path=case_dir)

    C0, diag = estimate_case_c0(case, bootstrap_samples=0)

    B = mode_neutral_factor(1.0)
    expected = np.sum(f_req[:2] * delta_E[:2]) / np.sum(B * delta_E[:2])
    assert np.isclose(C0, expected)
    assert diag["n_total"] == 3
    assert diag["n_single_hit"] == 2
    assert diag["n_used"] == 2


def test_ctc_c0_elastic_alpha_is_zero_by_convention(tmp_path):
    case_dir = _write_case(tmp_path, 1.00, 2.0, f_req=[5.0, 7.0], delta_E=[1.0, 1.0])
    case = CTCCase(alpha=1.00, theta=1.0, AR=2.0, path=case_dir)

    C0, diag = estimate_case_c0(case, bootstrap_samples=0)

    assert C0 == 0.0
    assert diag["status"] == "elastic_alpha_by_convention"


def test_hcs_balance_formula_allows_negative_effective_c0():
    theta = 0.1
    B = mode_neutral_factor(theta)
    C0 = hcs_balance_coefficient(mean_A=-0.3, mean_delta_E=1.15, theta=theta)

    assert np.isclose(C0, 1.0 - 0.3 / (B * 1.15))
    assert C0 < 0.0


def test_hcs_balance_estimator_requires_theta_table(tmp_path):
    _write_case(tmp_path, 0.80, 2.0, f_req=[0.3], delta_E=[1.0])

    try:
        build_c0_table(
            tmp_path,
            AR=2.0,
            estimator="hcs-balance",
            models=_FakeHCSModels(0.0),
        )
    except ValueError as exc:
        assert "theta_table" in str(exc)
    else:
        raise AssertionError("Expected hcs-balance without theta_table to fail")


def test_hcs_balance_estimator_uses_dsmc_exchange_not_ftr_data(tmp_path):
    theta = 0.1
    Erot = 2.0
    Etr_raw = 3.0
    Etotal_raw = Etr_raw + Erot
    ef = np.array([
        [Etr_raw, 1.0, 1.0, Etotal_raw, 0.5, 0.5, 0.2],
        [Etr_raw, 1.0, 1.0, Etotal_raw, 0.5, 0.5, 0.2],
    ])
    _write_case(
        tmp_path,
        0.80,
        2.0,
        f_req=[99.0, 99.0],
        delta_E=[99.0, 99.0],
        theta=1.0,
        ef=ef,
    )
    theta_table = {"(0.800, 2.0)": theta}
    payload = build_c0_table(
        tmp_path,
        AR=2.0,
        estimator="hcs-balance",
        models=_FakeHCSModels(epsilon_tr_f=0.0, gamma_max=1.0, one_hit=1.0),
        theta_table=theta_table,
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
    )

    row = payload["diagnostics"]["rows"][0]
    C0 = payload["table"]["(0.800, 2.0)"]
    Etr_scaled = 1.5 * theta * Erot
    Etotal = Etr_scaled + Erot
    A = -Etr_scaled
    expected_delta_E = 0.5 * Etotal
    expected = hcs_balance_coefficient(A, expected_delta_E, theta)

    assert np.isclose(C0, expected)
    assert C0 < 0.0
    assert row["negative_due_to_exchange"] is True
    assert np.isclose(row["mean_A"], A)
    assert np.isclose(row["mean_delta_E_closure"], expected_delta_E)


def test_fixed_point_solver_finds_relative_cooling_root_without_clipping():
    theta = 1.0
    Etr = np.full(4, 2.0)
    Erot = np.full(4, 2.0)
    Etotal = Etr + Erot
    eps_f = np.full(4, 0.4)
    delta_E = np.full(4, 0.2)

    C0, diag = solve_fixed_point_c0(
        theta=theta,
        Etr_i=Etr,
        Erot_i=Erot,
        Etotal_i=Etotal,
        epsilon_tr_f=eps_f,
        delta_E=delta_E,
        C_min=-5.0,
        C_max=5.0,
    )

    assert diag["status"] == "ok"
    assert abs(fixed_point_residual(
        C0,
        theta=theta,
        Etr_i=Etr,
        Erot_i=Erot,
        Etotal_i=Etotal,
        epsilon_tr_f=eps_f,
        delta_E=delta_E,
    )) < 1.0e-7


def test_hcs_fixed_point_estimator_uses_analytic_dsmc_pair_ensemble(tmp_path):
    _write_case(tmp_path, 0.80, 2.0, f_req=[0.3], delta_E=[1.0], ef=[[1, 1, 1, 1, 1, 1, 1]])
    payload = build_c0_table(
        tmp_path,
        AR=2.0,
        estimator="hcs-fixed-point",
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.4, one_hit=0.5),
        theta_table={"(0.800, 2.0)": 1.0},
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
        bootstrap_seed=123,
        fixed_point_samples=2000,
    )

    row = payload["diagnostics"]["rows"][0]
    C0 = payload["table"]["(0.800, 2.0)"]

    assert row["estimator"] == "hcs-fixed-point"
    assert row["n_used"] == 2000
    assert row["status"] == "ok"
    assert np.isfinite(C0)
    assert abs(row["residual"]) < 1.0e-6


def test_self_consistent_theta_solver_finds_root_without_theta_target(tmp_path):
    case_dir = _write_case(tmp_path, 0.80, 2.0, f_req=[0.6], delta_E=[1.0])
    case = CTCCase(alpha=0.80, theta=1.0, AR=2.0, path=case_dir)

    theta, diag = solve_self_consistent_theta(
        C0=1.0,
        case=case,
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.4, one_hit=0.5),
        beta_a=1.0,
        beta_b=1.0,
        n_samples=1000,
        seed=123,
        theta_min=0.3,
        theta_max=1.4,
    )

    assert diag["status"] == "ok"
    assert 0.3 <= theta <= 1.4
    assert abs(diag["theta_residual"]) < 1.0e-4


def test_hcs_self_consistent_builds_without_theta_table(tmp_path):
    _write_case(tmp_path, 0.80, 2.0, f_req=[0.6, 0.6], delta_E=[1.0, 1.0])

    payload = build_c0_table(
        tmp_path,
        AR=2.0,
        estimator="hcs-self-consistent",
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.4, one_hit=0.5),
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
        bootstrap_seed=123,
        fixed_point_samples=1000,
    )

    row = payload["diagnostics"]["rows"][0]
    assert payload["table"]["(0.800, 2.0)"] > 0.0
    assert row["estimator"] == "hcs-self-consistent"
    assert "theta_pred" in row
    assert np.isfinite(row["theta_pred"])


def test_hcs_grid_self_consistent_groups_theta_folders(tmp_path):
    root = tmp_path / "ctc"
    _write_case(root, 0.80, 1.5, f_req=[0.4, 0.4], delta_E=[1.0, 1.0], theta=0.5)
    _write_case(root, 0.80, 1.5, f_req=[0.7, 0.7], delta_E=[1.0, 1.0], theta=1.0)

    payload = build_c0_table(
        root,
        AR=1.5,
        estimator="hcs-grid-self-consistent",
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.4, one_hit=0.5),
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
        bootstrap_seed=123,
        fixed_point_samples=500,
        theta_scan_points=8,
    )

    rows = payload["diagnostics"]["rows"]
    assert len(rows) == 1
    assert rows[0]["theta_grid_count"] == 2
    assert payload["table"].keys() == {"(0.800, 1.5)"}


def test_hcs_grid_self_consistent_uses_interpolated_c0_at_predicted_theta(tmp_path):
    root = tmp_path / "ctc"
    _write_case(root, 0.80, 1.5, f_req=[0.3, 0.3], delta_E=[1.0, 1.0], theta=0.4)
    _write_case(root, 0.80, 1.5, f_req=[0.9, 0.9], delta_E=[1.0, 1.0], theta=1.2)

    payload = build_c0_table(
        root,
        AR=1.5,
        estimator="hcs-grid-self-consistent",
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.3, one_hit=0.5),
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
        bootstrap_seed=321,
        fixed_point_samples=500,
        theta_scan_points=8,
    )

    row = payload["diagnostics"]["rows"][0]
    expected = np.interp(
        row["theta_pred"],
        row["theta_grid_values"],
        row["C_mic_grid_values"],
    )
    assert np.isclose(payload["table"]["(0.800, 1.5)"], expected)


def test_hcs_grid_self_consistent_does_not_require_theta_target(tmp_path):
    root = tmp_path / "ctc"
    _write_case(root, 0.80, 1.5, f_req=[0.5], delta_E=[1.0], theta=0.5)
    _write_case(root, 0.80, 1.5, f_req=[0.6], delta_E=[1.0], theta=1.0)

    payload = build_c0_table(
        root,
        AR=1.5,
        estimator="hcs-grid-self-consistent",
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.4, one_hit=0.5),
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
        bootstrap_seed=123,
        fixed_point_samples=300,
        theta_scan_points=6,
    )

    assert payload["diagnostics"]["rows"][0]["estimator"] == "hcs-grid-self-consistent"


def test_hcs_grid_self_consistent_ignores_malformed_alpha_dirs(tmp_path):
    root = tmp_path / "ctc"
    _write_case(root, 0.80, 1.5, f_req=[0.5], delta_E=[1.0], theta=0.5)
    _write_case(root, 0.80, 1.5, f_req=[0.6], delta_E=[1.0], theta=1.0)
    bad = root / "alpha__r0.50_AR1.5"
    bad.mkdir(parents=True)
    np.savetxt(bad / "ftr_data.txt", [[-99.0, 0.0, 1.0]])
    np.savetxt(bad / "NPhit.txt", [1], fmt="%d")

    payload = build_c0_table(
        root,
        AR=1.5,
        estimator="hcs-grid-self-consistent",
        models=_FakeHCSModels(epsilon_tr_f=0.5, gamma_max=0.4, one_hit=0.5),
        beta_a=1.0,
        beta_b=1.0,
        bootstrap_samples=0,
        bootstrap_seed=123,
        fixed_point_samples=300,
        theta_scan_points=6,
    )

    row = payload["diagnostics"]["rows"][0]
    assert row["theta_grid_count"] == 2
    assert 99.0 not in row["C_mic_grid_values"]


def test_grid_theta_solver_reports_no_bracket(monkeypatch, tmp_path):
    case_dir = _write_case(tmp_path, 0.80, 1.5, f_req=[0.5], delta_E=[1.0], theta=0.5)
    case = CTCCase(alpha=0.80, theta=0.5, AR=1.5, path=case_dir)

    def always_positive(theta, theta_grid, C0_grid, case, models, beta_a, beta_b, n_samples, seed):
        return float(theta + 1.0), {"gamma_factor": 0.0}

    monkeypatch.setattr(ctc_c0, "_theta_grid_residual", always_positive)

    theta, diag = solve_grid_self_consistent_theta(
        theta_grid=np.array([0.5, 1.0]),
        C0_grid=np.array([1.0, 1.0]),
        case=case,
        models=_FakeHCSModels(epsilon_tr_f=0.5),
        beta_a=1.0,
        beta_b=1.0,
        n_samples=10,
        seed=1,
        scan_points=5,
    )

    assert diag["status"] == "no_bracket_min_abs_residual"
    assert np.isclose(theta, 0.5)


def test_ctc_c0_cli_writes_table_and_diagnostics(tmp_path):
    root = tmp_path / "ctc"
    _write_case(root, 0.90, 2.0, f_req=[0.6, 0.6, 0.6], delta_E=[1.0, 2.0, 3.0])
    _write_case(root, 1.00, 2.0, f_req=[0.0, 0.0], delta_E=[1.0, 1.0])
    compare = tmp_path / "old_C_alpha.json"
    compare.write_text(json.dumps({"(0.900, 2.0)": 0.25}))
    table = tmp_path / "C0_ctc_table_AR20.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.calibration.build_C0_ctc_table",
            "--root",
            str(root),
            "--AR",
            "2.0",
            "--output",
            str(table),
            "--compare-table",
            str(compare),
            "--bootstrap-samples",
            "0",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    table_payload = json.loads(table.read_text())
    diag_payload = json.loads(table.with_name("C0_ctc_diagnostics_AR20.json").read_text())
    csv_path = table.with_name("C0_ctc_diagnostics_AR20.csv")

    assert np.isclose(table_payload["(0.900, 2.0)"], 1.0)
    assert table_payload["(1.000, 2.0)"] == 0.0
    assert csv_path.exists()
    row_090 = [r for r in diag_payload["rows"] if np.isclose(r["alpha"], 0.90)][0]
    assert row_090["comparison_C_alpha"] == 0.25
    assert np.isclose(row_090["comparison_delta"], 0.75)


def test_generated_c0_table_loads_as_collisionmodels_c_alpha(tmp_path):
    root = tmp_path / "ctc"
    _write_case(root, 0.80, 2.0, f_req=[0.3, 0.3], delta_E=[1.0, 1.0])
    _write_case(root, 0.90, 2.0, f_req=[0.6, 0.6], delta_E=[1.0, 1.0])

    payload = build_c0_table(root, AR=2.0, bootstrap_samples=0)
    table = tmp_path / "C0_ctc_table_AR20.json"
    table.write_text(json.dumps(payload["table"]))

    models = CollisionModels(
        "models",
        gmm_npz_path="models/exchange_gmm/gmm_cond_AR20.npz",
        c_alpha_path=table,
    )

    assert np.isclose(models.get_C_alpha(0.80, 2.0), 0.5)
    assert np.isclose(models.get_C_alpha(0.85, 2.0), 0.75)
    assert np.isclose(models.get_C_alpha(0.90, 2.0), 1.0)
