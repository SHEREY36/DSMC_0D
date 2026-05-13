import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.particle import compute_particle_params
from src.simulation.usf_c2_calibration import (
    build_usf_C2_table,
    infer_C2_from_theta_gap,
    reduced_kinetic_a2,
)


def test_reduced_kinetic_a2_is_zero_for_isotropic_pressure():
    assert reduced_kinetic_a2(np.eye(3)) == 0.0


def test_reduced_kinetic_a2_matches_known_deviatoric_tensor():
    reduced = np.array([
        [1.4, 0.1, 0.0],
        [0.1, 0.8, 0.0],
        [0.0, 0.0, 0.8],
    ])
    dev = reduced - np.eye(3)
    expected = np.trace(dev @ dev) / 8.0

    assert np.isclose(reduced_kinetic_a2(reduced), expected)
    assert np.allclose(
        reduced_kinetic_a2(np.stack([np.eye(3), reduced])),
        [0.0, expected],
    )


def test_infer_C2_from_theta_gap_handles_sign_and_degenerate_cases():
    result = infer_C2_from_theta_gap(
        theta_dsmc=1.0,
        theta_lammps=1.2,
        a2_steady=0.1,
        C_alpha=0.5,
        theta_probe=1.05,
        probe_delta=-0.1,
    )
    assert result["valid"]
    assert result["chi"] < 0.0
    assert result["C2"] < 0.0

    zero_gap = infer_C2_from_theta_gap(
        theta_dsmc=1.0,
        theta_lammps=1.0,
        a2_steady=0.1,
        C_alpha=0.5,
        theta_probe=1.05,
        probe_delta=-0.1,
    )
    assert zero_gap["valid"]
    assert np.isclose(zero_gap["C2"], 0.0)

    bad = infer_C2_from_theta_gap(
        theta_dsmc=1.0,
        theta_lammps=1.2,
        a2_steady=1.0e-20,
        C_alpha=0.5,
        theta_probe=1.05,
        probe_delta=-0.1,
    )
    assert not bad["valid"]
    assert bad["status"] == "near_zero_a2"
    assert bad["C2"] == 0.0

    missing_probe = infer_C2_from_theta_gap(
        theta_dsmc=1.0,
        theta_lammps=1.2,
        a2_steady=0.1,
        C_alpha=0.5,
    )
    assert not missing_probe["valid"]
    assert missing_probe["status"] == "missing_probe_sensitivity"


def _write_pressure_file(path, n_density, T_tr, reduced_tensor, n_rows):
    Pk = n_density * T_tr * reduced_tensor
    rows = []
    for i in range(n_rows):
        t = float(i)
        rows.append([
            t,
            t * 0.1,
            Pk[0, 0],
            Pk[0, 1],
            Pk[0, 2],
            Pk[1, 1],
            Pk[1, 2],
            Pk[2, 2],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ])
    np.savetxt(path, np.asarray(rows))


def test_build_usf_C2_table_reads_fixture_and_writes_expected_json(tmp_path):
    dsmc_root = tmp_path / "runs" / "AR2_usf_vss_rank2"
    probe_root = tmp_path / "runs" / "AR2_usf_vss_rank2_probe_m010"
    case_dir = dsmc_root / "alpha_050"
    probe_case_dir = probe_root / "alpha_050"
    results_dir = case_dir / "results"
    probe_results_dir = probe_case_dir / "results"
    results_dir.mkdir(parents=True)
    probe_results_dir.mkdir(parents=True)

    config = {
        "particle": {"AR": 2.0, "radius": 0.5, "mass": 1.0},
        "system": {
            "alpha": 0.5,
            "phi": 0.01,
            "domain": [50.0, 50.0, 50.0],
        },
        "simulation": {
            "seeds": [11, 12],
            "output_dir": str(results_dir),
            "sphere_collision": False,
        },
    }
    with open(case_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    probe_config = json.loads(json.dumps(config))
    probe_config["simulation"]["output_dir"] = str(probe_results_dir)
    probe_config["simulation"]["ftr_rank0_probe_delta"] = -0.1
    with open(probe_case_dir / "config.yaml", "w") as f:
        yaml.safe_dump(probe_config, f)

    params = compute_particle_params(config)
    volume = np.prod(config["system"]["domain"])
    n_density = np.ceil(config["system"]["phi"] * volume / params.volume) / volume
    T_tr = 2.0
    T_rot = 1.0
    T_total = (3.0 * T_tr + 2.0 * T_rot) / 5.0
    reduced_tensor = np.array([
        [1.4, 0.1, 0.0],
        [0.1, 0.8, 0.0],
        [0.0, 0.0, 0.8],
    ])
    expected_a2 = reduced_kinetic_a2(reduced_tensor)

    for realization in [1, 2]:
        temp_rows = np.column_stack([
            np.arange(20, dtype=float),
            0.1 * np.arange(20, dtype=float),
            np.full(20, T_tr),
            np.full(20, T_rot),
            np.full(20, T_total),
        ])
        np.savetxt(results_dir / f"AR2_COR50_USF_R{realization}.txt", temp_rows)
        probe_temp_rows = np.column_stack([
            np.arange(20, dtype=float),
            0.1 * np.arange(20, dtype=float),
            np.full(20, 2.1),
            np.full(20, 1.0),
            np.full(20, (3.0 * 2.1 + 2.0) / 5.0),
        ])
        np.savetxt(
            probe_results_dir / f"AR2_COR50_USF_R{realization}.txt",
            probe_temp_rows,
        )
        _write_pressure_file(
            results_dir / f"AR2_COR50_USF_R{realization}_pressure.txt",
            n_density,
            T_tr,
            reduced_tensor,
            20,
        )
        _write_pressure_file(
            probe_results_dir / f"AR2_COR50_USF_R{realization}_pressure.txt",
            n_density,
            2.1,
            reduced_tensor,
            20,
        )

    lammps_root = tmp_path / "LAMMPS_data" / "USF2" / "AR2"
    lammps_case = lammps_root / "e_050"
    lammps_case.mkdir(parents=True)
    lammps_temp = np.column_stack([
        np.arange(20, dtype=float),
        np.full(20, 2.2),
        np.full(20, 1.0),
        np.full(20, 1.0),
        np.full(20, 1.0 / 2.2),
        np.zeros(20),
        np.zeros(20),
    ])
    np.savetxt(lammps_case / "temperature_stats.dat", lammps_temp)
    lammps_shear = np.column_stack([
        np.arange(20, dtype=float),
        np.full(20, 1.4),
        np.full(20, 0.8),
        np.full(20, 0.8),
        np.full(20, -0.3),
        np.zeros(20),
        np.zeros(20),
    ])
    np.savetxt(lammps_case / "shear_stats.dat", lammps_shear)

    C_alpha_path = tmp_path / "models" / "C_alpha_table_AR20.json"
    C_alpha_path.parent.mkdir()
    C_alpha_path.write_text(json.dumps({"(0.500, 2.0)": 0.5}))
    output = tmp_path / "models" / "C2_table_AR20.json"

    payload = build_usf_C2_table(
        dsmc_root=dsmc_root,
        lammps_root=lammps_root,
        C_alpha_table_file=C_alpha_path,
        AR=2.0,
        probe_root=probe_root,
        output_path=output,
        smooth_window=5,
    )
    row = payload["rows"][0]

    assert output.exists()
    assert payload["metadata"]["method"] == "usf_theta_gap_measured_sensitivity"
    assert row["valid"]
    assert row["n_dsmc_seeds"] == 2
    assert np.isclose(row["theta_DSMC"], 2.0)
    assert np.isclose(row["theta_LAMMPS"], 2.2)
    assert np.isclose(row["theta_probe"], 2.1)
    assert np.isclose(row["probe_delta"], -0.1)
    assert row["chi_DSMC"] < 0.0
    assert np.isclose(row["a2_steady"], expected_a2)
    assert row["C2"] < 0.0
    assert "dsmc_seed_rows" in row
