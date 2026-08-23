import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.collision import CollisionModels
from src.simulation.dsmc import run_simulation
from src.simulation.ntc import NTCWorkspace


def _base_config(tmp_path, sphere=False):
    return {
        "particle": {
            "AR": 1.0 if sphere else 2.0,
            "radius": 0.5,
            "mass": 1.0,
            "sigma_c_scale": 1.020,
        },
        "system": {
            "kTt": 1.0,
            "kTr": 1.0,
            "alpha": 0.95,
            "eta": 1.0,
            "phi": 0.01,
            "domain": [20, 20, 20],
            "C_alpha": None,
        },
        "time": {
            "dt": 0.02,
            "dtau": 0.2,
            "t_end": 0.08,
            "tau_end": None,
            "equilibration_time": 0.0,
        },
        "flow": {"mode": "hcs", "shear_rate": 0.0},
        "simulation": {
            "seeds": [42],
            "output_dir": str(tmp_path),
            "sphere_collision": bool(sphere),
            "use_isotropic_eps": True,
        },
        "preprocessing": {
            "model_output_dir": "models/",
            "gmm": {"gmm_cond_file": "models/exchange_gmm/gmm_cond_AR20.npz"},
            "dissipation": {"beta_a": 1.21, "beta_b": 3.67},
        },
        "calibration": {
            "C_alpha_table_file": "models/relaxation/C_alpha_table_AR20.json"
        },
    }


def test_ntc_workspace_screens_valid_pairs():
    workspace = NTCWorkspace(capacity=4, seed=123)
    vel = np.random.default_rng(1).normal(size=(12, 3))
    vrmax_temp, accepted = workspace.screen_candidates(vel, Np=12, n=50, vrmax=10.0)

    assert vrmax_temp >= 0.0
    assert np.all(workspace.p1[:50] != workspace.p2[:50])
    assert np.all(accepted >= 0)
    assert np.all(accepted < 50)


def test_collision_models_load_categorized_artifacts():
    models = CollisionModels(
        "models/",
        gmm_npz_path="models/exchange_gmm/gmm_cond_AR20.npz",
        c_alpha_path="models/relaxation/C_alpha_table_AR20.json",
    )

    assert models.cond_gmm is not None
    assert models.get_one_hit(0.95, 2.0) > 0.0
    assert models.get_C_alpha(0.95, 2.0) > 0.0


def test_short_public_hcs_sphere_run(tmp_path):
    cfg = _base_config(tmp_path, sphere=True)
    output = tmp_path / "sphere.txt"

    run_simulation(cfg, models=None, seed=42, output_path=str(output))

    data = np.loadtxt(output)
    data = np.atleast_2d(data)
    assert data.shape[1] == 5
    assert np.all(np.isfinite(data))


def test_short_public_hcs_spherocylinder_run(tmp_path):
    cfg = _base_config(tmp_path, sphere=False)
    models = CollisionModels(
        "models/",
        gmm_npz_path=cfg["preprocessing"]["gmm"]["gmm_cond_file"],
        c_alpha_path=cfg["calibration"]["C_alpha_table_file"],
    )
    output = tmp_path / "spherocylinder.txt"

    run_simulation(cfg, models=models, seed=42, output_path=str(output))

    data = np.loadtxt(output)
    data = np.atleast_2d(data)
    assert data.shape[1] == 5
    assert np.all(data[:, 2:] >= 0.0)


def test_public_runner_rejects_usf(tmp_path):
    cfg = _base_config(tmp_path, sphere=True)
    cfg = copy.deepcopy(cfg)
    cfg["flow"]["mode"] = "usf"
    config_path = tmp_path / "usf.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.public.run_simulation",
            "--config",
            str(config_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "supports only flow.mode='hcs'" in result.stderr
