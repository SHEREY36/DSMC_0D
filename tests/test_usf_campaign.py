import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prepare_usf_campaign_config import update_nested
from src.simulation.particle import model_ar_tag, result_ar_tag


def _base_config():
    return {
        "particle": {"AR": 2.0, "radius": 0.5, "mass": 1.0},
        "system": {
            "kTt": 1.0,
            "kTr": 1.0,
            "alpha": 1.0,
            "eta": 1.0,
            "phi": 0.01,
            "domain": [10, 10, 10],
        },
        "time": {"dt": 0.01, "dtau": 0.1, "t_end": 1.0},
        "flow": {"mode": "hcs", "shear_rate": 0.0},
        "simulation": {
            "seeds": [42],
            "output_dir": "results",
            "angular_transport_model": "current",
        },
        "preprocessing": {
            "model_output_dir": "models/",
            "gmm": {},
            "ftr": {"ftr_params_file": "models/ftr_params_AR20_r100.json"},
            "zr_eff": {"zr_eff_table_file": "models/zr_eff_table_AR20.json"},
            "dissipation": {"beta_a": 1.21, "beta_b": 3.67},
        },
        "calibration": {},
        "postprocessing": {},
        "calibration_sweep": {
            "output_root": "runs/base",
            "alpha_values": [0.8],
            "parallel_workers": 1,
        },
    }


def test_ar_tags_are_model_style_and_collision_safe():
    assert model_ar_tag(1.5) == "AR15"
    assert model_ar_tag(2.0) == "AR20"
    assert result_ar_tag(1.5) == "AR15"
    assert result_ar_tag(2.0) == "AR2"
    assert result_ar_tag(2.5) == "AR25"


def test_campaign_config_stress_weight_wires_ar_specific_models():
    cfg = update_nested(
        _base_config(),
        AR=2.5,
        mode="stress_weight",
        p_eta=0.6,
        output_root="runs/AR25_usf_peta060",
        workers=7,
        vss_table=None,
        seeds=[1, 2],
        alpha_values=[0.8, 0.9],
        t_end=123.0,
        dt=None,
        dtau=None,
    )

    assert cfg["particle"]["AR"] == 2.5
    assert cfg["flow"]["mode"] == "usf"
    assert cfg["simulation"]["angular_transport_model"] == "stress_weight"
    assert cfg["simulation"]["angular_transport_probability_override"] == 0.6
    assert cfg["preprocessing"]["gmm"]["gmm_cond_file"] == "models/gmm_cond_AR25.npz"
    assert cfg["calibration"]["C_alpha_table_file"] == "models/C_alpha_table_AR25.json"
    assert cfg["calibration_sweep"]["parallel_workers"] == 7
    assert cfg["calibration_sweep"]["t_end"] == 123.0


def test_campaign_config_vss_rank2_requires_and_wires_table():
    cfg = update_nested(
        _base_config(),
        AR=3.0,
        mode="vss_rank2",
        p_eta=None,
        output_root="runs/AR30_usf_vss_rank2",
        workers=5,
        vss_table="models/vss_alpha_eff_table_AR30_peta060.json",
        seeds=None,
        alpha_values=None,
        t_end=None,
        dt=None,
        dtau=None,
    )

    assert cfg["simulation"]["angular_transport_model"] == "vss_rank2"
    assert (
        cfg["simulation"]["vss_alpha_eff_table_file"]
        == "models/vss_alpha_eff_table_AR30_peta060.json"
    )
    assert "angular_transport_probability_override" not in cfg["simulation"]


def test_campaign_config_can_enable_rank2_C2_table():
    cfg = update_nested(
        _base_config(),
        AR=2.0,
        mode="vss_rank2",
        p_eta=None,
        output_root="runs/AR2_usf_vss_rank2_C2",
        workers=5,
        vss_table="models/vss_alpha_eff_table_AR20_peta060.json",
        seeds=None,
        alpha_values=None,
        t_end=None,
        dt=None,
        dtau=None,
        rank2_correction_enabled=True,
        C2_table="models/C2_table_AR20.json",
    )

    assert cfg["simulation"]["rank2_correction_enabled"] is True
    assert cfg["simulation"]["C2_table_file"] == "models/C2_table_AR20.json"


def test_campaign_config_probe_delta_disables_rank2_C2():
    cfg = update_nested(
        _base_config(),
        AR=2.0,
        mode="vss_rank2",
        p_eta=None,
        output_root="runs/AR2_usf_vss_rank2_probe_m010",
        workers=5,
        vss_table="models/vss_alpha_eff_table_AR20_peta060.json",
        seeds=None,
        alpha_values=None,
        t_end=None,
        dt=None,
        dtau=None,
        rank2_correction_enabled=True,
        C2_table="models/C2_table_AR20.json",
        ftr_rank0_probe_delta=-0.1,
    )

    assert cfg["simulation"]["rank2_correction_enabled"] is False
    assert cfg["simulation"]["ftr_rank0_probe_delta"] == -0.1


def test_generated_config_is_yaml_serializable(tmp_path):
    cfg = update_nested(
        _base_config(),
        AR=1.5,
        mode="current",
        p_eta=None,
        output_root="runs/AR15_usf_current",
        workers=1,
        vss_table=None,
        seeds=[42],
        alpha_values=[0.8],
        t_end=None,
        dt=None,
        dtau=None,
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    loaded = yaml.safe_load(path.read_text())
    assert loaded["preprocessing"]["gmm"]["gmm_cond_file"] == "models/gmm_cond_AR15.npz"
