import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.non_gaussian import (
    NonGaussianDiagnostics,
    cumulants_from_moments,
)
from src.postprocessing.non_gaussian import maxwell_energy_coupling_pdf


def test_cumulants_are_zero_for_reference_moments():
    values = cumulants_from_moments(c4=15.0 / 4.0, c6=105.0 / 8.0,
                                    w4=2.0, c2w2=1.5)

    assert abs(values["a2_tr"]) < 1e-14
    assert abs(values["a3_tr"]) < 1e-14
    assert abs(values["a2_rot"]) < 1e-14
    assert abs(values["a11"]) < 1e-14


def test_diagnostics_write_finite_hcs_outputs(tmp_path):
    rng = np.random.default_rng(123)
    Np = 2000
    mass = 1.0
    mI = 2.0
    Ttrans = 1.0
    Trot = 0.75
    vel = rng.normal(0.0, np.sqrt(Ttrans / mass), size=(Np, 3))
    vel -= vel.mean(axis=0)
    Er = rng.exponential(Trot, size=Np)
    Er *= Trot / np.mean(Er)

    config = {
        "diagnostics": {
            "non_gaussian": {
                "enabled": True,
                "sample_start_tau": 0.0,
                "hist_speed_bins": 80,
                "hist_rot_speed_bins": 80,
                "hist_energy_tr_bins": 80,
                "hist_energy_rot_bins": 80,
                "hist_energy_coupling_bins": 80,
            }
        }
    }
    output_path = tmp_path / "case.txt"
    diag = NonGaussianDiagnostics(
        config, str(output_path), seed=42, Np=Np, sphere_mode=False,
        flow_mode="hcs", mass=mass, mI=mI, t_end=2.0
    )
    diag.maybe_sample(0.0, 0.0, 0, vel, Er, Ttrans, Trot)
    diag.close(final_NColl=1000, final_tau=0.5)

    summary = json.loads((tmp_path / "case_ng_summary.json").read_text())
    assert summary["n_output_samples"] == 1
    assert summary["n_particle_samples"] == Np
    assert np.isfinite(summary["cumulants"]["a2_tr"])
    assert np.isfinite(summary["cumulants"]["a2_rot"])
    assert np.isfinite(summary["cumulants"]["a11"])
    assert np.isclose(summary["moments"]["c2"], 1.5, atol=2.0e-3)
    assert np.isclose(summary["moments"]["w2"], 1.0, atol=2.0e-3)

    for suffix in [
        "_ng_hist_speed.txt",
        "_ng_hist_rot_speed.txt",
        "_ng_hist_energy_tr.txt",
        "_ng_hist_energy_rot.txt",
        "_ng_hist_energy_coupling.txt",
    ]:
        data = np.loadtxt(tmp_path / f"case{suffix}")
        width = data[1, 0] - data[0, 0]
        integral = float(np.sum(data[:, 1] * width))
        assert 0.90 < integral < 1.10

    assert not (tmp_path / "case_ng_hist_rot_component.txt").exists()


def test_diagnostics_sample_scheduled_tau_window(tmp_path):
    rng = np.random.default_rng(321)
    Np = 500
    mass = 1.0
    Ttrans = 1.0
    Trot = 1.0
    vel = rng.normal(0.0, np.sqrt(Ttrans / mass), size=(Np, 3))
    Er = rng.exponential(Trot, size=Np)
    Er *= Trot / np.mean(Er)
    config = {
        "diagnostics": {
            "non_gaussian": {
                "enabled": True,
                "sample_start_tau": 1.0,
                "sample_end_tau": 3.0,
                "sample_delta_tau": 1.0,
                "hist_speed_bins": 40,
                "hist_rot_speed_bins": 40,
                "hist_energy_tr_bins": 40,
                "hist_energy_rot_bins": 40,
                "hist_energy_coupling_bins": 40,
            }
        }
    }
    diag = NonGaussianDiagnostics(
        config, str(tmp_path / "case.txt"), seed=7, Np=Np, sphere_mode=False,
        flow_mode="hcs", mass=mass, mI=1.0, t_end=5.0
    )
    for idx, tau in enumerate([0.5, 1.0, 2.0, 3.0, 4.0]):
        diag.maybe_sample(float(idx), tau, idx, vel, Er, Ttrans, Trot)
    diag.close(final_NColl=1500, final_tau=3.0)

    summary = json.loads((tmp_path / "case_ng_summary.json").read_text())
    assert summary["expected_output_samples"] == 3
    assert summary["n_output_samples"] == 3
    assert summary["sampling_complete"] is True


def test_diagnostics_sample_scheduled_physical_time_crossings(tmp_path):
    rng = np.random.default_rng(456)
    Np = 500
    mass = 1.0
    Ttrans = 1.0
    Trot = 1.0
    vel = rng.normal(0.0, np.sqrt(Ttrans / mass), size=(Np, 3))
    Er = rng.exponential(Trot, size=Np)
    Er *= Trot / np.mean(Er)
    config = {
        "diagnostics": {
            "non_gaussian": {
                "enabled": True,
                "sample_axis": "t",
                "sample_start_t": 150.0,
                "sample_end_t": 400.0,
                "sample_delta_t": 25.0,
                "hist_speed_bins": 40,
                "hist_rot_speed_bins": 40,
                "hist_energy_tr_bins": 40,
                "hist_energy_rot_bins": 40,
                "hist_energy_coupling_bins": 40,
            }
        }
    }
    diag = NonGaussianDiagnostics(
        config, str(tmp_path / "case.txt"), seed=8, Np=Np, sphere_mode=False,
        flow_mode="hcs", mass=mass, mI=1.0, t_end=405.0
    )
    times = [149.999999] + [150.0 + 25.0 * idx + 1.0e-8 for idx in range(11)]
    for idx, t in enumerate(times):
        diag.maybe_sample(t, float(idx), idx, vel, Er, Ttrans, Trot)
    diag.close(final_NColl=1500, final_tau=3.0, final_t=405.0)

    summary = json.loads((tmp_path / "case_ng_summary.json").read_text())
    assert summary["sample_axis"] == "t"
    assert summary["expected_output_samples"] == 11
    assert summary["n_output_samples"] == 11
    assert summary["sampling_complete"] is True


def test_maxwell_energy_coupling_reference_integrates_to_one():
    x = np.linspace(0.0, 80.0, 20000)
    integral = np.trapezoid(maxwell_energy_coupling_pdf(x), x)
    assert 0.995 < integral < 1.005
