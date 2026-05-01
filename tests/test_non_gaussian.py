import json

import numpy as np

from src.simulation.non_gaussian import (
    NonGaussianDiagnostics,
    cumulants_from_moments,
)


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

    config = {
        "diagnostics": {
            "non_gaussian": {
                "enabled": True,
                "start_tau": 0.0,
                "hist_tr_bins": 80,
                "hist_rot_bins": 80,
                "hist_speed_bins": 80,
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

    for suffix in [
        "_ng_hist_tr.txt",
        "_ng_hist_rot_component.txt",
        "_ng_hist_rot_speed.txt",
    ]:
        data = np.loadtxt(tmp_path / f"case{suffix}")
        integral = np.trapz(data[:, 1], data[:, 0])
        assert 0.90 < integral < 1.10
