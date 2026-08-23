import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.postprocessing.pof_hcs_figures import (
    DEFAULT_AR4_DSMC,
    DEFAULT_DEM_ROOT,
    HIST_ALPHAS,
    _hist_integral,
    load_ar4_dsmc,
    load_dem_case,
    load_hist_cases,
    load_macro_case,
    maxwell_rot_energy_pdf,
    maxwell_speed_pdf,
    validate_available_data,
)


def test_pof_loader_counts_completed_sweep():
    summary = validate_available_data(root="runs/paper_hcs_sweep")

    assert set(summary["macro_counts"].values()) == {3}
    assert set(summary["hist_counts"].values()) == {3}
    assert summary["external"]["ar4_dsmc"] is True
    assert all(summary["external"]["dem"].values())


def test_pof_loaders_parse_temperature_references():
    ar4 = load_ar4_dsmc(DEFAULT_AR4_DSMC)
    dem = load_dem_case(DEFAULT_DEM_ROOT, 4.0)
    macro = load_macro_case("runs/paper_hcs_sweep/macro", 2.0, 0.95)

    assert ar4.t.size > 10
    assert dem.t.size > 10
    assert len(macro) == 3
    assert np.all(np.isfinite(ar4.Ttotal))
    assert np.all(np.isfinite(dem.Ttotal))


def test_pof_histograms_integrate_and_references_are_finite():
    cases = load_hist_cases("runs/paper_hcs_sweep/hist")
    assert set(cases) == set(HIST_ALPHAS)

    for alpha, data in cases.items():
        speed = data["speed"]
        rot = data["rot_energy"]
        assert 0.8 < _hist_integral(speed) < 1.2
        assert 0.8 < _hist_integral(rot) < 1.2
        assert np.all(maxwell_speed_pdf(speed.x[:10]) >= 0.0)
        assert np.all(maxwell_rot_energy_pdf(rot.x[:10]) > 0.0)


def test_pof_speed_cli_writes_only_requested_figure(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "scripts.public.run_paper_figures",
        "--figure",
        "speed",
        "--root",
        "runs/paper_hcs_sweep",
        "--out-dir",
        str(tmp_path),
        "--formats",
        "png",
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)

    outputs = sorted(path.name for path in tmp_path.iterdir())
    assert outputs == ["fig4_reduced_speed_distribution.png"]


def test_pof_all_cli_writes_all_figures(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "scripts.public.run_paper_figures",
        "--figure",
        "all",
        "--root",
        "runs/paper_hcs_sweep",
        "--out-dir",
        str(tmp_path),
        "--formats",
        "png",
    ]
    subprocess.run(cmd, check=True, text=True, capture_output=True)

    outputs = sorted(path.name for path in tmp_path.iterdir())
    assert outputs == [
        "fig1_collision_time_scale.png",
        "fig2_hcs_cooling_law.png",
        "fig3_temperature_partition.png",
        "fig4_reduced_speed_distribution.png",
        "fig5_rotational_energy_distribution.png",
    ]
