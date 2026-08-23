import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.public.run_hcs_paper_sweep import (
    _build_config,
    _validate_task_model_paths,
    build_tasks,
)


def test_default_paper_sweep_task_matrix(tmp_path):
    tasks = build_tasks(tmp_path, models_dir="models")

    macro = [task for task in tasks if task.group == "macro"]
    hist = [task for task in tasks if task.group == "hist"]
    assert len(macro) == 21
    assert len(hist) == 9
    assert len(tasks) == 30

    overlapping = [
        task for task in tasks
        if task.AR == 2.0 and task.alpha == 0.95 and task.seed == 42
    ]
    assert {Path(task.output_path).parts[-5] for task in overlapping} == {
        "macro", "hist"
    }


def test_paper_sweep_model_paths_match_each_task_ar(tmp_path):
    with open("config/release_hcs.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    for task in build_tasks(tmp_path, models_dir="models"):
        cfg = _build_config(base_config, task, "models", [100, 100, 100], 0.01)
        _validate_task_model_paths(task, cfg)
        label = f"AR{int(round(task.AR * 10)):02d}"
        assert label in Path(cfg["preprocessing"]["gmm"]["gmm_cond_file"]).name
        assert label in Path(cfg["calibration"]["C_alpha_table_file"]).name


def test_paper_sweep_dry_run_writes_manifest_only(tmp_path):
    cmd = [
        sys.executable,
        "-m",
        "scripts.public.run_hcs_paper_sweep",
        "--output-root",
        str(tmp_path),
        "--groups",
        "hist",
        "--seeds",
        "42",
        "--dry-run",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)

    assert "Prepared 3 task(s): 0 macro, 3 hist" in result.stdout
    assert (tmp_path / "paper_sweep_manifest.csv").exists()
    assert (tmp_path / "paper_sweep_manifest.json").exists()
    assert not list(tmp_path.glob("hist/**/results/*.txt"))
