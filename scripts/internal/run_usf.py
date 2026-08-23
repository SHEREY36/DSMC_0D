#!/usr/bin/env python3
"""Internal runner for USF/rank-two DSMC studies."""

import argparse

import yaml

from src.simulation.collision import CollisionModels
from src.internal.usf.dsmc import run_all_realizations


def load_models(config):
    if config.get("simulation", {}).get("sphere_collision", False):
        return None

    model_dir = config["preprocessing"]["model_output_dir"]
    return CollisionModels(
        model_dir,
        gmm_npz_path=config["preprocessing"]["gmm"].get("gmm_cond_file"),
        ftr_params_path=config["preprocessing"].get("ftr", {}).get("ftr_params_file"),
        zr_eff_path=config.get("preprocessing", {}).get("zr_eff", {}).get(
            "zr_eff_table_file"
        ),
        c_alpha_path=config.get("calibration", {}).get("C_alpha_table_file"),
        stress_transport_path=config.get("simulation", {}).get(
            "stress_transport_weight_file"
        ),
        ctc_angular_path=config.get("simulation", {}).get("ctc_angular_file"),
        vss_alpha_eff_path=config.get("simulation", {}).get(
            "vss_alpha_eff_table_file"
        ),
        C2_path=(
            config.get("simulation", {}).get("C2_table_file")
            if config.get("simulation", {}).get("rank2_correction_enabled", False)
            else None
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/usf_sweep.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    models = load_models(config)
    run_all_realizations(config, models)


if __name__ == "__main__":
    main()
