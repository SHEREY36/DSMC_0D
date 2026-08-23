#!/usr/bin/env python3
"""Generate AR/mode-specific USF sweep configs for the VSS campaign."""
import argparse
import copy
import os
from pathlib import Path

import yaml

from src.simulation.particle import model_ar_tag


DEFAULT_ALPHA_VALUES = [
    0.50, 0.55, 0.60, 0.65, 0.70,
    0.75, 0.80, 0.85, 0.90, 0.95,
]


def _parse_csv_floats(text):
    if text is None or str(text).strip() == "":
        return None
    return [float(part.strip()) for part in str(text).split(",") if part.strip()]


def _parse_csv_ints(text):
    if text is None or str(text).strip() == "":
        return None
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def peta_tag(p_eta):
    return f"peta{int(round(float(p_eta) * 100.0)):03d}"


def probe_tag(delta):
    value = int(round(abs(float(delta)) * 100.0))
    sign = "m" if float(delta) < 0.0 else "p"
    return f"probe_{sign}{value:03d}"


def default_output_root(AR, mode, p_eta=None, rank2_correction_enabled=False,
                        ftr_rank0_probe_delta=None):
    ar_tag = model_ar_tag(AR)
    if ftr_rank0_probe_delta is not None:
        suffix = f"_{probe_tag(ftr_rank0_probe_delta)}"
    else:
        suffix = "_C2" if rank2_correction_enabled else ""
    if mode == "stress_weight":
        return f"runs/{ar_tag}_usf_{peta_tag(p_eta)}{suffix}"
    if mode == "vss_rank2":
        return f"runs/{ar_tag}_usf_vss_rank2{suffix}"
    return f"runs/{ar_tag}_usf_current{suffix}"


def default_lammps_dir(AR):
    tag = model_ar_tag(AR)
    if tag == "AR20":
        return "LAMMPS_data/USF/sphcyl_USF_AR2"
    return f"LAMMPS_data/USF/{tag}"


def update_nested(cfg, AR, mode, p_eta, output_root, workers, vss_table,
                  seeds, alpha_values, t_end, dt, dtau,
                  rank2_correction_enabled=False, C2_table=None,
                  ftr_rank0_probe_delta=None):
    ar_tag = model_ar_tag(AR)
    cfg = copy.deepcopy(cfg)

    cfg.setdefault("particle", {})
    cfg["particle"]["AR"] = float(AR)

    cfg.setdefault("flow", {})
    cfg["flow"]["mode"] = "usf"

    cfg.setdefault("simulation", {})
    sim = cfg["simulation"]
    sim["output_dir"] = os.path.join(output_root, "results")
    sim["sphere_collision"] = False
    sim.pop("angular_transport_probability_override", None)
    sim.pop("stress_transport_weight_file", None)
    sim.pop("vss_alpha_eff_table_file", None)
    sim["rank2_correction_enabled"] = (
        bool(rank2_correction_enabled)
        and ftr_rank0_probe_delta is None
    )
    sim["C2_table_file"] = str(
        C2_table or f"models/relaxation/C2_table_{ar_tag}.json"
    )
    sim["ftr_rank0_probe_delta"] = (
        None if ftr_rank0_probe_delta is None
        else float(ftr_rank0_probe_delta)
    )

    if mode == "current":
        sim["angular_transport_model"] = "current"
    elif mode == "stress_weight":
        if p_eta is None:
            raise ValueError("--p-eta is required for --mode stress_weight")
        sim["angular_transport_model"] = "stress_weight"
        sim["angular_transport_probability_override"] = float(p_eta)
    elif mode == "vss_rank2":
        sim["angular_transport_model"] = "vss_rank2"
        if vss_table is None:
            raise ValueError("--vss-table is required for --mode vss_rank2")
        sim["vss_alpha_eff_table_file"] = str(vss_table)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if seeds is not None:
        sim["seeds"] = seeds

    cfg.setdefault("preprocessing", {})
    pre = cfg["preprocessing"]
    pre.setdefault("gmm", {})
    pre["gmm"]["gmm_cond_file"] = f"models/exchange_gmm/gmm_cond_{ar_tag}.npz"
    pre.setdefault("ftr", {})
    # f_tr tables are currently AR=2-only in this repo.  Leave absent for
    # other ARs so CollisionModels falls back cleanly.
    ftr_path = f"models/archive/ftr_params_{ar_tag}_r100.json"
    if Path(ftr_path).exists():
        pre["ftr"]["ftr_params_file"] = ftr_path
    else:
        pre["ftr"].pop("ftr_params_file", None)
    pre.setdefault("zr_eff", {})
    zr_path = f"models/targets/zr_eff_table_{ar_tag}.json"
    if Path(zr_path).exists():
        pre["zr_eff"]["zr_eff_table_file"] = zr_path
    else:
        pre["zr_eff"].pop("zr_eff_table_file", None)
    pre["model_output_dir"] = pre.get("model_output_dir", "models/")

    cfg.setdefault("calibration", {})
    cfg["calibration"]["C_alpha_table_file"] = (
        f"models/relaxation/C_alpha_table_{ar_tag}.json"
    )
    cfg["calibration"]["theta_target_table_file"] = (
        f"models/targets/theta_target_table_{ar_tag}.json"
    )

    cfg.setdefault("postprocessing", {})
    post = cfg["postprocessing"]
    post["results_dir"] = os.path.join(output_root, "results")
    post["figures_dir"] = os.path.join(output_root, "figures")
    post["sweep_root"] = output_root
    post["sweep_figures_dir"] = os.path.join(output_root, "figures")
    post["lammps_sphcyl_dir"] = default_lammps_dir(AR)

    cfg.setdefault("calibration_sweep", {})
    sweep = cfg["calibration_sweep"]
    sweep["AR"] = float(AR)
    sweep["output_root"] = output_root
    sweep["alpha_values"] = alpha_values or list(DEFAULT_ALPHA_VALUES)
    sweep["default_eta"] = float(sweep.get("default_eta", 1.0))
    sweep["parallel_workers"] = int(workers)
    if seeds is not None:
        sweep["seeds"] = seeds
    if t_end is not None:
        sweep["t_end"] = float(t_end)
    if dt is not None:
        sweep["dt"] = float(dt)
    if dtau is not None:
        sweep["dtau"] = float(dtau)

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Generate a parameterized USF sweep config for AR/p_eta/VSS campaigns."
    )
    parser.add_argument("--base-config", default="config/usf_sweep.yaml")
    parser.add_argument("--config-out", required=True)
    parser.add_argument("--AR", type=float, required=True)
    parser.add_argument(
        "--mode", choices=("current", "stress_weight", "vss_rank2"),
        default="current",
    )
    parser.add_argument("--p-eta", type=float, default=None)
    parser.add_argument("--vss-table", default=None)
    parser.add_argument(
        "--rank2-correction-enabled",
        action="store_true",
        help="Enable the USF C2*a2 correction in the generated DSMC config.",
    )
    parser.add_argument(
        "--C2-table",
        default=None,
        help="Rank-2 C2 table path. Defaults to models/relaxation/C2_table_AR*.json.",
    )
    parser.add_argument(
        "--ftr-rank0-probe-delta",
        type=float,
        default=None,
        help=(
            "Finite-difference probe delta for rank-0 f_tr. "
            "When set, rank2 correction is disabled in the generated config."
        ),
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seeds", default=None, help="Comma-separated integer seeds.")
    parser.add_argument(
        "--alpha-values",
        default=None,
        help="Comma-separated alpha values. Defaults to 0.50..0.95.",
    )
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--dtau", type=float, default=None)
    args = parser.parse_args()

    with open(args.base_config, "r") as f:
        base = yaml.safe_load(f)

    output_root = args.output_root or default_output_root(
        args.AR,
        args.mode,
        args.p_eta,
        rank2_correction_enabled=args.rank2_correction_enabled,
        ftr_rank0_probe_delta=args.ftr_rank0_probe_delta,
    )
    cfg = update_nested(
        base,
        AR=args.AR,
        mode=args.mode,
        p_eta=args.p_eta,
        output_root=output_root,
        workers=args.workers,
        vss_table=args.vss_table,
        seeds=_parse_csv_ints(args.seeds),
        alpha_values=_parse_csv_floats(args.alpha_values),
        t_end=args.t_end,
        dt=args.dt,
        dtau=args.dtau,
        rank2_correction_enabled=args.rank2_correction_enabled,
        C2_table=args.C2_table,
        ftr_rank0_probe_delta=args.ftr_rank0_probe_delta,
    )

    out_path = Path(args.config_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote USF campaign config: {out_path}")
    print(f"  AR={args.AR:g} ({model_ar_tag(args.AR)}) mode={args.mode}")
    if args.p_eta is not None:
        print(f"  p_eta={args.p_eta:g}")
    if args.vss_table:
        print(f"  vss_table={args.vss_table}")
    if args.rank2_correction_enabled:
        print(f"  C2_table={cfg['simulation']['C2_table_file']}")
    if args.ftr_rank0_probe_delta is not None:
        print(f"  ftr_rank0_probe_delta={args.ftr_rank0_probe_delta:g}")
    print(f"  output_root={output_root}")


if __name__ == "__main__":
    main()
