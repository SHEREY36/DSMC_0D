"""Reviewer-facing 0D DSMC driver for homogeneous cooling.

This module intentionally contains only the public HCS time march.  Research
extensions such as USF, rank-two probes, calibration campaigns, and
non-Gaussian production sweeps live under ``src.internal`` or ``src.calibration``.
"""

import math
import os

import numpy as np

from .ntc import NTCWorkspace, candidate_count
from .observables import ensure_parent, temperatures, write_temperature_row
from .particle import compute_particle_params, result_ar_tag
from .state import initialize_particles, reference_temperature, rescale_hcs_state
from .kernels.sphere import SphereCollisionKernel
from .kernels.spherocylinder_hcs import SpherocylinderHCSKernel
from .kernels.spherocylinder_hcs import chi_hs as _chi_hs


OUTPUT_BUFFER_SIZE = 65536


def _validate_public_hcs_config(config):
    flow_mode = config.get("flow", {}).get("mode", "hcs")
    if flow_mode != "hcs":
        raise ValueError(
            "The public DSMC driver supports only flow.mode='hcs'. "
            "Use the internal USF runner for uniform-shear/rank-two studies."
        )

    sim_cfg = config.get("simulation", {})
    if sim_cfg.get("rank2_correction_enabled", False):
        raise ValueError(
            "rank2_correction_enabled is an internal USF extension and is not "
            "available in the public HCS driver."
        )
    if sim_cfg.get("ftr_rank0_probe_delta") is not None:
        raise ValueError(
            "ftr_rank0_probe_delta is an internal sensitivity probe and is not "
            "available in the public HCS driver."
        )


def _output_buffer_size(config):
    value = int(config.get("simulation", {}).get("output_buffer_size", OUTPUT_BUFFER_SIZE))
    if value == 0 or value < -1:
        raise ValueError(
            "simulation.output_buffer_size must be -1, 1, or a positive integer, "
            f"got {value}"
        )
    return value


def _time_settings(config):
    dt = float(config["time"]["dt"])
    dtau = float(config["time"]["dtau"])
    t_end = float(config["time"]["t_end"])
    tau_end = config["time"].get("tau_end")
    if tau_end is not None:
        tau_end = float(tau_end)
        if tau_end <= 0.0:
            raise ValueError(f"time.tau_end must be > 0 when set, got {tau_end}")
    equilibration_time = float(config["time"].get("equilibration_time", 0.0))
    if equilibration_time < 0.0:
        raise ValueError(
            f"time.equilibration_time must be >= 0, got {equilibration_time}"
        )
    return dt, dtau, t_end, tau_end, equilibration_time


def _build_kernel(config, params, models, sphere_mode, equilibration_time):
    alpha = float(config["system"]["alpha"])
    if sphere_mode:
        return SphereCollisionKernel(alpha)

    if models is None:
        raise ValueError("Spherocylinder mode requires a CollisionModels instance.")

    if alpha < 1.0:
        gamma_max = models.get_gamma_max(alpha, params.AR)
        prob_one_hit = models.get_one_hit(alpha, params.AR)
    else:
        gamma_max = 0.0
        prob_one_hit = 1.0

    sim_cfg = config.get("simulation", {})
    return SpherocylinderHCSKernel(
        params=params,
        models=models,
        alpha=alpha,
        eta=float(config["system"].get("eta", 1.0)),
        beta_a=float(config["preprocessing"]["dissipation"]["beta_a"]),
        beta_b=float(config["preprocessing"]["dissipation"]["beta_b"]),
        C_alpha=float(config["system"].get("C_alpha") or models.get_C_alpha(alpha, params.AR)),
        gamma_max=gamma_max,
        prob_one_hit=prob_one_hit,
        equilibration_time=equilibration_time,
        use_isotropic_eps=bool(sim_cfg.get("use_isotropic_eps", True)),
    )


def _should_continue(t, t_end, tau, tau_end, Ntau, dtau):
    return t < t_end and (
        tau_end is None
        or tau < tau_end
        or tau >= Ntau * dtau
    )


def run_simulation(config, models, seed, output_path, pressure_path=None):
    """Run one public HCS DSMC realization.

    The output format is unchanged:
    ``t tau T_trans T_rot T_total``.
    """
    _validate_public_hcs_config(config)
    np.random.seed(seed)

    params = compute_particle_params(config)
    sphere_mode = bool(config.get("simulation", {}).get("sphere_collision", False))

    alpha = float(config["system"]["alpha"])
    eta = float(config["system"].get("eta", 1.0))
    if eta <= 0.0:
        raise ValueError(f"system.eta must be > 0, got {eta}")

    kTt = float(config["system"]["kTt"])
    kTr = float(config["system"]["kTr"])
    phi = float(config["system"]["phi"])
    lx, ly, lz = config["system"]["domain"]
    volume = float(lx * ly * lz)
    Np = math.ceil(phi * volume / params.volume)

    dt, dtau, t_end, tau_end, equilibration_time = _time_settings(config)
    output_buffer_size = _output_buffer_size(config)

    hcs_rescale_temperature = bool(
        config.get("simulation", {}).get("hcs_rescale_temperature", False)
    )
    hcs_rescale_reference = config.get("simulation", {}).get(
        "hcs_rescale_reference", "initial"
    )
    if hcs_rescale_reference != "initial":
        raise ValueError(
            "simulation.hcs_rescale_reference currently supports only 'initial'"
        )
    hcs_rescale_vrmax_policy = config.get("simulation", {}).get(
        "hcs_rescale_vrmax_policy", "reset"
    )
    if hcs_rescale_vrmax_policy not in ("reset", "scale"):
        raise ValueError(
            "simulation.hcs_rescale_vrmax_policy must be 'reset' or 'scale', "
            f"got {hcs_rescale_vrmax_policy!r}"
        )

    vel, _, Er = initialize_particles(
        Np, kTt, kTr, params.mass, params.mI, sphere_mode=sphere_mode
    )
    kernel = _build_kernel(config, params, models, sphere_mode, equilibration_time)
    target_temperature = reference_temperature(kTt, kTr, sphere_mode=sphere_mode)
    vrmax = 5.0 * np.sqrt(2.0) * np.sqrt(kTt * params.omass)

    print(
        f"  Np={Np}, alpha={alpha:.4f}, AR={params.AR:g}, "
        f"sphere_collision={sphere_mode}, sigma_c={params.sigma_c:.6f}, "
        f"hcs_rescale={hcs_rescale_temperature}"
    )

    t = 0.0
    NColl = 0
    Ntau = 0
    ntc_workspace = NTCWorkspace(capacity=1024, seed=seed)

    ensure_parent(output_path)
    with open(output_path, "w", buffering=output_buffer_size) as file:
        while _should_continue(t, t_end, NColl / float(Np), tau_end, Ntau, dtau):
            tau = NColl / float(Np)
            if tau >= Ntau * dtau:
                write_temperature_row(file, t, tau, vel, Er, params, Np)
                Ntau += 1

            Ttrans, Trot, _ = temperatures(vel, Er, params, Np)
            temp_ratio = Ttrans / Trot if Trot > 0.0 else 1.0

            n_cands = candidate_count(Np, params.sigma_c, vrmax, volume, dt)
            vrmax_temp = 0.0
            if n_cands > 0:
                vrmax_temp, accepted_idx = ntc_workspace.screen_candidates(
                    vel, Np, n_cands, vrmax
                )
                for k in accepted_idx:
                    p1 = int(ntc_workspace.p1[k])
                    p2 = int(ntc_workspace.p2[k])
                    eij = ntc_workspace.eij[k].copy()

                    v1 = vel[p1].copy()
                    v2 = vel[p2].copy()
                    vrel_vec = v1 - v2
                    cr = float(np.dot(eij, vrel_vec))
                    if cr < 0.0:
                        eij = -eij
                        cr = -cr
                    vr = float(np.linalg.norm(vrel_vec))

                    NColl += kernel.collide(
                        vel, Er, p1, p2, eij, v1, v2, vrel_vec, cr, vr, t,
                        temp_ratio
                    )

            if vrmax < vrmax_temp:
                vrmax = vrmax_temp

            if hcs_rescale_temperature:
                scale = rescale_hcs_state(
                    vel, Er, params, Np, target_temperature,
                    sphere_mode=sphere_mode
                )
                if hcs_rescale_vrmax_policy == "scale":
                    vrmax *= scale
                else:
                    Ttrans_after, _, _ = temperatures(vel, Er, params, Np)
                    thermal_vrmax = (
                        5.0 * np.sqrt(2.0)
                        * np.sqrt(max(Ttrans_after, 1.0e-30) * params.omass)
                    )
                    vrmax = max(thermal_vrmax, vrmax_temp * scale)

            t += dt

    print(f"  Simulation complete. NColl={NColl}, output: {output_path}")


def run_all_realizations(config, models):
    """Run public HCS DSMC for all seeds specified in the config."""
    _validate_public_hcs_config(config)
    seeds = config["simulation"]["seeds"]
    output_dir = config["simulation"]["output_dir"]
    AR = config["particle"]["AR"]
    alpha = config["system"]["alpha"]
    ar_tag = result_ar_tag(AR)

    for i, seed in enumerate(seeds, start=1):
        filename = f"{ar_tag}_COR{int(alpha * 100)}_R{i}.txt"
        output_path = os.path.join(output_dir, filename)
        print(f"Running realization {i}/{len(seeds)} (seed={seed})...")
        run_simulation(config, models, seed, output_path)
