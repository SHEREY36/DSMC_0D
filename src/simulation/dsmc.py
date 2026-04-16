import os
import math
import numpy as np

from .particle import compute_particle_params
from .collision import (
    CollisionModels, init_p_chi_distribution, sample_chi,
    sample_dissp, update_velocities
)
from .pressure import compute_pij_k, accumulate_pij_c, normalise_pij_c
from src.preprocessing.relaxation import prepare_theta, Zr


def initialize_particles(Np, kTt, kTr, mass, mI):
    """Initialize particle velocities and rotational energies.

    Translational velocities: Maxwell-Boltzmann at temperature kTt.
    Rotational velocities: Maxwell-Boltzmann at temperature kTr.
    Bulk velocity is removed after initialization.

    Returns (vel, omega, Er) arrays.
    """
    omass = 1.0 / mass
    omI = 1.0 / mI
    sqkTt = np.sqrt(kTt * omass)
    sqkTr = np.sqrt(kTr * omI)

    vel = np.random.randn(Np, 3) * sqkTt
    omega = np.random.randn(Np, 3) * sqkTr
    omega[:, 0] = 0.0  # no rotation about symmetry axis
    Er = 0.5 * mI * (omega[:, 1]**2 + omega[:, 2]**2)

    # Remove bulk velocity
    vbulk = np.sum(vel, axis=0) / Np
    vel -= vbulk

    return vel, omega, Er


def run_simulation(config, models, seed, output_path, pressure_path):
    """Run a single DSMC realization.

    Parameters:
        config: dict from YAML config
        models: CollisionModels instance
        seed: random seed for this realization
        output_path: path for the temperature output .txt file
        pressure_path: path for the pressure tensor output .txt file
    """
    np.random.seed(seed)

    sphere_mode = config.get('simulation', {}).get('sphere_collision', False)

    # Particle properties
    params = compute_particle_params(config)

    # System properties
    alpha = config['system']['alpha']
    eta = config['system'].get('eta', 2.0)
    if eta <= 0.0:
        raise ValueError(f"system.eta must be > 0, got {eta}")
    kTt = config['system']['kTt']
    kTr = config['system']['kTr']
    phi = config['system']['phi']
    lx, ly, lz = config['system']['domain']
    volsys = lx * ly * lz
    ovol = 1.0 / volsys
    Np = math.ceil(phi * volsys / params.volume)

    # Dissipation parameters — skipped entirely in sphere mode
    if sphere_mode:
        gamma_max = 0.0
        prob_one_hit = 1.0
        beta_a = beta_b = 0.0
        use_zr_eff = False
        theta_star_eff = gmm_theta2 = None
        Z_R_eff_val = None
        C_alpha = 1.0
    else:
        if alpha < 1.0:
            gamma_max = models.get_gamma_max(alpha, params.AR)
            prob_one_hit = models.get_one_hit(alpha, params.AR)
        else:
            gamma_max = 0.0
            prob_one_hit = 1.0

        beta_a = config['preprocessing']['dissipation']['beta_a']
        beta_b = config['preprocessing']['dissipation']['beta_b']
        # Z_R_eff lookup (only used when simulation.use_zr_eff is true)
        use_zr_eff_cfg = config.get('simulation', {}).get('use_zr_eff', False)
        zr_eff_result = models.get_zr_eff(alpha, params.AR) if use_zr_eff_cfg else None
        if zr_eff_result is not None:
            theta_star_eff, Z_R_eff_val = zr_eff_result
            gmm_theta2 = prepare_theta(theta_star_eff)  # fixed GMM input at theta*
            use_zr_eff = True
        else:
            theta_star_eff = None
            gmm_theta2 = None
            use_zr_eff = False
            Z_R_eff_val = None

        # C_alpha: config override takes priority, then table lookup, then 1.0
        C_alpha = config['system'].get('C_alpha') or models.get_C_alpha(alpha, params.AR)

    # Flow mode
    flow_mode = config.get('flow', {}).get('mode', 'hcs')
    gdot = float(config.get('flow', {}).get('shear_rate', 0.0))
    if flow_mode not in ('hcs', 'usf'):
        raise ValueError(f"flow.mode must be 'hcs' or 'usf', got {flow_mode!r}")
    if flow_mode == 'usf' and gdot == 0.0:
        print("  Warning: USF mode with gdot=0 is equivalent to HCS.")

    # Time parameters
    dt = config['time']['dt']
    halfdt = dt * 0.5
    dtau = config['time']['dtau']
    t_end = config['time']['t_end']
    equilibration_time = config['time'].get('equilibration_time', 0.0)
    if equilibration_time < 0.0:
        raise ValueError(
            f"time.equilibration_time must be >= 0, got {equilibration_time}"
        )

    # Scattering angle distribution — skipped in sphere mode (isotropic scattering)
    if not sphere_mode:
        p_chi_fn_target, p_max_target = init_p_chi_distribution(
            params.AR, alpha, models
        )
        if equilibration_time > 0.0 and alpha < 1.0:
            p_chi_fn_eq, p_max_eq = init_p_chi_distribution(params.AR, 1.0, models)
        else:
            p_chi_fn_eq, p_max_eq = p_chi_fn_target, p_max_target
    else:
        p_chi_fn_target = p_max_target = p_chi_fn_eq = p_max_eq = None

    # Initialize particles
    vel, omega, Er = initialize_particles(
        Np, kTt, kTr, params.mass, params.mI
    )
    if sphere_mode:
        Er[:] = 0.0
        omega[:] = 0.0

    vrmax = 5.0 * np.sqrt(2.0) * np.sqrt(kTt * params.omass)

    NColl = 0
    t = 0.0
    Ntau = 0
    pij_c_acc = np.zeros((3, 3))
    t_last_output = 0.0

    flow_str = f"flow={flow_mode}" + (f", gdot={gdot:.4f}" if flow_mode == 'usf' else "")
    if sphere_mode:
        print(f"  Np={Np}, sphere_collision=True, alpha={alpha:.4f}, "
              f"sigma_c={params.sigma_c:.6f}, {flow_str}")
    elif use_zr_eff:
        print(f"  Np={Np}, Z_R_eff={Z_R_eff_val:.4f}, theta*={theta_star_eff:.4f}, "
              f"C_alpha={C_alpha:.4f}, gamma_max={gamma_max:.6f}, "
              f"prob_one_hit={prob_one_hit:.6f}, equilibration_time={equilibration_time:.3f}, "
              f"{flow_str}")
    else:
        print(f"  Np={Np}, eta={eta:.4f} (Zr), C_alpha={C_alpha:.4f}, "
              f"gamma_max={gamma_max:.6f}, prob_one_hit={prob_one_hit:.6f}, "
              f"equilibration_time={equilibration_time:.3f}, {flow_str}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', buffering=1) as file, open(pressure_path, 'w', buffering=1) as pfile:
        while t < t_end:
            # Output at intervals
            if NColl / Np >= Ntau * dtau:
                Eksum = (2.0 * 0.5 * params.mass
                         * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np))
                Ersum = np.sum(Er) / float(Np)
                T_total = (3.0 * Eksum + 2.0 * Ersum) / 5.0
                file.write(
                    f"{t:13.6f} {NColl / float(Np):13.6f} "
                    f"{Eksum:13.6f} {Ersum:13.6f} {T_total:13.6f}\n"
                )

                # Pressure tensor output
                pij_k = compute_pij_k(vel, params.mass, volsys)
                dt_output = t - t_last_output
                pij_c = normalise_pij_c(pij_c_acc, dt_output, volsys)
                pfile.write(
                    f"{t:13.6f} {NColl / float(Np):13.6f} "
                    f"{pij_k[0,0]:13.6f} {pij_k[0,1]:13.6f} {pij_k[0,2]:13.6f} "
                    f"{pij_k[1,1]:13.6f} {pij_k[1,2]:13.6f} {pij_k[2,2]:13.6f} "
                    f"{pij_c[0,0]:13.6f} {pij_c[0,1]:13.6f} {pij_c[0,2]:13.6f} "
                    f"{pij_c[1,1]:13.6f} {pij_c[1,2]:13.6f} {pij_c[2,2]:13.6f}\n"
                )
                pij_c_acc[:] = 0.0
                t_last_output = t

                Ntau += 1

            # ---- USF shear drift (implements -gdot*Vy * df/dVx term) ----
            if flow_mode == 'usf':
                vel[:, 0] -= gdot * vel[:, 1] * dt
                # Remove net bulk momentum introduced by drift
                vel -= vel.mean(axis=0)

            # Compute temperatures
            Ttrans = (2.0 * 0.5 * params.mass
                      * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np))
            Trot = np.sum(Er) / float(Np)
            temp_ratio = Ttrans / Trot if Trot > 0.0 else 1.0

            # NTC collision selection
            totalSel = (float(Np) * float(Np - 1)
                        * params.sigma_c * vrmax * ovol * halfdt)

            RR = np.random.rand()
            totalSel = np.floor(totalSel + RR)

            vrmax_temp = 0.0

            while totalSel > 0:
                totalSel -= 1

                # Select pair
                p1 = np.random.randint(0, Np)
                p2 = p1
                while p2 == p1:
                    p2 = np.random.randint(0, Np)

                v1 = vel[p1, :]
                v2 = vel[p2, :]
                vr = np.linalg.norm(v2 - v1)

                if vr >= vrmax_temp:
                    vrmax_temp = vr

                # Accept/reject
                RR = np.random.random()
                if vr < vrmax * RR:
                    continue

                NColl += 2

                if sphere_mode:
                    # Isotropic scattering + COR dissipation (mirrors Fortran collide.f90)
                    eij = np.random.randn(3)
                    eij /= np.linalg.norm(eij)
                    vrel_vec = v1 - v2
                    CR = np.dot(eij, vrel_vec)
                    COR_PP = (alpha + 1.0) * 0.5
                    vel[p1, :] = v1 - COR_PP * CR * eij
                    vel[p2, :] = v2 + COR_PP * CR * eij
                else:
                    vcom = (v1 + v2) * 0.5
                    v1com = v1 - vcom
                    v2com = v2 - vcom

                    Etrans_i = 0.5 * params.mass * (
                        np.dot(v1com, v1com) + np.dot(v2com, v2com)
                    )
                    Erot_i = Er[p1] + Er[p2]
                    Etotal_i = Etrans_i + Erot_i

                    epsilon_tr_i = Etrans_i / Etotal_i
                    epsilon_rot_1_i = Er[p1] / Erot_i if Erot_i > 0 else 0.5

                    in_equilibration = t < equilibration_time

                    # Sample scattering angle
                    if in_equilibration:
                        chi = sample_chi(p_chi_fn_eq, p_max_eq)
                    else:
                        chi = sample_chi(p_chi_fn_target, p_max_target)
                    chi_rad = np.pi * chi

                    # Rotational relaxation
                    theta = temp_ratio
                    if use_zr_eff:
                        Zr_val = Z_R_eff_val
                    else:
                        Zr_val = Zr(theta, eta=1.0, alpha=alpha)
                    P_r = min(1.0 / Zr_val, 0.5)  # cap so that 2*P_r never exceeds 1

                    relax_p1 = False
                    relax_p2 = False
                    Rn = np.random.random()
                    if Rn < P_r:
                        relax_p1 = True
                    elif Rn < 2.0 * P_r:
                        relax_p2 = True

                    # theta2: GMM conditioning input. When Z_R_eff is used, fix at
                    # theta* so the exchange always samples from the steady-state
                    # correlated distribution. Otherwise track the current theta.
                    theta2 = gmm_theta2 if use_zr_eff else prepare_theta(temp_ratio)

                    if relax_p1:
                        if alpha >= 1.0:
                            # Analytical Borgnakke-Larsen: exact equipartition at equilibrium.
                            # Beta(3/2, 1) has mean 3/5 = 0.6 (3 trans + 2 rot DOF).
                            # Avoids GMM bias at theta=1 that drives theta* below 1.
                            epsilon_tr_f = np.random.beta(2.0, 2.0)
                            epsilon_rot_1_f = np.random.random()
                            epsilon_rot_2_f = 1.0 - epsilon_rot_1_f
                        else:
                            sample = models.cond_gmm.sample_conditionals(
                                r=theta2, e_tr=epsilon_tr_i,
                                e_r1=epsilon_rot_1_i, n_samples=1
                            )
                            epsilon_tr_f = sample[0, 0]
                            epsilon_rot_1_f = sample[0, 1]
                            epsilon_rot_2_f = 1.0 - epsilon_rot_1_f

                    elif relax_p2:
                        if alpha >= 1.0:
                            # Analytical Borgnakke-Larsen (same draw, p1/p2 roles swapped)
                            epsilon_tr_f = np.random.beta(2.0, 2.0)
                            epsilon_rot_2_f = np.random.random()
                            epsilon_rot_1_f = 1.0 - epsilon_rot_2_f
                        else:
                            sample = models.cond_gmm.sample_conditionals(
                                r=theta2, e_tr=epsilon_tr_i,
                                e_r1=epsilon_rot_1_i, n_samples=1
                            )
                            epsilon_tr_f = sample[0, 0]
                            epsilon_rot_2_f = sample[0, 1]
                            epsilon_rot_1_f = 1.0 - epsilon_rot_2_f

                    else:
                        # Elastic: no T-R exchange, preserve individual rotational energies
                        epsilon_tr_f = epsilon_tr_i
                        epsilon_rot_1_f = Er[p1] / Erot_i if Erot_i > 1e-30 else 0.5
                        epsilon_rot_2_f = 1.0 - epsilon_rot_1_f

                    # Dissipation
                    if in_equilibration or gamma_max <= 0.0:
                        gamma = 0.0
                    else:
                        gamma = sample_dissp(beta_a, beta_b)
                        gamma = gamma * gamma_max * prob_one_hit

                    _theta = max(temp_ratio, 1e-10)
                    f_tr = C_alpha * 3.0 * _theta / (3.0 * _theta + 2.0)

                    delta_E = gamma * Etotal_i
                    Etrans_f = epsilon_tr_f * Etotal_i - f_tr * delta_E
                    Erot_f   = (1.0 - epsilon_tr_f) * Etotal_i - (1.0 - f_tr) * delta_E

                    # Physical guard: negative energy is unphysical even if f_tr < 0
                    # Redistribute the violation back
                    if Etrans_f < 0:
                        Erot_f += Etrans_f   # transfer the overshoot to rotation
                        Etrans_f = 1e-30
                    if Erot_f < 0:
                        Etrans_f += Erot_f
                        Erot_f = 1e-30

                    Er[p1] = epsilon_rot_1_f * Erot_f
                    Er[p2] = epsilon_rot_2_f * Erot_f

                    cr_new = np.sqrt(Etrans_f * params.omass)
                    cr_new = max(cr_new, 1e-14)

                    RR = np.random.random()
                    eps = 2 * np.pi * RR
                    vel[p1, :], vel[p2, :] = update_velocities(
                        vel[p1, :], vel[p2, :], chi_rad, eps, cr_new
                    )

                # Collisional pressure accumulation.
                # v1, v2 are pre-collision (saved above); vel[p1,:] is now post-collision.
                accumulate_pij_c(pij_c_acc, v1, v2, vel[p1, :], params.mass, vr)

            if vrmax < vrmax_temp:
                vrmax = vrmax_temp

            t += dt

    print(f"  Simulation complete. NColl={NColl}, output: {output_path}")


def run_all_realizations(config, models):
    """Run DSMC for all seeds specified in the config."""
    seeds = config['simulation']['seeds']
    output_dir = config['simulation']['output_dir']
    AR = config['particle']['AR']
    alpha = config['system']['alpha']

    for i, seed in enumerate(seeds, start=1):
        flow_tag = "_USF" if config.get('flow', {}).get('mode') == 'usf' else ""
        filename = (
            f"AR{AR:.0f}_COR{int(alpha * 100)}{flow_tag}_R{i}.txt"
        )
        output_path = os.path.join(output_dir, filename)
        pressure_path = os.path.join(
            output_dir,
            f"AR{AR:.0f}_COR{int(alpha * 100)}{flow_tag}_R{i}_pressure.txt"
        )
        print(f"Running realization {i}/{len(seeds)} (seed={seed})...")
        run_simulation(config, models, seed, output_path, pressure_path)
