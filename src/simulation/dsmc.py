import os
import math
import numpy as np

from .particle import compute_particle_params
from .collision import (
    CollisionModels, init_p_chi_distribution, sample_chi,
    sample_dissp, update_velocities
)
from src.preprocessing.relaxation import prepare_theta


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

    vel = np.zeros((Np, 3))
    omega = np.zeros((Np, 3))
    Er = np.zeros(Np)

    for i in range(Np):
        valid = False
        while not valid:
            rand_t = np.random.randn(3)
            rand_r = np.random.randn(3)
            vel[i, :] = rand_t * sqkTt
            omega[i, :] = rand_r * sqkTr
            omega[i, 0] = 0.0  # no rotation about symmetry axis
            Er[i] = 0.5 * mI * (omega[i, 1]**2 + omega[i, 2]**2)
            if not (np.isnan(vel[i]).any() or np.isnan(omega[i]).any()):
                valid = True

    # Remove bulk velocity
    vbulk = np.sum(vel, axis=0) / Np
    vel -= vbulk

    return vel, omega, Er


def run_simulation(config, models, seed, output_path):
    """Run a single DSMC realization.

    Parameters:
        config: dict from YAML config
        models: CollisionModels instance
        seed: random seed for this realization
        output_path: path for the output .txt file
    """
    np.random.seed(seed)

    # Particle properties
    params = compute_particle_params(config)

    # System properties
    alpha = config['system']['alpha']
    kTt = config['system']['kTt']
    kTr = config['system']['kTr']
    phi = config['system']['phi']
    lx, ly, lz = config['system']['domain']
    volsys = lx * ly * lz
    ovol = 1.0 / volsys
    Np = math.ceil(phi * volsys / params.volume)

    # Dissipation parameters (fail-fast lookup)
    gamma_max = models.get_gamma_max(alpha, params.AR)
    prob_one_hit = models.get_one_hit(alpha, params.AR)
    beta_a = config['preprocessing']['dissipation']['beta_a']
    beta_b = config['preprocessing']['dissipation']['beta_b']

    # Time parameters
    dt = config['time']['dt']
    halfdt = dt * 0.5
    dtau = config['time']['dtau']
    t_end = config['time']['t_end']

    # Scattering angle distribution
    p_chi_fn, p_max = init_p_chi_distribution(params.AR, alpha, models)

    # Initialize particles
    vel, omega, Er = initialize_particles(
        Np, kTt, kTr, params.mass, params.mI
    )

    vrmax = 10 * np.sqrt(2.0) * np.sqrt(kTt * params.omass)

    NColl = 0
    t = 0.0
    Ntau = 0

    print(f"  Np={Np}, gamma_max={gamma_max:.6f}, "
          f"prob_one_hit={prob_one_hit:.6f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as file:
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
                Ntau += 1

            # Compute temperatures
            Ttrans = (2.0 * 0.5 * params.mass
                      * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np))
            Trot = np.sum(Er) / float(Np)
            temp_ratio = Ttrans / Trot

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

                # Sample scattering angle
                chi = sample_chi(p_chi_fn, p_max)
                chi_rad = np.pi * chi

                # Rotational relaxation
                theta = temp_ratio
                Zr_val = (0.39 * theta**2 + 0.09 * theta + 1.67) * 2
                P_r = 1.0 / Zr_val

                # Inelastic vs elastic branch
                RR = np.random.random()
                if RR < P_r:
                    theta2 = prepare_theta(temp_ratio)
                    sample = models.cond_gmm.sample_conditionals(
                        r=theta2, e_tr=epsilon_tr_i,
                        e_r1=epsilon_rot_1_i, n_samples=1
                    )
                    epsilon_tr_f = sample[0, 0]
                    epsilon_rot_1_f = sample[0, 1]
                    epsilon_rot_2_f = 1.0 - epsilon_rot_1_f
                else:
                    epsilon_tr_f = epsilon_tr_i
                    RR = np.random.random()
                    epsilon_rot_1_f = RR
                    epsilon_rot_2_f = 1.0 - epsilon_rot_1_f

                # Dissipation
                gamma = sample_dissp(beta_a, beta_b)
                gamma = gamma * gamma_max * prob_one_hit

                Etotal_f = Etotal_i * (1.0 - gamma)

                Etrans_f = float(Etotal_f * epsilon_tr_f)
                Erot_f = float(Etotal_f - Etrans_f)

                Er[p1] = epsilon_rot_1_f * Erot_f
                Er[p2] = epsilon_rot_2_f * Erot_f

                cr_new = np.sqrt(Etrans_f * params.omass)
                cr_new = max(cr_new, 1e-14)

                RR = np.random.random()
                eps = 2 * np.pi * RR
                vel[p1, :], vel[p2, :] = update_velocities(
                    vel[p1, :], vel[p2, :], chi_rad, eps, cr_new
                )

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
        filename = (
            f"AR{AR:.0f}_COR{int(alpha * 100)}_R{i}.txt"
        )
        output_path = os.path.join(output_dir, filename)
        print(f"Running realization {i}/{len(seeds)} (seed={seed})...")
        run_simulation(config, models, seed, output_path)
