import os
import math
import ctypes
from contextlib import nullcontext

import numpy as np

from .particle import compute_particle_params, result_ar_tag
from .collision import sample_dissp
from .pressure import compute_pij_k, accumulate_pij_c, normalise_pij_c
from .non_gaussian import NonGaussianDiagnostics
from .mu_joint import mu_plane_post_relative_with_eps
from .vss_rank2 import sample_vss_chi
from .rank2_correction import (
    apply_rank0_ftr_probe,
    apply_rank2_ftr_correction,
    compute_rank2_a2,
)
from src.preprocessing.relaxation import prepare_theta, Zr


OUTPUT_BUFFER_SIZE = 65536
_LIBC = None
_LIBC_LOADED = False


class _NTCWorkspace:
    """Reusable buffers for vectorized NTC candidate screening."""

    def __init__(self, capacity, seed):
        self.rng = np.random.default_rng(int(seed) + 0x9E3779B97F4A7C15)
        self.capacity = 0
        self.p1 = None
        self.p2 = None
        self.rand = None
        self.eij = None
        self.v1 = None
        self.v2 = None
        self.vrel = None
        self.prod = None
        self.cr = None
        self.abs_cr = None
        self.norm = None
        self.mask = None
        self.ensure_capacity(capacity)

    def ensure_capacity(self, n):
        n = int(max(1, n))
        if n <= self.capacity:
            return
        capacity = max(n, 2 * self.capacity if self.capacity else 1024)
        self.capacity = capacity
        self.p1 = np.empty(capacity, dtype=np.int64)
        self.p2 = np.empty(capacity, dtype=np.int64)
        self.rand = np.empty(capacity, dtype=np.float64)
        self.eij = np.empty((capacity, 3), dtype=np.float64)
        self.v1 = np.empty((capacity, 3), dtype=np.float64)
        self.v2 = np.empty((capacity, 3), dtype=np.float64)
        self.vrel = np.empty((capacity, 3), dtype=np.float64)
        self.prod = np.empty((capacity, 3), dtype=np.float64)
        self.cr = np.empty(capacity, dtype=np.float64)
        self.abs_cr = np.empty(capacity, dtype=np.float64)
        self.norm = np.empty(capacity, dtype=np.float64)
        self.mask = np.empty(capacity, dtype=bool)

    def fill_particle_indices(self, Np, n):
        self.rng.random(n, out=self.rand[:n])
        np.multiply(self.rand[:n], float(Np), out=self.rand[:n])
        np.floor(self.rand[:n], out=self.rand[:n])
        self.p1[:n] = self.rand[:n]

        self.rng.random(n, out=self.rand[:n])
        np.multiply(self.rand[:n], float(Np), out=self.rand[:n])
        np.floor(self.rand[:n], out=self.rand[:n])
        self.p2[:n] = self.rand[:n]

        np.equal(self.p2[:n], self.p1[:n], out=self.mask[:n])
        same_idx = np.nonzero(self.mask[:n])[0]
        if same_idx.size:
            self.p2[same_idx] = (self.p2[same_idx] + 1) % Np

    def screen_candidates(self, vel, Np, n, vrmax):
        self.ensure_capacity(n)
        self.fill_particle_indices(Np, n)

        self.rng.standard_normal(size=(n, 3), out=self.eij[:n])
        np.multiply(self.eij[:n], self.eij[:n], out=self.prod[:n])
        np.sum(self.prod[:n], axis=1, out=self.norm[:n])
        np.sqrt(self.norm[:n], out=self.norm[:n])
        np.maximum(self.norm[:n], 1.0e-30, out=self.norm[:n])
        np.divide(self.eij[:n], self.norm[:n, None], out=self.eij[:n])

        np.take(vel, self.p1[:n], axis=0, out=self.v1[:n])
        np.take(vel, self.p2[:n], axis=0, out=self.v2[:n])
        np.subtract(self.v1[:n], self.v2[:n], out=self.vrel[:n])
        np.multiply(self.eij[:n], self.vrel[:n], out=self.prod[:n])
        np.sum(self.prod[:n], axis=1, out=self.cr[:n])
        np.abs(self.cr[:n], out=self.abs_cr[:n])

        vrmax_temp = float(np.max(self.abs_cr[:n]))
        self.rng.random(n, out=self.rand[:n])
        np.multiply(self.rand[:n], vrmax, out=self.rand[:n])
        np.greater_equal(self.abs_cr[:n], self.rand[:n], out=self.mask[:n])
        return vrmax_temp, np.nonzero(self.mask[:n])[0]


def _load_libc():
    global _LIBC, _LIBC_LOADED
    if not _LIBC_LOADED:
        _LIBC_LOADED = True
        try:
            _LIBC = ctypes.CDLL(None)
        except Exception:
            _LIBC = None
    return _LIBC


def _malloc_trim():
    libc = _load_libc()
    if libc is None or not hasattr(libc, "malloc_trim"):
        return False
    try:
        libc.malloc_trim(0)
    except Exception:
        return False
    return True


def _read_proc_memory_kb():
    values = {}
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    key, rest = line.split(":", 1)
                    parts = rest.strip().split()
                    if parts:
                        values[key] = int(parts[0])
    except OSError:
        pass
    return values


def _chi_hs(mu, alpha):
    """Hard-sphere scattering angle chi_hs(mu, alpha).

    Replaces the stochastic conditional-chi Beta model for the direction
    update.  Validation (diagnose_eps_model.py, plot_eps_model_tensor.py)
    shows chi_cond introduces wrong-sign relax_gxy artifacts; chi_hs gives
    the correct directional behaviour when combined with the vonMises eps model.
    """
    denom = np.sqrt(max(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cos_chi = (1.0 - (1.0 + alpha) * mu * mu) / denom
    return float(np.arccos(np.clip(cos_chi, -1.0, 1.0)))


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
        models: CollisionModels instance (or None in sphere_mode)
        seed: random seed for this realization
        output_path: path for the temperature output .txt file
        pressure_path: path for the pressure tensor output .txt file
    """
    np.random.seed(seed)

    sphere_mode = config.get('simulation', {}).get('sphere_collision', False)
    use_isotropic_eps = config.get('simulation', {}).get('use_isotropic_eps', False)
    sim_cfg = config.get('simulation', {})
    angular_transport_model = sim_cfg.get('angular_transport_model')
    if angular_transport_model is None:
        angular_transport_model = 'current'
    if angular_transport_model not in ('current', 'stress_weight', 'vss_rank2'):
        raise ValueError(
            "simulation.angular_transport_model must be 'current', "
            f"'stress_weight', or 'vss_rank2', got {angular_transport_model!r}"
        )
    output_buffer_size = int(sim_cfg.get('output_buffer_size', OUTPUT_BUFFER_SIZE))
    if output_buffer_size == 0 or output_buffer_size < -1:
        raise ValueError(
            "simulation.output_buffer_size must be -1, 1, or a positive integer, "
            f"got {output_buffer_size}"
        )
    angular_probability_override = sim_cfg.get(
        'angular_transport_probability_override'
    )
    if angular_probability_override is not None:
        angular_probability_override = float(angular_probability_override)
        if not np.isfinite(angular_probability_override):
            raise ValueError(
                "simulation.angular_transport_probability_override must be finite, "
                f"got {angular_probability_override!r}"
            )
        if angular_probability_override < 0.0 or angular_probability_override > 1.0:
            raise ValueError(
                "simulation.angular_transport_probability_override must be in "
                f"[0, 1], got {angular_probability_override:.6g}"
            )

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
        C_alpha = 1.0
        C2_alpha = 0.0
        p_eta = 1.0
        p_eta_elastic = 1.0
        vss_alpha_eff = None
        vss_alpha_eff_elastic = None
    else:
        if alpha < 1.0:
            gamma_max = models.get_gamma_max(alpha, params.AR)
            prob_one_hit = models.get_one_hit(alpha, params.AR)
        else:
            gamma_max = 0.0
            prob_one_hit = 1.0

        beta_a = config['preprocessing']['dissipation']['beta_a']
        beta_b = config['preprocessing']['dissipation']['beta_b']

        C_alpha = config['system'].get('C_alpha') or models.get_C_alpha(alpha, params.AR)
        C2_alpha = 0.0
        if angular_transport_model == 'stress_weight':
            if angular_probability_override is None:
                raise ValueError(
                    "simulation.angular_transport_model='stress_weight' now "
                    "requires simulation.angular_transport_probability_override."
                )
            p_eta = angular_probability_override
            p_eta_elastic = angular_probability_override
        else:
            p_eta = 1.0
            p_eta_elastic = 1.0

        if angular_transport_model == 'vss_rank2':
            vss_alpha_eff = models.get_vss_alpha_eff(alpha, params.AR)
            vss_alpha_eff_elastic = models.get_vss_alpha_eff(1.0, params.AR)
        else:
            vss_alpha_eff = None
            vss_alpha_eff_elastic = None

    # Flow mode
    flow_mode = config.get('flow', {}).get('mode', 'hcs')
    gdot = float(config.get('flow', {}).get('shear_rate', 0.0))
    if flow_mode not in ('hcs', 'usf'):
        raise ValueError(f"flow.mode must be 'hcs' or 'usf', got {flow_mode!r}")
    if flow_mode == 'usf' and gdot == 0.0:
        print("  Warning: USF mode with gdot=0 is equivalent to HCS.")
    ftr_rank0_probe_delta = sim_cfg.get('ftr_rank0_probe_delta', None)
    ftr_rank0_probe_active = ftr_rank0_probe_delta is not None
    if ftr_rank0_probe_active:
        ftr_rank0_probe_delta = float(ftr_rank0_probe_delta)
    rank2_correction_active = (
        bool(sim_cfg.get('rank2_correction_enabled', False))
        and flow_mode == 'usf'
        and not sphere_mode
        and not ftr_rank0_probe_active
    )
    if rank2_correction_active:
        C2_alpha = models.get_C2(alpha, params.AR)
    hcs_rescale_temperature = bool(
        config.get('simulation', {}).get('hcs_rescale_temperature', False)
    )
    hcs_rescale_reference = config.get(
        'simulation', {}
    ).get('hcs_rescale_reference', 'initial')
    if hcs_rescale_reference != 'initial':
        raise ValueError(
            "simulation.hcs_rescale_reference currently supports only 'initial'"
        )
    if hcs_rescale_temperature and flow_mode != 'hcs':
        raise ValueError("HCS temperature rescaling is only valid for flow.mode='hcs'")
    hcs_rescale_vrmax_policy = sim_cfg.get(
        'hcs_rescale_vrmax_policy', 'reset'
    )
    if hcs_rescale_vrmax_policy not in ('reset', 'scale'):
        raise ValueError(
            "simulation.hcs_rescale_vrmax_policy must be 'reset' or 'scale', "
            f"got {hcs_rescale_vrmax_policy!r}"
        )

    # Time parameters
    dt = config['time']['dt']
    halfdt = dt * 0.5
    dtau = config['time']['dtau']
    t_end = config['time']['t_end']
    tau_end = config['time'].get('tau_end')
    if tau_end is not None:
        tau_end = float(tau_end)
        if tau_end <= 0.0:
            raise ValueError(f"time.tau_end must be > 0 when set, got {tau_end}")
    equilibration_time = config['time'].get('equilibration_time', 0.0)
    if equilibration_time < 0.0:
        raise ValueError(
            f"time.equilibration_time must be >= 0, got {equilibration_time}"
        )
    malloc_trim_interval_steps = int(
        sim_cfg.get('malloc_trim_interval_steps', 0) or 0
    )
    if malloc_trim_interval_steps < 0:
        raise ValueError(
            "simulation.malloc_trim_interval_steps must be >= 0, "
            f"got {malloc_trim_interval_steps}"
        )
    mem_diag_cfg = sim_cfg.get('memory_diagnostics', {}) or {}
    mem_diag_enabled = bool(mem_diag_cfg.get('enabled', False))
    mem_diag_interval_steps = int(mem_diag_cfg.get('interval_steps', 0) or 0)
    if mem_diag_interval_steps < 0:
        raise ValueError(
            "simulation.memory_diagnostics.interval_steps must be >= 0, "
            f"got {mem_diag_interval_steps}"
        )
    mem_diag_on_output = bool(mem_diag_cfg.get('print_on_output', True))

    # Initialize particles
    vel, omega, Er = initialize_particles(Np, kTt, kTr, params.mass, params.mI)
    if sphere_mode:
        Er[:] = 0.0
        omega[:] = 0.0
    if sphere_mode:
        hcs_T_ref = float(kTt)
    else:
        hcs_T_ref = (3.0 * float(kTt) + 2.0 * float(kTr)) / 5.0

    def _rescale_hcs_state():
        if sphere_mode:
            T_now = (params.mass * np.sum(np.sum(vel**2, axis=1))
                     / (3.0 * Np))
            if T_now <= 0.0:
                raise FloatingPointError(
                    f"Cannot HCS-rescale sphere state with Ttrans={T_now}"
                )
            scale2 = hcs_T_ref / T_now
            scale = np.sqrt(scale2)
            vel[:] *= scale
            return scale

        Ttrans_now = (params.mass * np.sum(np.sum(vel**2, axis=1))
                      / (3.0 * Np))
        Trot_now = np.sum(Er) / float(Np)
        Ttotal_now = (3.0 * Ttrans_now + 2.0 * Trot_now) / 5.0
        if Ttotal_now <= 0.0:
            raise FloatingPointError(
                "Cannot HCS-rescale spherocylinder state with "
                f"T_total={Ttotal_now}"
            )
        scale2 = hcs_T_ref / Ttotal_now
        scale = np.sqrt(scale2)
        vel[:] *= scale
        Er[:] *= scale2
        return scale

    vrmax = 5.0 * np.sqrt(2.0) * np.sqrt(kTt * params.omass)

    NColl = 0
    t = 0.0
    Ntau = 0
    step_count = 0
    last_n_cands = 0
    last_n_accepted = 0
    write_pressure = flow_mode == 'usf'
    pij_c_acc = np.zeros((3, 3)) if write_pressure else None
    t_last_output = 0.0
    ntc_workspace = _NTCWorkspace(capacity=1024, seed=seed)

    def _print_memory_diag(reason, n_cands, n_accepted):
        memory = _read_proc_memory_kb()
        rss_kb = memory.get("VmRSS")
        hwm_kb = memory.get("VmHWM")
        if rss_kb is None and hwm_kb is None:
            return
        tau_now = NColl / float(Np)
        rss_str = f"{rss_kb / 1024.0:.1f}MiB" if rss_kb is not None else "n/a"
        hwm_str = f"{hwm_kb / 1024.0:.1f}MiB" if hwm_kb is not None else "n/a"
        print(
            f"  mem[{reason}] step={step_count} t={t:.3f} tau={tau_now:.3f} "
            f"NColl={NColl} n_cands={n_cands} accepted={n_accepted} "
            f"rss={rss_str} maxrss={hwm_str}",
            flush=True,
        )

    flow_str = f"flow={flow_mode}" + (f", gdot={gdot:.4f}" if flow_mode == 'usf' else "")
    if sphere_mode:
        print(f"  Np={Np}, sphere_collision=True, alpha={alpha:.4f}, "
              f"sigma_c={params.sigma_c:.6f}, "
              f"hcs_rescale={hcs_rescale_temperature}, {flow_str}")
    else:
        print(f"  Np={Np}, eta={eta:.4f} (Zr), C_alpha={C_alpha:.4f}, "
              f"mu_chi_model={models.has_mu_chi_model}, "
              f"isotropic_eps={use_isotropic_eps}, "
              f"gamma_max={gamma_max:.6f}, prob_one_hit={prob_one_hit:.6f}, "
              f"angular_transport={angular_transport_model}, "
              f"p_eta={p_eta:.6f}, "
              f"vss_alpha_eff={vss_alpha_eff if vss_alpha_eff is not None else 'n/a'}, "
              f"C2={C2_alpha:.6f}, "
              f"rank2_correction={rank2_correction_active}, "
              f"ftr_rank0_probe_delta={ftr_rank0_probe_delta if ftr_rank0_probe_active else 'n/a'}, "
              f"hcs_rescale={hcs_rescale_temperature}, "
              f"equilibration_time={equilibration_time:.3f}, {flow_str}")

    ng_diag = NonGaussianDiagnostics(
        config, output_path, seed, Np, sphere_mode, flow_mode,
        params.mass, params.mI, t_end
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        pressure_context = (
            open(pressure_path, 'w', buffering=output_buffer_size)
            if write_pressure else nullcontext(None)
        )
        with open(
            output_path, 'w', buffering=output_buffer_size
        ) as file, pressure_context as pfile:
            while (
                t < t_end
                and (
                    tau_end is None
                    or NColl / float(Np) < tau_end
                    or NColl / float(Np) >= Ntau * dtau
                )
            ):
                # Output at intervals
                if NColl / Np >= Ntau * dtau:
                    tau = NColl / float(Np)
                    Eksum = (2.0 * 0.5 * params.mass
                             * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np))
                    Ersum = np.sum(Er) / float(Np)
                    T_total = (3.0 * Eksum + 2.0 * Ersum) / 5.0
                    file.write(
                        f"{t:13.6f} {tau:13.6f} "
                        f"{Eksum:13.6f} {Ersum:13.6f} {T_total:13.6f}\n"
                    )

                    ng_diag.maybe_sample(t, tau, Ntau, vel, Er, Eksum, Ersum)
                    if mem_diag_enabled and mem_diag_on_output:
                        _print_memory_diag(
                            "output", last_n_cands, last_n_accepted
                        )

                    # Pressure tensor output
                    if write_pressure:
                        pij_k = compute_pij_k(vel, params.mass, volsys)
                        dt_output = t - t_last_output
                        pij_c = normalise_pij_c(pij_c_acc, dt_output, volsys)
                        pfile.write(
                            f"{t:13.6f} {tau:13.6f} "
                            f"{pij_k[0,0]:13.6f} {pij_k[0,1]:13.6f} {pij_k[0,2]:13.6f} "
                            f"{pij_k[1,1]:13.6f} {pij_k[1,2]:13.6f} {pij_k[2,2]:13.6f} "
                            f"{pij_c[0,0]:13.6f} {pij_c[0,1]:13.6f} {pij_c[0,2]:13.6f} "
                            f"{pij_c[1,1]:13.6f} {pij_c[1,2]:13.6f} {pij_c[2,2]:13.6f}\n"
                        )
                        pij_c_acc[:] = 0.0
                        t_last_output = t

                    Ntau += 1

                # ---- USF shear drift ----
                if flow_mode == 'usf':
                    vel[:, 0] -= gdot * vel[:, 1] * dt

                # Compute temperatures
                Ttrans = (2.0 * 0.5 * params.mass
                          * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np))
                Trot = np.sum(Er) / float(Np)
                temp_ratio = Ttrans / Trot if Trot > 0.0 else 1.0
                rank2_a2_live = (
                    compute_rank2_a2(vel, params.mass)
                    if rank2_correction_active else 0.0
                )

                # NTC collision selection — batch all candidates at once.
                # Rejection screening is vectorized; only the ~16% accepted pairs
                # go through the full per-collision Python path.
                _ntc_mean = (2.0 * float(Np) * float(Np - 1)
                             * params.sigma_c * vrmax * ovol * halfdt)
                n_cands = int(np.floor(_ntc_mean + np.random.rand()))
                last_n_cands = n_cands
                last_n_accepted = 0
                vrmax_temp = 0.0

                if n_cands > 0:
                    vrmax_temp, accepted_idx = ntc_workspace.screen_candidates(
                        vel, Np, n_cands, vrmax
                    )
                    last_n_accepted = int(accepted_idx.size)

                    for k in accepted_idx:
                        p1 = int(ntc_workspace.p1[k])
                        p2 = int(ntc_workspace.p2[k])
                        eij = ntc_workspace.eij[k].copy()

                        # Re-fetch to capture updates from earlier collisions this step
                        v1 = vel[p1].copy()
                        v2 = vel[p2].copy()
                        vrel_vec = v1 - v2

                        CR = float(np.dot(eij, vrel_vec))
                        if CR < 0:
                            eij = -eij
                            CR = -CR
                        vr = float(np.linalg.norm(vrel_vec))

                        if sphere_mode:
                            NColl += 2
                            COR_PP = (alpha + 1.0) * 0.5
                            vel[p1, :] = v1 - COR_PP * CR * eij
                            vel[p2, :] = v2 + COR_PP * CR * eij
                            if write_pressure:
                                accumulate_pij_c(
                                    pij_c_acc, v1, v2, vel[p1, :], params.mass, vr,
                                    eij_override=eij
                                )

                        else:
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

                            in_equilibration = t < equilibration_time

                            ghat = vrel_vec / max(vr, 1.0e-30)
                            mu_abs = abs(float(np.dot(eij, ghat)))
                            _alpha_scat = 1.0 if in_equilibration else alpha
                            if angular_transport_model == 'vss_rank2':
                                alpha_eff_scat = (
                                    vss_alpha_eff_elastic
                                    if in_equilibration else vss_alpha_eff
                                )
                                chi_rad = sample_vss_chi(alpha_eff_scat)
                            else:
                                # Validation showed the stochastic conditional-chi
                                # Beta model introduces wrong-sign anisotropy artifacts;
                                # chi_hs paired with the vonMises eps model gives the
                                # correct directional distribution (see plot_eps_model_tensor.py).
                                chi_rad = _chi_hs(mu_abs, _alpha_scat)

                            # Rotational relaxation
                            theta = temp_ratio
                            Zr_val = Zr(theta, eta=1.0, alpha=alpha)
                            P_r = min(1.0 / Zr_val, 0.5)

                            relax_p1 = False
                            relax_p2 = False
                            Rn = np.random.random()
                            if Rn < P_r:
                                relax_p1 = True
                            elif Rn < 2.0 * P_r:
                                relax_p2 = True

                            theta2 = prepare_theta(temp_ratio)

                            if relax_p1:
                                if alpha >= 1.0:
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
                            if ftr_rank0_probe_active and not in_equilibration:
                                f_tr = apply_rank0_ftr_probe(
                                    C_alpha, _theta, delta=ftr_rank0_probe_delta
                                )
                            else:
                                C2_step = (
                                    C2_alpha
                                    if rank2_correction_active and not in_equilibration
                                    else 0.0
                                )
                                f_tr = apply_rank2_ftr_correction(
                                    C_alpha, _theta, C2=C2_step, a2=rank2_a2_live
                                )

                            delta_E = gamma * Etotal_i
                            Etrans_f = epsilon_tr_f * Etotal_i - f_tr * delta_E
                            Erot_f = ((1.0 - epsilon_tr_f) * Etotal_i
                                      - (1.0 - f_tr) * delta_E)

                            if Etrans_f < 0:
                                Erot_f += Etrans_f
                                Etrans_f = 1e-30
                            if Erot_f < 0:
                                Etrans_f += Erot_f
                                Erot_f = 1e-30

                            Er[p1] = epsilon_rot_1_f * Erot_f
                            Er[p2] = epsilon_rot_2_f * Erot_f

                            cr_new = np.sqrt(Etrans_f * params.omass)
                            cr_new = max(cr_new, 1e-14)

                            # Scatter g' using azimuthal angle eps around ghat.
                            # eps=0 → in-plane with eij (hard-sphere limit).
                            # use_isotropic_eps → eps ~ Uniform(0,2pi) (orientation-
                            #   averaged result for smooth spherocylinders).
                            # has_eps_model → eps ~ vonMises(0, kappa(mu,alpha,AR)).
                            # The p_eta mixture keeps the scalar energy model
                            # unchanged and thins only the angular direction update.
                            gpost_mag = 2.0 * cr_new
                            if angular_transport_model == 'stress_weight':
                                p_eta_scat = p_eta_elastic if in_equilibration else p_eta
                                if np.random.random() < p_eta_scat:
                                    if use_isotropic_eps:
                                        eps_rad = np.random.uniform(0.0, 2.0 * np.pi)
                                    elif models.has_eps_model:
                                        eps_rad = models.sample_eps(mu_abs, _alpha_scat, params.AR)
                                    else:
                                        eps_rad = 0.0
                                    gpost = mu_plane_post_relative_with_eps(
                                        vrel_vec, eij, chi_rad, gpost_mag, eps_rad
                                    )
                                else:
                                    gpost = gpost_mag * ghat
                            elif angular_transport_model == 'vss_rank2':
                                eps_rad = np.random.uniform(0.0, 2.0 * np.pi)
                                gpost = mu_plane_post_relative_with_eps(
                                    vrel_vec, eij, chi_rad, gpost_mag, eps_rad
                                )
                            else:
                                if use_isotropic_eps:
                                    eps_rad = np.random.uniform(0.0, 2.0 * np.pi)
                                elif models.has_eps_model:
                                    eps_rad = models.sample_eps(mu_abs, _alpha_scat, params.AR)
                                else:
                                    eps_rad = 0.0
                                gpost = mu_plane_post_relative_with_eps(
                                    vrel_vec, eij, chi_rad, gpost_mag, eps_rad
                                )
                            vel[p1, :] = vcom + 0.5 * gpost
                            vel[p2, :] = vcom - 0.5 * gpost
                            if write_pressure:
                                accumulate_pij_c(
                                    pij_c_acc, v1, v2, vel[p1, :], params.mass, vr,
                                    eij_override=eij
                                )

                if vrmax < vrmax_temp:
                    vrmax = vrmax_temp

                if hcs_rescale_temperature:
                    hcs_scale = _rescale_hcs_state()
                    if hcs_rescale_vrmax_policy == 'scale':
                        vrmax *= hcs_scale
                    else:
                        Ttrans_after = (
                            params.mass * np.sum(np.sum(vel**2, axis=1))
                            / (3.0 * Np)
                        )
                        thermal_vrmax = (
                            5.0 * np.sqrt(2.0)
                            * np.sqrt(max(Ttrans_after, 1.0e-30) * params.omass)
                        )
                        vrmax = max(thermal_vrmax, vrmax_temp * hcs_scale)

                t += dt
                step_count += 1
                if (
                    malloc_trim_interval_steps > 0
                    and step_count % malloc_trim_interval_steps == 0
                ):
                    _malloc_trim()
                if (
                    mem_diag_enabled
                    and mem_diag_interval_steps > 0
                    and step_count % mem_diag_interval_steps == 0
                ):
                    _print_memory_diag(
                        "step", last_n_cands, last_n_accepted
                    )
    finally:
        ng_diag.close(NColl, NColl / float(Np), final_t=t)

    print(f"  Simulation complete. NColl={NColl}, output: {output_path}")


def run_all_realizations(config, models):
    """Run DSMC for all seeds specified in the config."""
    seeds = config['simulation']['seeds']
    output_dir = config['simulation']['output_dir']
    AR = config['particle']['AR']
    alpha = config['system']['alpha']
    ar_tag = result_ar_tag(AR)

    for i, seed in enumerate(seeds, start=1):
        flow_tag = "_USF" if config.get('flow', {}).get('mode') == 'usf' else ""
        filename = f"{ar_tag}_COR{int(alpha * 100)}{flow_tag}_R{i}.txt"
        output_path = os.path.join(output_dir, filename)
        pressure_path = os.path.join(
            output_dir,
            f"{ar_tag}_COR{int(alpha * 100)}{flow_tag}_R{i}_pressure.txt"
        )
        print(f"Running realization {i}/{len(seeds)} (seed={seed})...")
        run_simulation(config, models, seed, output_path, pressure_path)
