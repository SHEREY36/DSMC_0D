"""Spherocylinder HCS collision kernel."""

import numpy as np

from src.preprocessing.relaxation import Zr, prepare_theta
from src.simulation.collision import sample_dissp
from src.simulation.mu_joint import mu_plane_post_relative_with_eps


def chi_hs(mu, alpha):
    """Hard-sphere scattering angle used by the public HCS kernel."""
    denom = np.sqrt(max(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cos_chi = (1.0 - (1.0 + alpha) * mu * mu) / denom
    return float(np.arccos(np.clip(cos_chi, -1.0, 1.0)))


def rank0_ftr(C_alpha, theta):
    """Rank-zero dissipative routing fraction retained in the HCS release path."""
    theta = max(float(theta), 1.0e-10)
    return float(C_alpha) * 3.0 * theta / (3.0 * theta + 2.0)


class SpherocylinderHCSKernel:
    """Data-informed HCS collision kernel for smooth spherocylinders."""

    def __init__(self, params, models, alpha, eta, beta_a, beta_b, C_alpha,
                 gamma_max, prob_one_hit, equilibration_time=0.0,
                 use_isotropic_eps=True):
        self.params = params
        self.models = models
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.beta_a = float(beta_a)
        self.beta_b = float(beta_b)
        self.C_alpha = float(C_alpha)
        self.gamma_max = float(gamma_max)
        self.prob_one_hit = float(prob_one_hit)
        self.equilibration_time = float(equilibration_time)
        self.use_isotropic_eps = bool(use_isotropic_eps)

    def _sample_energy_fractions(self, theta, epsilon_tr_i, epsilon_rot_1_i,
                                 Er, p1, p2, Erot_i, relax_p1, relax_p2):
        if relax_p1:
            if self.alpha >= 1.0:
                return np.random.beta(2.0, 2.0), np.random.random(), None
            sample = self.models.cond_gmm.sample_conditionals(
                r=prepare_theta(theta),
                e_tr=epsilon_tr_i,
                e_r1=epsilon_rot_1_i,
                n_samples=1,
            )
            return sample[0, 0], sample[0, 1], None

        if relax_p2:
            if self.alpha >= 1.0:
                epsilon_rot_2_f = np.random.random()
                return np.random.beta(2.0, 2.0), None, epsilon_rot_2_f
            sample = self.models.cond_gmm.sample_conditionals(
                r=prepare_theta(theta),
                e_tr=epsilon_tr_i,
                e_r1=epsilon_rot_1_i,
                n_samples=1,
            )
            return sample[0, 0], None, sample[0, 1]

        epsilon_rot_1_f = Er[p1] / Erot_i if Erot_i > 1.0e-30 else 0.5
        return epsilon_tr_i, epsilon_rot_1_f, 1.0 - epsilon_rot_1_f

    def collide(self, vel, Er, p1, p2, eij, v1, v2, vrel_vec, cr, vr, t, temp_ratio):
        params = self.params
        vcom = (v1 + v2) * 0.5
        v1com = v1 - vcom
        v2com = v2 - vcom

        Etrans_i = 0.5 * params.mass * (np.dot(v1com, v1com) + np.dot(v2com, v2com))
        Erot_i = Er[p1] + Er[p2]
        Etotal_i = Etrans_i + Erot_i
        if Etotal_i <= 0.0:
            return 0

        epsilon_tr_i = Etrans_i / Etotal_i
        epsilon_rot_1_i = Er[p1] / Erot_i if Erot_i > 0.0 else 0.5
        in_equilibration = t < self.equilibration_time

        theta = max(temp_ratio, 1.0e-10)
        Zr_val = Zr(theta, eta=1.0, alpha=self.alpha)
        P_r = min(1.0 / Zr_val, 0.5)
        Rn = np.random.random()
        relax_p1 = Rn < P_r
        relax_p2 = (not relax_p1) and Rn < 2.0 * P_r

        epsilon_tr_f, epsilon_rot_1_f, epsilon_rot_2_f = self._sample_energy_fractions(
            theta, epsilon_tr_i, epsilon_rot_1_i, Er, p1, p2, Erot_i, relax_p1,
            relax_p2
        )
        if epsilon_rot_1_f is None:
            epsilon_rot_1_f = 1.0 - epsilon_rot_2_f
        if epsilon_rot_2_f is None:
            epsilon_rot_2_f = 1.0 - epsilon_rot_1_f

        if in_equilibration or self.gamma_max <= 0.0:
            gamma = 0.0
        else:
            gamma = sample_dissp(self.beta_a, self.beta_b)
            gamma *= self.gamma_max * self.prob_one_hit

        delta_E = gamma * Etotal_i
        f_tr = rank0_ftr(self.C_alpha, theta)
        Etrans_f = epsilon_tr_f * Etotal_i - f_tr * delta_E
        Erot_f = (1.0 - epsilon_tr_f) * Etotal_i - (1.0 - f_tr) * delta_E

        if Etrans_f < 0.0:
            Erot_f += Etrans_f
            Etrans_f = 1.0e-30
        if Erot_f < 0.0:
            Etrans_f += Erot_f
            Erot_f = 1.0e-30

        Er[p1] = epsilon_rot_1_f * Erot_f
        Er[p2] = epsilon_rot_2_f * Erot_f

        ghat = vrel_vec / max(vr, 1.0e-30)
        mu_abs = abs(float(np.dot(eij, ghat)))
        alpha_scat = 1.0 if in_equilibration else self.alpha
        chi_rad = chi_hs(mu_abs, alpha_scat)
        eps_rad = np.random.uniform(0.0, 2.0 * np.pi) if self.use_isotropic_eps else 0.0
        gpost_mag = 2.0 * max(np.sqrt(Etrans_f * params.omass), 1.0e-14)
        gpost = mu_plane_post_relative_with_eps(
            vrel_vec, eij, chi_rad, gpost_mag, eps_rad
        )
        vel[p1, :] = vcom + 0.5 * gpost
        vel[p2, :] = vcom - 0.5 * gpost
        return 2
