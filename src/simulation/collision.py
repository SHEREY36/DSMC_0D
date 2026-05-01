import os
import glob
import numpy as np

from src.preprocessing.scattering_angle import p_chi_AR_alpha
from src.preprocessing.gmm_energy import ConditionalGMM
from src.preprocessing.dissipation import (
    load_table, lookup_gamma_max, lookup_one_hit, _interpolate_alpha_for_AR
)
from src.preprocessing.ftr_distribution import load_ftr_table, lookup_ftr_params
from src.preprocessing.zr_eff_table import load_zr_eff_table, lookup_zr_eff
from src.preprocessing.mu_chi_model import load_mu_chi_model, sample_chi_given_mu


class CollisionModels:
    """Container for all pre-computed collision model artifacts."""

    def __init__(self, model_dir, gmm_npz_path=None, ftr_params_path=None,
                 zr_eff_path=None, c_alpha_path=None):
        """Load all model artifacts.

        Parameters:
            model_dir: directory containing scattering_coeffs.npz and
                       lookup table JSONs
            gmm_npz_path: path to the conditional GMM .npz file.
                          If None, auto-detects gmm_cond_*.npz in model_dir.
            ftr_params_path: path to ftr_params_*.json for Laplace f_tr sampling.
                             If None, auto-detects ftr_params_*.json in model_dir.
                             If no file is found, f_tr sampling is disabled (f_tr=0).
            zr_eff_path: optional explicit Z_R_eff table path.
            c_alpha_path: optional explicit C_alpha calibration table path.
        """
        self.model_dir = model_dir
        self._load_all(gmm_npz_path, ftr_params_path, zr_eff_path, c_alpha_path)

    def _load_all(self, gmm_npz_path, ftr_params_path, zr_eff_path, c_alpha_path):
        """Load all serialized model artifacts from disk."""
        # Conditional GMM (from pre-computed .npz)
        if gmm_npz_path is None:
            candidates = sorted(
                glob.glob(os.path.join(self.model_dir, "gmm_cond_*.npz"))
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No gmm_cond_*.npz found in {self.model_dir}"
                )
            gmm_npz_path = candidates[0]
            print(f"  Auto-detected GMM: {gmm_npz_path}")

        self.cond_gmm = ConditionalGMM(gmm_npz_path)

        # Scattering angle polynomials
        scat = np.load(
            os.path.join(self.model_dir, "scattering_coeffs.npz")
        )
        self.a_elastic = scat['a_elastic']
        self.a_inelastic = scat['a_inelastic']
        self.scat_M = int(scat['M'])
        self.scat_N = int(scat['N'])
        self.scat_K = int(scat['K'])
        self.scat_beta = float(scat['beta'])

        # Lookup tables
        self.gamma_max_table = load_table(
            os.path.join(self.model_dir, "gamma_max_table.json")
        )
        self.one_hit_table = load_table(
            os.path.join(self.model_dir, "one_hit_table.json")
        )

        # f_tr Laplace parameters (optional — graceful fallback if not found)
        if ftr_params_path is None:
            candidates = sorted(
                glob.glob(os.path.join(self.model_dir, "ftr_params_*.json"))
            )
            ftr_params_path = candidates[0] if candidates else None
            if ftr_params_path:
                print(f"  Auto-detected f_tr table: {ftr_params_path}")

        if ftr_params_path and os.path.exists(ftr_params_path):
            self.ftr_table = load_ftr_table(ftr_params_path)
        else:
            self.ftr_table = None
            if ftr_params_path:
                print(f"  Warning: f_tr table not found at {ftr_params_path}, "
                      f"f_tr sampling disabled (f_tr=0).")

        # Z_R_eff table (optional; graceful fallback if absent)
        if zr_eff_path is None:
            zr_eff_path = os.path.join(self.model_dir, "zr_eff_table_AR20.json")
        if os.path.exists(zr_eff_path):
            self.zr_eff_table = load_zr_eff_table(zr_eff_path)
            print(f"  Loaded Z_R_eff table: {zr_eff_path}")
        else:
            self.zr_eff_table = None

        # C_alpha calibration table (optional)
        if c_alpha_path is None:
            c_alpha_path = os.path.join(self.model_dir, "C_alpha_table_AR20.json")
        if os.path.exists(c_alpha_path):
            self.C_alpha_table = load_table(c_alpha_path)
            print(f"  Loaded C_alpha table: {c_alpha_path}")
        else:
            self.C_alpha_table = None

        # Conditional chi Beta model (optional — falls back to marginal rejection sampler)
        mu_chi_path = os.path.join(self.model_dir, "mu_chi_beta_coeffs.npz")
        if os.path.exists(mu_chi_path):
            self.mu_chi_model = load_mu_chi_model(mu_chi_path)
            print(f"  Loaded mu-chi Beta model: {mu_chi_path}")
        else:
            self.mu_chi_model = None

    def get_gamma_max(self, alpha, AR):
        """Look up gamma_max for (alpha, AR). Raises KeyError if not found."""
        return lookup_gamma_max(self.gamma_max_table, alpha, AR)

    def get_one_hit(self, alpha, AR):
        """Look up one-hit probability for (alpha, AR). Raises KeyError if not found."""
        return lookup_one_hit(self.one_hit_table, alpha, AR)

    def get_ftr_params(self, alpha, AR):
        """Look up Laplace f_tr parameters (loc, scale) for (alpha, AR).

        Returns None if f_tr table has not been loaded.
        """
        if self.ftr_table is None:
            return None
        return lookup_ftr_params(self.ftr_table, alpha, AR)

    def get_zr_eff(self, alpha, AR):
        """Look up (theta_star, Z_R_eff) for (alpha, AR).

        Returns None if Z_R_eff table has not been loaded.
        """
        if self.zr_eff_table is None:
            return None
        return lookup_zr_eff(self.zr_eff_table, alpha, AR)

    def sample_chi_conditional(self, mu, alpha, AR, rng=np.random):
        """Sample chi (radians) from p(chi | mu, alpha, AR).

        Uses the conditional Beta model when available; raises RuntimeError
        if the model has not been loaded (caller should check has_mu_chi_model).
        """
        if self.mu_chi_model is None:
            raise RuntimeError(
                "mu-chi Beta model not loaded. Run run_mu_chi_fit.py first "
                "or check that models/mu_chi_beta_coeffs.npz exists."
            )
        c_a, c_b, M, N, J, beta_exp = self.mu_chi_model
        return sample_chi_given_mu(mu, alpha, AR, c_a, c_b, M, N, J,
                                   beta_exp=beta_exp, rng=rng)

    @property
    def has_mu_chi_model(self):
        return self.mu_chi_model is not None

    def get_C_alpha(self, alpha, AR):
        """Look up calibration constant C(alpha, AR).

        Returns 1.0 if C_alpha table has not been loaded or key is missing.
        """
        if self.C_alpha_table is None:
            return 1.0
        key = f"({alpha:.3f}, {float(AR):.1f})"
        if key in self.C_alpha_table:
            return float(self.C_alpha_table[key])
        try:
            return float(_interpolate_alpha_for_AR(self.C_alpha_table, alpha, AR, "C_alpha"))
        except KeyError:
            return 1.0


def init_p_chi_distribution(AR, alpha, models):
    """Set up the scattering angle PDF and its maximum for rejection sampling.

    Returns (p_chi_fn, p_max) where p_chi_fn(chi) evaluates the PDF.
    """
    chi_vals = np.linspace(0, 1, 1000)
    p_vals = p_chi_AR_alpha(
        chi_vals, AR, alpha,
        models.a_elastic, models.a_inelastic,
        models.scat_M, models.scat_N, models.scat_K, models.scat_beta
    )
    p_max = np.max(p_vals) * 1.05

    def p_chi_fn(chi):
        return p_chi_AR_alpha(
            chi, AR, alpha,
            models.a_elastic, models.a_inelastic,
            models.scat_M, models.scat_N, models.scat_K, models.scat_beta
        )

    return p_chi_fn, p_max


def sample_chi(p_chi_fn, p_max, rng=np.random):
    """Sample a scattering angle from p(chi) via rejection sampling.

    Returns chi in [0, 1] (normalized by pi).
    """
    while True:
        chi_star = rng.uniform(0.0, 1.0)
        u = rng.uniform(0.0, p_max)
        if u <= p_chi_fn(chi_star):
            return chi_star


def sample_dissp(a, b, rng=np.random):
    """Sample a single dissipation fraction from Beta(a, b)."""
    return rng.beta(a, b)


def eps_from_eij(vrel_vec, eij, rng=np.random):
    """Map an accepted lab-frame eij direction to Bird-frame azimuth eps.

    The scattering kernel in update_velocities rotates the pre-collision
    relative velocity by polar angle chi and azimuth eps about ghat using a
    frame implicit in Bird's formula. This helper projects the accepted eij
    into that same frame so the acceptance geometry controls the azimuth.
    """
    vr = np.linalg.norm(vrel_vec)
    if vr <= 1.0e-14:
        return 2.0 * np.pi * rng.random()

    ghat = vrel_vec / vr
    eij_perp = eij - np.dot(eij, ghat) * ghat
    eij_perp_norm = np.linalg.norm(eij_perp)
    if eij_perp_norm <= 1.0e-10:
        return 2.0 * np.pi * rng.random()

    ur, vr_comp, wr_comp = ghat
    vrwr = np.sqrt(vr_comp**2 + wr_comp**2)
    if vrwr <= 1.0e-8:
        return 2.0 * np.pi * rng.random()

    e1_ref = np.array([0.0, wr_comp / vrwr, -vr_comp / vrwr])
    e2_ref = np.cross(ghat, e1_ref)
    eij_perp_hat = eij_perp / eij_perp_norm

    cos_eps = np.dot(eij_perp_hat, e1_ref)
    sin_eps = np.dot(eij_perp_hat, e2_ref)
    return np.arctan2(sin_eps, cos_eps)


def update_velocities(velA, velB, chi, eps, crmag):
    """Compute post-collision velocities given scattering angles.

    Parameters:
        velA, velB: (3,) velocity vectors of the two particles
        chi: scattering angle (radians)
        eps: azimuthal angle (radians) in the Bird frame about the incoming ghat
        crmag: magnitude of post-collision relative velocity

    Returns (velA_new, velB_new) as (1,3) arrays.
    """
    coschi = np.cos(chi)
    sinchi = np.sin(chi)
    coseps = np.cos(eps)
    sineps = np.sin(eps)

    vcom = (velA + velB) * 0.5
    crA = velA - vcom
    crmagA = np.linalg.norm(crA)

    ur, vr, wr = crA
    vrwr = np.sqrt(vr**2 + wr**2)

    if vrwr >= 1.0e-8:
        crel = [
            coschi * ur + sinchi * sineps * vrwr,
            coschi * vr + sinchi * (crmagA * wr * coseps - ur * vr * sineps) / vrwr,
            coschi * wr - sinchi * (crmagA * vr * coseps + ur * wr * sineps) / vrwr,
        ]
    else:
        crel = [
            coschi * vr + sinchi * (crmagA * coseps - ur * sineps),
            coschi * wr - sinchi * (crmagA * coseps + ur * sineps),
            0.0,
        ]

    crelf = np.array(crel).reshape(1, 3)
    crelf_mag = np.linalg.norm(crelf)
    crelf = crelf / crelf_mag

    crmagA = crmag
    crmagB = crmag

    velA_new = vcom + crelf * crmagA
    velB_new = vcom - crelf * crmagB

    return velA_new, velB_new
