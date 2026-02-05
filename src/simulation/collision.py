import os
import glob
import numpy as np
from scipy.stats import beta as beta_dist

from src.preprocessing.scattering_angle import p_chi_AR_alpha
from src.preprocessing.gmm_energy import ConditionalGMM
from src.preprocessing.dissipation import load_table, lookup_gamma_max, lookup_one_hit


class CollisionModels:
    """Container for all pre-computed collision model artifacts."""

    def __init__(self, model_dir, gmm_npz_path=None):
        """Load all model artifacts.

        Parameters:
            model_dir: directory containing scattering_coeffs.npz and
                       lookup table JSONs
            gmm_npz_path: path to the conditional GMM .npz file.
                          If None, auto-detects gmm_cond_*.npz in model_dir.
        """
        self.model_dir = model_dir
        self._load_all(gmm_npz_path)

    def _load_all(self, gmm_npz_path):
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

    def get_gamma_max(self, alpha, AR):
        """Look up gamma_max for (alpha, AR). Raises KeyError if not found."""
        return lookup_gamma_max(self.gamma_max_table, alpha, AR)

    def get_one_hit(self, alpha, AR):
        """Look up one-hit probability for (alpha, AR). Raises KeyError if not found."""
        return lookup_one_hit(self.one_hit_table, alpha, AR)


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


def sample_dissp(a, b):
    """Sample a single dissipation fraction from Beta(a, b)."""
    return beta_dist.rvs(a, b, size=1)[0]


def update_velocities(velA, velB, chi, eps, crmag):
    """Compute post-collision velocities given scattering angles.

    Parameters:
        velA, velB: (3,) velocity vectors of the two particles
        chi: scattering angle (radians)
        eps: azimuthal angle (radians)
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
