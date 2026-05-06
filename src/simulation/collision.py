import os
import glob
import json
import numpy as np

from src.preprocessing.scattering_angle import p_chi_AR_alpha
from src.preprocessing.gmm_energy import ConditionalGMM
from src.preprocessing.dissipation import (
    load_table, lookup_gamma_max, lookup_one_hit, _interpolate_alpha_for_AR
)
from src.preprocessing.ftr_distribution import load_ftr_table, lookup_ftr_params
from src.preprocessing.zr_eff_table import load_zr_eff_table, lookup_zr_eff
from src.preprocessing.mu_chi_model import load_mu_chi_model, sample_chi_given_mu
from src.preprocessing.fit_eps_model import load_eps_model, sample_eps_given_mu
from src.simulation.ctc_angular import CTCAngularModel
from src.simulation.vss_rank2 import (
    load_vss_alpha_eff_table,
    lookup_vss_alpha_eff,
)


class CollisionModels:
    """Container for all pre-computed collision model artifacts."""

    def __init__(self, model_dir, gmm_npz_path=None, ftr_params_path=None,
                 zr_eff_path=None, c_alpha_path=None, stress_transport_path=None,
                 ctc_angular_path=None, vss_alpha_eff_path=None):
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
            stress_transport_path: optional stress-transport weight JSON from
                                   diagnose_stress_transport_weight.py.
            ctc_angular_path: optional CTC angular conditional .npz artifact.
            vss_alpha_eff_path: optional VSS rank-2 alpha_eff JSON table.
        """
        self.model_dir = model_dir
        self._load_all(
            gmm_npz_path, ftr_params_path, zr_eff_path, c_alpha_path,
            stress_transport_path, ctc_angular_path, vss_alpha_eff_path
        )

    def _load_all(self, gmm_npz_path, ftr_params_path, zr_eff_path, c_alpha_path,
                  stress_transport_path, ctc_angular_path, vss_alpha_eff_path):
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

        # Z_R_eff table (optional; load only when explicitly requested)
        if zr_eff_path and os.path.exists(zr_eff_path):
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

        # Rank-2 stress-transport weights (optional)
        if stress_transport_path and os.path.exists(stress_transport_path):
            self.stress_transport_table = load_stress_transport_weights(
                stress_transport_path
            )
            print(f"  Loaded stress-transport weights: {stress_transport_path}")
        else:
            self.stress_transport_table = None
            if stress_transport_path:
                print(
                    f"  Warning: stress-transport weight file not found at "
                    f"{stress_transport_path}; angular transport weighting disabled."
                )

        # CTC-conditioned angular endpoint sampler (optional)
        if ctc_angular_path and os.path.exists(ctc_angular_path):
            self.ctc_angular_model = CTCAngularModel(ctc_angular_path)
            print(f"  Loaded CTC angular model: {ctc_angular_path}")
        else:
            self.ctc_angular_model = None
            if ctc_angular_path:
                print(
                    f"  Warning: CTC angular model not found at "
                    f"{ctc_angular_path}; CTC angular transport disabled."
                )

        # VSS rank-2 alpha_eff table (optional)
        if vss_alpha_eff_path and os.path.exists(vss_alpha_eff_path):
            self.vss_alpha_eff_table = load_vss_alpha_eff_table(
                vss_alpha_eff_path
            )
            print(f"  Loaded VSS alpha_eff table: {vss_alpha_eff_path}")
        else:
            self.vss_alpha_eff_table = None
            if vss_alpha_eff_path:
                print(
                    f"  Warning: VSS alpha_eff table not found at "
                    f"{vss_alpha_eff_path}; vss_rank2 angular mode disabled."
                )

        # Conditional chi Beta model (optional — falls back to marginal rejection sampler)
        mu_chi_path = os.path.join(self.model_dir, "mu_chi_beta_coeffs.npz")
        if os.path.exists(mu_chi_path):
            self.mu_chi_model = load_mu_chi_model(mu_chi_path)
            print(f"  Loaded mu-chi Beta model: {mu_chi_path}")
        else:
            self.mu_chi_model = None

        # Azimuthal eps von Mises model (optional — falls back to eps=0 in-plane)
        eps_path = os.path.join(self.model_dir, "eps_azimuth_coeffs.npz")
        result = load_eps_model(eps_path)
        if result is not None:
            self.eps_model = result
            print(f"  Loaded eps azimuth model: {eps_path}")
        else:
            self.eps_model = None

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

    def sample_eps(self, mu, alpha, AR, rng=np.random):
        """Sample azimuthal angle eps ~ vonMises(0, kappa(mu, alpha, AR)).

        Returns eps in (-pi, pi].  Requires eps_model to be loaded.
        """
        if self.eps_model is None:
            raise RuntimeError(
                "eps azimuth model not loaded. Run run_eps_fit.py first "
                "or check that models/eps_azimuth_coeffs.npz exists."
            )
        c_kappa, M, N, J, beta_exp = self.eps_model
        return sample_eps_given_mu(mu, alpha, AR, c_kappa, M, N, J, beta_exp, rng=rng)

    @property
    def has_eps_model(self):
        return self.eps_model is not None

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

    def get_stress_transport_weight(self, alpha, AR):
        """Look up angular-only w_eta(alpha, AR).

        Returns 1.0 when the optional table has not been loaded.
        """
        if self.stress_transport_table is None:
            return 1.0
        return lookup_stress_transport_weight(self.stress_transport_table, alpha, AR)

    @property
    def has_ctc_angular_model(self):
        return self.ctc_angular_model is not None

    def sample_ctc_angular(self, alpha, AR, mu, rng=np.random):
        """Sample (chi_rad, eps_rad) from the CTC angular conditional model."""
        if self.ctc_angular_model is None:
            raise RuntimeError(
                "CTC angular model not loaded. Run build_ctc_angular_lookup.py "
                "or set simulation.ctc_angular_file."
            )
        return self.ctc_angular_model.sample(alpha, AR, mu, rng=rng)

    def get_vss_alpha_eff(self, alpha, AR):
        """Look up the VSS rank-2 alpha_eff exponent."""
        if self.vss_alpha_eff_table is None:
            raise RuntimeError(
                "VSS alpha_eff table not loaded. Run "
                "build_vss_alpha_eff_table.py or set "
                "simulation.vss_alpha_eff_table_file."
            )
        return lookup_vss_alpha_eff(self.vss_alpha_eff_table, alpha, AR)


def load_stress_transport_weights(filepath):
    """Load angular-only stress-transport weights from diagnostic JSON.

    Returns a dictionary keyed by (alpha, AR).  Only the primary q=1 w_eta is
    used for the DSMC mixture operator; q-weighted columns remain diagnostic.
    """
    with open(filepath, "r") as f:
        payload = json.load(f)

    rows = payload.get("rows", payload if isinstance(payload, list) else None)
    if not isinstance(rows, list):
        raise ValueError(
            f"Stress-transport weight file {filepath} must contain a 'rows' list"
        )

    table = {}
    for row in rows:
        alpha = float(row["alpha"])
        AR = float(row["AR"])
        w_eta = float(row["w_eta"])
        if not np.isfinite(w_eta):
            raise ValueError(
                f"Non-finite w_eta for alpha={alpha:g}, AR={AR:g} in {filepath}"
            )
        table[(alpha, AR)] = w_eta
    if not table:
        raise ValueError(f"No stress-transport weights found in {filepath}")
    return table


def lookup_stress_transport_weight(table, alpha, AR):
    """Interpolate w_eta first in alpha at fixed AR, then across AR.

    Raises ValueError for w_eta > 1 because the current DSMC implementation
    uses w_eta as the probability of applying a full angular transport event.
    AR is not extrapolated beyond the table bounds; the production table only
    covers the spherocylinder range we intend to treat.
    """
    alpha = float(alpha)
    AR = float(AR)

    available_ars = sorted({key[1] for key in table})
    if not available_ars:
        raise KeyError("stress-transport weight table is empty")

    def interp_alpha(ar_value):
        pairs = sorted((a, w) for (a, ar), w in table.items()
                       if np.isclose(ar, ar_value, atol=1e-12))
        if not pairs:
            raise KeyError(f"No stress-transport weights for AR={ar_value:g}")
        alphas = np.array([p[0] for p in pairs], dtype=float)
        weights = np.array([p[1] for p in pairs], dtype=float)

        idx = np.where(np.isclose(alphas, alpha, atol=1e-12))[0]
        if idx.size:
            return float(weights[idx[0]])

        if alpha <= alphas[0]:
            if len(alphas) == 1:
                return float(weights[0])
            slope = (weights[1] - weights[0]) / (alphas[1] - alphas[0])
            return float(weights[0] + slope * (alpha - alphas[0]))

        if alpha >= alphas[-1]:
            if len(alphas) == 1:
                return float(weights[-1])
            slope = (weights[-1] - weights[-2]) / (alphas[-1] - alphas[-2])
            return float(weights[-1] + slope * (alpha - alphas[-1]))

        return float(np.interp(alpha, alphas, weights))

    min_AR = available_ars[0]
    max_AR = available_ars[-1]
    if AR < min_AR - 1e-12 or AR > max_AR + 1e-12:
        raise KeyError(
            f"stress-transport weights are available only for "
            f"AR in [{min_AR:g}, {max_AR:g}], got AR={AR:g}"
        )

    for ar_value in available_ars:
        if np.isclose(ar_value, AR, atol=1e-12):
            w_eta = interp_alpha(ar_value)
            break
    else:
        ar_arr = np.array(available_ars, dtype=float)
        values = np.array([interp_alpha(ar_value) for ar_value in ar_arr])
        w_eta = float(np.interp(AR, ar_arr, values))

    if w_eta < 0.0:
        raise ValueError(
            f"stress-transport w_eta must be >= 0 for alpha={alpha:g}, "
            f"AR={AR:g}; got {w_eta:.6g}"
        )
    if w_eta > 1.0:
        raise ValueError(
            f"stress-transport w_eta={w_eta:.6g} > 1 for alpha={alpha:g}, "
            f"AR={AR:g}; current mixture operator only supports w_eta <= 1"
        )
    return w_eta


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
