import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------

def preprocess_data(Z, eps=1e-8):
    """Apply logit transform to energy fractions (columns 1-4) and log to r (column 0).

    Input Z columns: [r, eps_tr, eps_r1, eps_tr', eps_r1']
    """
    Z_proc = np.empty_like(Z, dtype=np.float64)
    Z_proc[:, 0] = np.log(Z[:, 0])
    Z_clipped = np.clip(Z[:, 1:], eps, 1.0 - eps)
    Z_proc[:, 1:] = np.log(Z_clipped / (1.0 - Z_clipped))
    return Z_proc


def inverse_preprocess_data(Z_proc):
    """Invert the logit/log transforms applied by preprocess_data."""
    Z = np.empty_like(Z_proc, dtype=np.float64)
    Z[:, 0] = np.exp(Z_proc[:, 0])
    Z[:, 1:] = 1.0 / (1.0 + np.exp(-Z_proc[:, 1:]))
    return Z


# ---------------------------------------------------------------------------
# GMM training (expensive -- run once during preprocessing)
# ---------------------------------------------------------------------------

def find_best_gmm_bic(Z_scaled, max_components=40, random_state=42,
                       covariance_type='full'):
    """Select optimal number of GMM components via BIC.

    Returns (best_n, bics_list).
    """
    lowest_bic = np.inf
    best_n = None
    bics = []

    for n in range(1, max_components + 1):
        gmm = GaussianMixture(
            n_components=n, covariance_type=covariance_type,
            random_state=random_state
        )
        gmm.fit(Z_scaled)
        bic = gmm.bic(Z_scaled)
        bics.append(bic)
        if bic < lowest_bic:
            best_n = n
            lowest_bic = bic

    print(f"Best n_components: {best_n}, BIC: {lowest_bic:.2f}")
    return best_n, bics


def train_gmm(Z_scaled, n_components=32, covariance_type='full',
              random_state=42):
    """Train a Gaussian Mixture Model on the scaled data."""
    gmm = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type,
        random_state=random_state
    )
    gmm.fit(Z_scaled)
    return gmm


# ---------------------------------------------------------------------------
# Export conditional GMM to .npz (run once after training)
# ---------------------------------------------------------------------------

def export_conditional_gmm_npz(gmm, scaler, out_path, D_x=3, D_y=2,
                                jitter=1e-12):
    """Pre-compute and save all conditional GMM parameters to a .npz file.

    Saves: weights, means, inv_xx, logdet_xx, A, mu_y, L (Cholesky of
    conditional covariance), scaler_mean, scaler_scale.

    This avoids recomputing matrix inversions at simulation runtime.
    """
    w = gmm.weights_.astype(np.float64)
    means = gmm.means_.astype(np.float64)
    covs = gmm.covariances_.astype(np.float64)
    M = w.shape[0]

    inv_xx = np.empty((M, D_x, D_x), dtype=np.float64)
    logdet_xx = np.empty((M,), dtype=np.float64)
    A = np.empty((M, D_y, D_x), dtype=np.float64)
    mu_y = means[:, D_x:D_x + D_y].copy()
    L = np.empty((M, D_y, D_y), dtype=np.float64)

    for m in range(M):
        S = covs[m]
        Sxx = S[:D_x, :D_x]
        Sxy = S[:D_x, D_x:D_x + D_y]
        Syx = S[D_x:D_x + D_y, :D_x]
        Syy = S[D_x:D_x + D_y, D_x:D_x + D_y]

        sign, ld = np.linalg.slogdet(Sxx)
        if sign <= 0:
            Sxx = Sxx + np.eye(D_x) * jitter
            sign, ld = np.linalg.slogdet(Sxx)
        invSxx = np.linalg.inv(Sxx)

        inv_xx[m] = invSxx
        logdet_xx[m] = ld
        A[m] = Syx @ invSxx

        Scond = Syy - A[m] @ Sxy
        Scond = 0.5 * (Scond + Scond.T)
        try:
            L[m] = np.linalg.cholesky(Scond)
        except np.linalg.LinAlgError:
            L[m] = np.linalg.cholesky(
                Scond + np.eye(D_y) * max(jitter, 1e-10)
            )

    scaler_mean = scaler.mean_.astype(np.float64)
    scaler_scale = scaler.scale_.astype(np.float64)

    np.savez(out_path,
             weights=w, means=means,
             inv_xx=inv_xx, logdet_xx=logdet_xx,
             A=A, mu_y=mu_y, L=L,
             scaler_mean=scaler_mean, scaler_scale=scaler_scale)
    print(f"Saved conditional-GMM package to: {out_path}")


# ---------------------------------------------------------------------------
# ConditionalGMM: load from .npz and sample efficiently at runtime
# ---------------------------------------------------------------------------

class ConditionalGMM:
    """Conditional GMM loaded from a pre-computed .npz file.

    All matrix inversions and Cholesky decompositions are pre-computed.
    Sampling only requires matrix-vector products and random draws.
    """

    def __init__(self, npz_path, D_x=3, D_y=2):
        data = np.load(npz_path)
        self.weights = data['weights']          # (M,)
        self.means = data['means']              # (M, D)
        self.inv_xx = data['inv_xx']            # (M, D_x, D_x)
        self.logdet_xx = data['logdet_xx']      # (M,)
        self.A = data['A']                      # (M, D_y, D_x)
        self.mu_y = data['mu_y']                # (M, D_y)
        self.L = data['L']                      # (M, D_y, D_y)
        self.scaler_mean = data['scaler_mean']  # (D,)
        self.scaler_scale = data['scaler_scale']  # (D,)

        self.M = len(self.weights)
        self.D_x = D_x
        self.D_y = D_y
        self._log_weights = np.log(self.weights + 1e-300)
        self._const = -0.5 * D_x * np.log(2 * np.pi)

    def _scale_x(self, r, e_tr, e_r1, eps=1e-8):
        """Transform (r, e_tr, e_r1) to scaled space."""
        x_raw = np.array([r, e_tr, e_r1], dtype=np.float64)
        # log transform for r
        x_proc = np.empty(3, dtype=np.float64)
        x_proc[0] = np.log(x_raw[0])
        # logit transform for energy fractions
        clipped = np.clip(x_raw[1:], eps, 1.0 - eps)
        x_proc[1:] = np.log(clipped / (1.0 - clipped))
        # standardize
        x_scaled = (x_proc - self.scaler_mean[:self.D_x]) / self.scaler_scale[:self.D_x]
        return x_scaled

    def _unscale_y(self, y_scaled):
        """Transform y from scaled space back to original (0, 1) fractions."""
        y_proc = y_scaled * self.scaler_scale[self.D_x:] + self.scaler_mean[self.D_x:]
        # inverse logit
        y = 1.0 / (1.0 + np.exp(-y_proc))
        return y

    def _sample_one(self, x_scaled):
        """Draw one sample from p(y | x) using pre-computed arrays."""
        mu_x = self.means[:, :self.D_x]  # (M, D_x)

        # Compute log-responsibilities for each component
        log_resp = np.empty(self.M)
        for m in range(self.M):
            diff = x_scaled - mu_x[m]
            mahal = diff @ self.inv_xx[m] @ diff
            log_resp[m] = (self._log_weights[m]
                           + self._const
                           - 0.5 * self.logdet_xx[m]
                           - 0.5 * mahal)

        # Normalize via log-sum-exp
        max_lr = np.max(log_resp)
        log_resp -= max_lr
        resp = np.exp(log_resp)
        resp_sum = np.sum(resp)
        if resp_sum > 1e-300:
            resp /= resp_sum
        else:
            resp = np.ones(self.M) / self.M

        # Choose component
        m = np.random.choice(self.M, p=resp)

        # Conditional mean: mu_y_m + A_m @ (x_scaled - mu_x_m)
        mu_cond = self.mu_y[m] + self.A[m] @ (x_scaled - mu_x[m])

        # Sample: y = mu_cond + L_m @ z,  z ~ N(0, I)
        z = np.random.randn(self.D_y)
        y_scaled = mu_cond + self.L[m] @ z

        return y_scaled

    def sample_conditionals(self, r, e_tr, e_r1, n_samples=1):
        """Sample post-collision energy fractions (eps_tr', eps_r1').

        Parameters:
            r: temperature ratio (theta = T_trans / T_rot)
            e_tr: pre-collision translational energy fraction
            e_r1: pre-collision rotational energy fraction (particle 1)
            n_samples: number of samples to draw

        Returns: (n_samples, 2) array of [eps_tr', eps_r1']
        """
        x_scaled = self._scale_x(r, e_tr, e_r1)
        samples = np.empty((n_samples, self.D_y))
        for i in range(n_samples):
            y_scaled = self._sample_one(x_scaled)
            samples[i] = self._unscale_y(y_scaled)
        return samples
