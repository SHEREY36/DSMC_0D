import glob
import os
import re

import numpy as np


ROW_COLUMNS = ("mu", "chi_rad", "q_total", "eps_tr_post", "eps_rot1_post")


def _alpha_key(alpha):
    return f"a{int(round(float(alpha) * 1000)):04d}"


def _parse_alpha_from_case(path):
    match = re.search(r"alpha_([0-9]+(?:\.[0-9]+)?)_r1\.00_AR2\.0$", path)
    if not match:
        return None
    return float(match.group(1))


def _case_rows(case_dir):
    chi_path = os.path.join(case_dir, "chi.txt")
    ef_path = os.path.join(case_dir, "Ef.txt")
    chi = np.loadtxt(chi_path)
    ef = np.loadtxt(ef_path)
    if chi.ndim == 1:
        chi = chi.reshape(1, -1)
    if ef.ndim == 1:
        ef = ef.reshape(1, -1)
    if chi.shape[0] != ef.shape[0]:
        raise ValueError(
            f"Row count mismatch for {case_dir}: chi={chi.shape[0]}, Ef={ef.shape[0]}"
        )

    mu = chi[:, 3]
    chi_rad = chi[:, 1]
    e_pre = ef[:, 0] + ef[:, 1] + ef[:, 2]
    e_post = ef[:, 3] + ef[:, 4] + ef[:, 5]
    e_rot_post = ef[:, 4] + ef[:, 5]

    with np.errstate(divide="ignore", invalid="ignore"):
        q_total = e_post / e_pre
        eps_tr_post = ef[:, 3] / e_post
        eps_rot1_post = ef[:, 4] / e_rot_post
    eps_rot1_post = np.where(e_rot_post > 1.0e-30, eps_rot1_post, 0.5)

    rows = np.column_stack([
        mu,
        chi_rad,
        q_total,
        eps_tr_post,
        eps_rot1_post,
    ])
    mask = np.all(np.isfinite(rows), axis=1)
    mask &= (rows[:, 0] >= 0.0) & (rows[:, 0] <= 1.0)
    mask &= (rows[:, 1] >= 0.0) & (rows[:, 1] <= np.pi)
    mask &= rows[:, 2] > 0.0
    mask &= (rows[:, 3] >= 0.0) & (rows[:, 3] <= 1.0)
    mask &= (rows[:, 4] >= 0.0) & (rows[:, 4] <= 1.0)
    return rows[mask]


def _equal_count_edges(mu, n_bins):
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(mu, quantiles)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges = np.maximum.accumulate(edges)
    return edges


def build_mu_joint_ar2_lookup(source_root, output_path, n_mu_bins=64):
    """Build a compressed lookup from AR=2, r=1.00 CTC output folders."""
    pattern = os.path.join(source_root, "alpha_*_r1.00_AR2.0")
    cases = []
    for case_dir in sorted(glob.glob(pattern)):
        alpha = _parse_alpha_from_case(case_dir)
        if alpha is not None:
            cases.append((alpha, case_dir))
    if not cases:
        raise FileNotFoundError(f"No AR=2 CTC case folders found under {source_root}")

    payload = {
        "alphas": np.array([alpha for alpha, _ in cases], dtype=float),
        "n_mu_bins": np.array([int(n_mu_bins)], dtype=int),
        "columns": np.array(ROW_COLUMNS),
    }
    for alpha, case_dir in cases:
        rows = _case_rows(case_dir)
        if rows.size == 0:
            raise ValueError(f"No valid mu-joint rows found in {case_dir}")
        key = _alpha_key(alpha)
        payload[f"{key}_rows"] = rows.astype(np.float64)
        payload[f"{key}_edges"] = _equal_count_edges(rows[:, 0], n_mu_bins)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, output_path)
    return output_path


class MuJointAR2Model:
    """Empirical AR=2 model for joint mu-conditioned vector collisions."""

    def __init__(self, lookup_path, source_root=None, n_mu_bins=64,
                 auto_build=True):
        self.lookup_path = lookup_path
        if not os.path.exists(lookup_path):
            if not auto_build:
                raise FileNotFoundError(f"Mu-joint lookup not found: {lookup_path}")
            if not source_root:
                raise FileNotFoundError(
                    f"Mu-joint lookup not found and no source_root was provided: {lookup_path}"
                )
            build_mu_joint_ar2_lookup(source_root, lookup_path, n_mu_bins=n_mu_bins)

        data = np.load(lookup_path, allow_pickle=False)
        self.alphas = np.array(data["alphas"], dtype=float)
        self.rows = {}
        self.edges = {}
        for alpha in self.alphas:
            key = _alpha_key(alpha)
            self.rows[key] = np.array(data[f"{key}_rows"], dtype=float)
            self.edges[key] = np.array(data[f"{key}_edges"], dtype=float)

    def _key_for_alpha(self, alpha):
        matches = np.where(np.isclose(self.alphas, float(alpha), atol=5.0e-6))[0]
        if matches.size == 0:
            available = ", ".join(f"{a:.2f}" for a in self.alphas)
            raise ValueError(
                f"mu_joint_ar2 requires an exact alpha table for {alpha:.6f}; "
                f"available: {available}"
            )
        return _alpha_key(self.alphas[matches[0]])

    def sample(self, alpha, mu, rng=np.random):
        """Sample (chi_rad, q_total, eps_tr_post, eps_rot1_post) at |mu|."""
        key = self._key_for_alpha(alpha)
        rows = self.rows[key]
        edges = self.edges[key]
        mu = float(np.clip(mu, 0.0, 1.0))
        bin_idx = np.searchsorted(edges, mu, side="right") - 1
        bin_idx = int(np.clip(bin_idx, 0, edges.size - 2))

        lo = edges[bin_idx]
        hi = edges[bin_idx + 1]
        if bin_idx == edges.size - 2:
            mask = (rows[:, 0] >= lo) & (rows[:, 0] <= hi)
        else:
            mask = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            candidates = np.arange(rows.shape[0])
        row = rows[int(rng.choice(candidates))]
        return {
            "mu": float(row[0]),
            "chi_rad": float(row[1]),
            "q_total": float(row[2]),
            "eps_tr_post": float(row[3]),
            "eps_rot1_post": float(row[4]),
        }


def perpendicular_unit(ghat, rng=np.random):
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(ghat, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    out = ref - np.dot(ref, ghat) * ghat
    norm = np.linalg.norm(out)
    if norm <= 1.0e-14:
        out = rng.normal(size=3)
        out -= np.dot(out, ghat) * ghat
        norm = np.linalg.norm(out)
    return out / norm


def mu_plane_post_relative(vrel_vec, eij, chi_rad, gpost_mag, rng=np.random):
    """Return g' in the accepted-normal plane with sphere-compatible sign."""
    gmag = np.linalg.norm(vrel_vec)
    if gmag <= 1.0e-14:
        return np.zeros(3)
    ghat = vrel_vec / gmag
    mu = float(np.clip(np.dot(eij, ghat), 0.0, 1.0))
    n_perp = eij - mu * ghat
    n_perp_norm = np.linalg.norm(n_perp)
    if n_perp_norm <= 1.0e-12:
        n_perp_hat = perpendicular_unit(ghat, rng=rng)
    else:
        n_perp_hat = n_perp / n_perp_norm
    ghat_post = np.cos(chi_rad) * ghat - np.sin(chi_rad) * n_perp_hat
    ghat_post /= np.linalg.norm(ghat_post)
    return gpost_mag * ghat_post


def hard_sphere_chi_and_gmag(vrel_vec, eij, alpha):
    """Exact hard-sphere scattering angle and |g'| for geometry self-tests."""
    gmag = np.linalg.norm(vrel_vec)
    if gmag <= 1.0e-14:
        return 0.0, 0.0
    ghat = vrel_vec / gmag
    mu = float(np.clip(np.dot(eij, ghat), 0.0, 1.0))
    denom = np.sqrt(max(1.0 - (1.0 - alpha * alpha) * mu * mu, 1.0e-30))
    cos_chi = (1.0 - (1.0 + alpha) * mu * mu) / denom
    chi = np.arccos(np.clip(cos_chi, -1.0, 1.0))
    return chi, gmag * denom
