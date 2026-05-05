import glob
import os
import re

import numpy as np


DEFAULT_SOURCE_ROOT = "/home/muhammed/Documents/Thesis/Coll_Models/results"
ROW_COLUMNS = ("mu", "chi_rad", "eps_rad")
CASE_RE = re.compile(
    r"alpha_([0-9]+(?:\.[0-9]+)?)_r1\.00_AR([0-9]+(?:\.[0-9]+)?)$"
)


def _rng_integers(rng, low, high):
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def alpha_key(alpha):
    return f"a{int(round(float(alpha) * 1000)):04d}"


def ar_key(AR):
    return f"AR{int(round(float(AR) * 10)):02d}"


def case_key(alpha, AR):
    return f"{alpha_key(alpha)}_{ar_key(AR)}"


def parse_case_dir(case_dir):
    match = CASE_RE.search(os.path.basename(str(case_dir)))
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def equal_dsmc_mu_edges(n_mu_bins):
    """Equal-probability edges under p_DSMC(mu)=2*mu."""
    return np.sqrt(np.linspace(0.0, 1.0, int(n_mu_bins) + 1))


def _rows_from_endpoints(eij, ghat_post):
    """Vectorized inverse of mu_plane_post_relative_with_eps."""
    ghat0 = np.array([-1.0, 0.0, 0.0])
    dot_eg = eij @ ghat0
    flip = dot_eg < 0.0
    eij = eij.copy()
    eij[flip] *= -1.0
    dot_eg = np.abs(dot_eg)

    mu = np.clip(dot_eg, 0.0, 1.0)
    c = np.clip(ghat_post @ ghat0, -1.0, 1.0)
    chi_rad = np.arccos(c)

    n_perp = eij - mu[:, None] * ghat0
    n_perp_norm = np.linalg.norm(n_perp, axis=1)
    sin_chi = np.sin(chi_rad)
    valid = (n_perp_norm > 1.0e-12) & (sin_chi > 1.0e-12)

    eps_rad = np.zeros_like(chi_rad)
    if np.any(valid):
        n_perp_hat = n_perp[valid] / n_perp_norm[valid, None]
        n2 = np.cross(np.broadcast_to(ghat0, n_perp_hat.shape), n_perp_hat)
        gperp = ((ghat_post[valid] - c[valid, None] * ghat0)
                 / sin_chi[valid, None])
        cos_eps = -np.einsum("ij,ij->i", gperp, n_perp_hat)
        sin_eps = np.einsum("ij,ij->i", gperp, n2)
        eps_rad[valid] = np.arctan2(sin_eps, cos_eps)

    return np.column_stack([mu, chi_rad, eps_rad])


def load_case_rows(case_dir, norm_tol=5.0e-3):
    """Load CTC angular rows [mu, chi_rad, eps_rad] for one case."""
    chi_path = os.path.join(case_dir, "chi.txt")
    if not os.path.exists(chi_path):
        raise FileNotFoundError(f"Missing chi.txt: {chi_path}")
    chi = np.loadtxt(chi_path)
    if chi.ndim == 1:
        chi = chi.reshape(1, -1)
    if chi.shape[1] < 10:
        raise ValueError(f"{chi_path} must have at least 10 columns")

    eij = np.array(chi[:, 4:7], dtype=float)
    ghat_post = np.array(chi[:, 7:10], dtype=float)
    finite = np.all(np.isfinite(eij), axis=1) & np.all(np.isfinite(ghat_post), axis=1)
    eij = eij[finite]
    ghat_post = ghat_post[finite]
    if eij.shape[0] == 0:
        raise ValueError(f"No finite CTC angular rows in {case_dir}")

    eij_norm = np.linalg.norm(eij, axis=1)
    gpost_norm = np.linalg.norm(ghat_post, axis=1)
    valid = (eij_norm > 1.0e-12) & (gpost_norm > 1.0e-12)
    eij = eij[valid] / eij_norm[valid, None]
    ghat_post = ghat_post[valid] / gpost_norm[valid, None]
    if eij.shape[0] == 0:
        raise ValueError(f"No nonzero CTC angular rows in {case_dir}")

    max_gpost_norm_error = float(np.max(np.abs(gpost_norm[valid] - 1.0)))
    if max_gpost_norm_error > norm_tol:
        raise ValueError(
            f"{case_dir}: max |norm(ghat_post)-1|={max_gpost_norm_error:.3e} "
            f"exceeds norm_tol={norm_tol:.3e}"
        )

    rows = _rows_from_endpoints(eij, ghat_post)
    mask = np.all(np.isfinite(rows), axis=1)
    mask &= (rows[:, 0] >= 0.0) & (rows[:, 0] <= 1.0)
    mask &= (rows[:, 1] >= 0.0) & (rows[:, 1] <= np.pi)
    return rows[mask], max_gpost_norm_error


def _bin_offsets(mu, edges):
    bin_idx = np.searchsorted(edges, mu, side="right") - 1
    bin_idx = np.clip(bin_idx, 0, edges.size - 2)
    order = np.argsort(bin_idx, kind="mergesort")
    sorted_bins = bin_idx[order]
    counts = np.bincount(sorted_bins, minlength=edges.size - 1)
    offsets = np.zeros(edges.size, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return order, counts, offsets


def build_ctc_angular_lookup(source_root, output_path, n_mu_bins=20,
                             min_AR=1.5, norm_tol=5.0e-3):
    """Build a compressed CTC conditional angular lookup artifact."""
    cases = []
    for case_dir in sorted(glob.glob(os.path.join(source_root, "alpha_*_r1.00_AR*"))):
        parsed = parse_case_dir(case_dir)
        if parsed is None:
            continue
        alpha, AR = parsed
        if AR >= min_AR:
            cases.append((alpha, AR, case_dir))
    if not cases:
        raise FileNotFoundError(f"No CTC angular cases found under {source_root}")

    edges = equal_dsmc_mu_edges(n_mu_bins)
    payload = {
        "alphas": np.array(sorted({alpha for alpha, _, _ in cases}), dtype=float),
        "ARs": np.array(sorted({AR for _, AR, _ in cases}), dtype=float),
        "n_mu_bins": np.array([int(n_mu_bins)], dtype=np.int64),
        "mu_edges": edges.astype(np.float64),
        "columns": np.array(ROW_COLUMNS),
        "min_AR": np.array([float(min_AR)], dtype=float),
    }

    for alpha, AR, case_dir in cases:
        rows, norm_err = load_case_rows(case_dir, norm_tol=norm_tol)
        if rows.size == 0:
            raise ValueError(f"No valid angular rows in {case_dir}")
        order, counts, offsets = _bin_offsets(rows[:, 0], edges)
        key = case_key(alpha, AR)
        payload[f"{key}_rows"] = rows[order].astype(np.float64)
        payload[f"{key}_counts"] = counts.astype(np.int64)
        payload[f"{key}_offsets"] = offsets.astype(np.int64)
        payload[f"{key}_norm_error"] = np.array([norm_err], dtype=float)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **payload)
    os.replace(tmp_path, output_path)
    return output_path


class CTCAngularModel:
    """CTC endpoint angular sampler conditioned on exact (alpha, AR, mu) bins."""

    def __init__(self, lookup_path):
        if not os.path.exists(lookup_path):
            raise FileNotFoundError(f"CTC angular lookup not found: {lookup_path}")
        self.lookup_path = lookup_path
        data = np.load(lookup_path, allow_pickle=False)
        self.alphas = np.array(data["alphas"], dtype=float)
        self.ARs = np.array(data["ARs"], dtype=float)
        self.mu_edges = np.array(data["mu_edges"], dtype=float)
        self.n_mu_bins = int(data["n_mu_bins"][0])
        self.rows = {}
        self.counts = {}
        self.offsets = {}
        for alpha in self.alphas:
            for AR in self.ARs:
                key = case_key(alpha, AR)
                rows_key = f"{key}_rows"
                if rows_key in data:
                    self.rows[key] = np.array(data[rows_key], dtype=float)
                    self.counts[key] = np.array(data[f"{key}_counts"], dtype=np.int64)
                    self.offsets[key] = np.array(data[f"{key}_offsets"], dtype=np.int64)

    def _key_for_case(self, alpha, AR):
        alpha_matches = np.where(np.isclose(self.alphas, float(alpha), atol=5.0e-6))[0]
        ar_matches = np.where(np.isclose(self.ARs, float(AR), atol=5.0e-6))[0]
        if alpha_matches.size == 0 or ar_matches.size == 0:
            available = ", ".join(
                f"({a:.2f}, {ar:.1f})"
                for a in self.alphas for ar in self.ARs
                if case_key(a, ar) in self.rows
            )
            raise ValueError(
                f"CTC angular model requires exact alpha/AR table for "
                f"({float(alpha):.6f}, {float(AR):.6f}); available: {available}"
            )
        key = case_key(self.alphas[alpha_matches[0]], self.ARs[ar_matches[0]])
        if key not in self.rows:
            raise ValueError(f"CTC angular model missing table for {key}")
        return key

    def _bin_for_mu(self, mu):
        mu = float(np.clip(mu, 0.0, 1.0))
        idx = np.searchsorted(self.mu_edges, mu, side="right") - 1
        return int(np.clip(idx, 0, self.n_mu_bins - 1))

    def _nearest_nonempty_bin(self, counts, bin_idx):
        if counts[bin_idx] > 0:
            return bin_idx
        nonempty = np.flatnonzero(counts > 0)
        if nonempty.size == 0:
            raise ValueError("CTC angular table has no nonempty mu bins")
        return int(nonempty[np.argmin(np.abs(nonempty - bin_idx))])

    def sample(self, alpha, AR, mu, rng=np.random):
        """Sample (chi_rad, eps_rad) from exact case and nearest mu bin."""
        key = self._key_for_case(alpha, AR)
        counts = self.counts[key]
        offsets = self.offsets[key]
        rows = self.rows[key]
        bin_idx = self._nearest_nonempty_bin(counts, self._bin_for_mu(mu))
        start = int(offsets[bin_idx])
        stop = int(offsets[bin_idx + 1])
        row = rows[_rng_integers(rng, start, stop)]
        return float(row[1]), float(row[2])

    def lambda2_by_bin(self, alpha, AR):
        """Return CTC <P2(cos chi)> by stored mu bin."""
        key = self._key_for_case(alpha, AR)
        rows = self.rows[key]
        counts = self.counts[key]
        offsets = self.offsets[key]
        out = np.full(self.n_mu_bins, np.nan)
        for i in range(self.n_mu_bins):
            if counts[i] <= 0:
                continue
            vals = rows[offsets[i]:offsets[i + 1], 1]
            out[i] = float(np.mean(0.5 * (3.0 * np.cos(vals) ** 2 - 1.0)))
        return out
