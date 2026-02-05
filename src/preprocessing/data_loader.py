import os
import numpy as np


def load_single_r_data(folder_r):
    """Load energy fraction data from a single temperature-ratio folder.

    Reads Ef.txt and computes pre/post collision energy fractions:
      - epsilon_tr     = E_trans / E_total  (pre-collision)
      - epsilon_tr'    = E_trans' / E_total' (post-collision)
      - epsilon_rot_1  = E_rot1 / E_rot     (pre-collision)
      - epsilon_rot_1' = E_rot1' / E_rot'   (post-collision)

    Returns an (N, 5) array: [r, eps_tr, eps_r1, eps_tr', eps_r1']
    """
    ef_file = os.path.join(folder_r, "Ef.txt")
    E_data = np.loadtxt(ef_file)

    folder_name = os.path.basename(folder_r)
    i_r = int(folder_name[1:])
    r_val = 0.1 * i_r

    pre_energy = E_data[:, 0] + E_data[:, 1] + E_data[:, 2]
    post_energy = E_data[:, 3] + E_data[:, 4] + E_data[:, 5]
    E_rot = E_data[:, 1] + E_data[:, 2]
    E_rot_prime = E_data[:, 4] + E_data[:, 5]

    epsilon_tr = E_data[:, 0] / pre_energy
    epsilon_tr_prime = E_data[:, 3] / post_energy
    epsilon_rot_1 = E_data[:, 1] / E_rot
    epsilon_rot_1_prime = E_data[:, 4] / E_rot_prime

    r_column = np.full_like(epsilon_tr, r_val)
    return np.column_stack(
        (r_column, epsilon_tr, epsilon_rot_1, epsilon_tr_prime, epsilon_rot_1_prime)
    )


def load_all_data(base_dir, r_range=range(1, 13)):
    """Load and stack energy fraction data from all temperature-ratio folders."""
    all_data = []
    for i in r_range:
        folder = os.path.join(base_dir, f"r{i:02d}")
        all_data.append(load_single_r_data(folder))
    return np.vstack(all_data)


def load_chi_data(filepath):
    """Load scattering angle data and normalize to [0, 1] by dividing by pi."""
    data = np.loadtxt(filepath)
    return data[:, 1] / np.pi


def compute_one_hit_ratio(nphit_path):
    """Compute the ratio of single-contact-point collisions to total hits."""
    nphit = np.loadtxt(nphit_path, dtype=int)
    total_hits = np.sum(nphit)
    one_hit_rows = np.sum(nphit == 1)
    return one_hit_rows / total_hits if total_hits > 0 else np.nan


def load_max_dissipation(ef_path):
    """Compute maximum fractional energy dissipation from Ef.txt."""
    E_data = np.loadtxt(ef_path)
    pre_coll_energy = E_data[:, 0] + E_data[:, 1] + E_data[:, 2]
    post_coll_energy = E_data[:, 3] + E_data[:, 4] + E_data[:, 5]
    dissp_energy = (pre_coll_energy - post_coll_energy) / pre_coll_energy
    return np.max(dissp_energy)
