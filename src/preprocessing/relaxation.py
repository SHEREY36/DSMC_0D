import numpy as np


def Zr(theta, eta=1.0, alpha=None):
    """Rotational collision number as a function of temperature ratio.

    For elastic collisions (alpha=1.0): returns the fixed value 1.67 (the
    theta-independent limit, since there is no preferred temperature ratio).
    For inelastic collisions: Z_r(theta, eta) = (0.39*theta^2 + 0.09*theta + 1.67) * eta
    """
    if alpha is not None and alpha >= 1.0:
        return 1.67
    return (0.39 * theta**2 + 0.09 * theta + 1.67) * eta


def prepare_theta(temp_ratio):
    """Discretize temperature ratio to nearest 0.1 for GMM lookup.

    Clamps to [0.1, 1.2] silently. No error is raised for out-of-range values.
    """
    r = round(temp_ratio * 10) / 10.0

    if r < 0.1:
        r = 0.1
    elif r > 1.2:
        r = 1.2

    return r

def sample_f_tr(loc, scale):
    """Sample translational dissipation fraction from a fitted Laplace distribution."""
    return np.random.laplace(loc, scale)
