import numpy as np


def Zr(theta):
    """Rotational collision number as a function of temperature ratio.

    Z_r(theta) = 0.39*theta^2 + 0.09*theta + 1.19
    """
    return 0.39 * theta**2 + 0.09 * theta + 1.19


def prepare_theta(temp_ratio):
    """Discretize temperature ratio to nearest 0.1 for GMM lookup.

    Clamps to [0.1, 1.2]. Raises ValueError if temp_ratio is outside (0, 1.3).
    """
    if temp_ratio <= 0.0 or temp_ratio >= 1.3:
        raise ValueError(
            f"Error: temp_ratio must be in (0, 1.3). "
            f"Received temp_ratio = {temp_ratio}"
        )

    r = round(temp_ratio * 10) / 10.0

    if r < 0.1:
        r = 0.1
    elif r > 1.2:
        r = 1.2

    return r
