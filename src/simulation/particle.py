import numpy as np
from dataclasses import dataclass


@dataclass
class SpherocylinderParams:
    """Derived physical properties of a spherocylinder particle."""
    AR: float
    radius: float
    mass: float
    diameter: float
    lcycl: float
    volume: float
    rho: float
    mI: float
    omass: float
    omI: float
    sigma_c: float


def compute_particle_params(config):
    """Compute all derived spherocylinder properties from config.

    Returns a SpherocylinderParams dataclass.
    """
    AR = config['particle']['AR']
    radius = config['particle']['radius']
    mass = config['particle']['mass']

    diameter = 2 * radius
    lcycl = (AR - 1) * diameter
    volume = (np.pi * diameter**3 / 6) + (np.pi * lcycl * radius**2)
    rho = mass / volume

    # Moment of inertia of Spherocylinder (perpendicular axes)
    mI = (
        (np.pi / 48) * rho * diameter**2 * lcycl**3
        + (3.0 * np.pi / 64.0) * rho * diameter**4 * lcycl
        + (np.pi / 60) * rho * diameter**5
        + (np.pi / 24.0) * rho * diameter**3 * lcycl**2
    )

    omass = 1.0 / mass
    omI = 1.0 / mI

    # Collision cross-section (optional scale factor from config for calibration)
    sigma_c = (0.32 * AR**2 + 0.694 * AR - 0.0213) * np.pi
    sigma_c_scale = config['particle'].get('sigma_c_scale', 1.0)
    sigma_c *= sigma_c_scale

    return SpherocylinderParams(
        AR=AR, radius=radius, mass=mass, diameter=diameter,
        lcycl=lcycl, volume=volume, rho=rho, mI=mI,
        omass=omass, omI=omI, sigma_c=sigma_c
    )
