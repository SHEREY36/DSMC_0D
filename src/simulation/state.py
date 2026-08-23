"""Particle state initialization and HCS thermostat helpers."""

import numpy as np


def initialize_particles(Np, kTt, kTr, mass, mI, sphere_mode=False):
    """Initialize velocities and rotational energies for one realization."""
    sqkTt = np.sqrt(kTt / mass)
    sqkTr = np.sqrt(kTr / mI)

    vel = np.random.randn(Np, 3) * sqkTt
    omega = np.random.randn(Np, 3) * sqkTr
    omega[:, 0] = 0.0
    Er = 0.5 * mI * (omega[:, 1]**2 + omega[:, 2]**2)

    vel -= np.sum(vel, axis=0) / Np
    if sphere_mode:
        omega[:] = 0.0
        Er[:] = 0.0
    return vel, omega, Er


def reference_temperature(kTt, kTr, sphere_mode=False):
    """Initial total granular temperature used by HCS rescaling."""
    if sphere_mode:
        return float(kTt)
    return (3.0 * float(kTt) + 2.0 * float(kTr)) / 5.0


def rescale_hcs_state(vel, Er, params, Np, target_temperature, sphere_mode=False):
    """Rescale HCS state to a fixed total temperature for diagnostics."""
    if sphere_mode:
        T_now = params.mass * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np)
        if T_now <= 0.0:
            raise FloatingPointError(f"Cannot HCS-rescale sphere state with T={T_now}")
        scale = np.sqrt(target_temperature / T_now)
        vel[:] *= scale
        return scale

    Ttrans_now = params.mass * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np)
    Trot_now = np.sum(Er) / float(Np)
    Ttotal_now = (3.0 * Ttrans_now + 2.0 * Trot_now) / 5.0
    if Ttotal_now <= 0.0:
        raise FloatingPointError(f"Cannot HCS-rescale state with T_total={Ttotal_now}")
    scale2 = target_temperature / Ttotal_now
    scale = np.sqrt(scale2)
    vel[:] *= scale
    Er[:] *= scale2
    return scale
