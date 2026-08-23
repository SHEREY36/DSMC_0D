"""Observable calculations and output formatting for HCS DSMC."""

import os

import numpy as np


def temperatures(vel, Er, params, Np):
    """Return translational, rotational, and total granular temperatures."""
    Ttrans = params.mass * np.sum(np.sum(vel**2, axis=1)) / (3.0 * Np)
    Trot = np.sum(Er) / float(Np)
    Ttotal = (3.0 * Ttrans + 2.0 * Trot) / 5.0
    return float(Ttrans), float(Trot), float(Ttotal)


def write_temperature_row(file_obj, t, tau, vel, Er, params, Np):
    """Write the public result row: t tau T_trans T_rot T_total."""
    Ttrans, Trot, Ttotal = temperatures(vel, Er, params, Np)
    file_obj.write(
        f"{t:13.6f} {tau:13.6f} "
        f"{Ttrans:13.6f} {Trot:13.6f} {Ttotal:13.6f}\n"
    )
    return Ttrans, Trot, Ttotal


def ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
