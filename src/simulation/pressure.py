import numpy as np


def compute_pij_k(vel, mass, volsys):
    """Kinetic pressure tensor from current peculiar velocities.

    P_ij^k = (mass / V) * sum_p  vel[p,i] * vel[p,j]

    vel must already be peculiar (zero-mean) velocities.
    Returns a (3,3) symmetric array.
    """
    return (mass / volsys) * (vel.T @ vel)


def accumulate_pij_c(pij_c_acc, v1_pre, v2_pre, v1_post, mass, vr):
    """Accumulate collisional pressure contribution for one accepted collision.

    Parameters
    ----------
    pij_c_acc : (3,3) ndarray — running accumulator, updated in-place
    v1_pre    : (3,)  pre-collision velocity of particle 1
    v2_pre    : (3,)  pre-collision velocity of particle 2
    v1_post   : (3,)  post-collision velocity of particle 1
    mass      : float — particle mass
    vr        : float — pre-collision relative speed |v2_pre - v1_pre|

    Convention (follows DSMC-NSP collide.f90):
      delta_v1 = v1_pre - v1_post          (velocity change = impulse / mass)
      eij      = (v2_pre - v1_pre) / vr    (collision-normal unit vector)
      contribution = mass * outer(delta_v1, eij)

    Only particle 1 is counted per pair (no 1/2 factor; consistent with
    DSMC-NSP which omits the factor and counts one particle per collision).
    """
    if vr < 1e-14:
        return
    delta_v1 = v1_pre - v1_post
    eij = (v2_pre - v1_pre) / vr
    pij_c_acc += mass * np.outer(delta_v1, eij)


def normalise_pij_c(pij_c_acc, dt_output, volsys):
    """Normalise accumulated collisional pressure by time interval and volume.

    Returns (3,3) array. If dt_output <= 0 returns zeros.
    """
    if dt_output <= 0.0:
        return np.zeros((3, 3))
    return pij_c_acc / (dt_output * volsys)
