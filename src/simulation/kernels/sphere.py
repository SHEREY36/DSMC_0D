"""Elastic/inelastic smooth-sphere collision update."""


class SphereCollisionKernel:
    """Hard-sphere HCS collision kernel."""

    def __init__(self, alpha):
        self.alpha = float(alpha)

    def collide(self, vel, Er, p1, p2, eij, v1, v2, vrel_vec, cr, vr, t, temp_ratio):
        cor_pp = 0.5 * (1.0 + self.alpha)
        vel[p1, :] = v1 - cor_pp * cr * eij
        vel[p2, :] = v2 + cor_pp * cr * eij
        return 2
