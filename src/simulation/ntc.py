"""No-time-counter candidate selection for homogeneous DSMC."""

import numpy as np


class NTCWorkspace:
    """Reusable buffers for vectorized NTC candidate screening."""

    def __init__(self, capacity, seed):
        self.rng = np.random.default_rng(int(seed) + 0x9E3779B97F4A7C15)
        self.capacity = 0
        self.p1 = None
        self.p2 = None
        self.rand = None
        self.eij = None
        self.v1 = None
        self.v2 = None
        self.vrel = None
        self.prod = None
        self.cr = None
        self.abs_cr = None
        self.norm = None
        self.mask = None
        self.ensure_capacity(capacity)

    def ensure_capacity(self, n):
        n = int(max(1, n))
        if n <= self.capacity:
            return
        capacity = max(n, 2 * self.capacity if self.capacity else 1024)
        self.capacity = capacity
        self.p1 = np.empty(capacity, dtype=np.int64)
        self.p2 = np.empty(capacity, dtype=np.int64)
        self.rand = np.empty(capacity, dtype=np.float64)
        self.eij = np.empty((capacity, 3), dtype=np.float64)
        self.v1 = np.empty((capacity, 3), dtype=np.float64)
        self.v2 = np.empty((capacity, 3), dtype=np.float64)
        self.vrel = np.empty((capacity, 3), dtype=np.float64)
        self.prod = np.empty((capacity, 3), dtype=np.float64)
        self.cr = np.empty(capacity, dtype=np.float64)
        self.abs_cr = np.empty(capacity, dtype=np.float64)
        self.norm = np.empty(capacity, dtype=np.float64)
        self.mask = np.empty(capacity, dtype=bool)

    def fill_particle_indices(self, Np, n):
        self.rng.random(n, out=self.rand[:n])
        np.multiply(self.rand[:n], float(Np), out=self.rand[:n])
        np.floor(self.rand[:n], out=self.rand[:n])
        self.p1[:n] = self.rand[:n]

        self.rng.random(n, out=self.rand[:n])
        np.multiply(self.rand[:n], float(Np), out=self.rand[:n])
        np.floor(self.rand[:n], out=self.rand[:n])
        self.p2[:n] = self.rand[:n]

        np.equal(self.p2[:n], self.p1[:n], out=self.mask[:n])
        same_idx = np.nonzero(self.mask[:n])[0]
        if same_idx.size:
            self.p2[same_idx] = (self.p2[same_idx] + 1) % Np

    def screen_candidates(self, vel, Np, n, vrmax):
        self.ensure_capacity(n)
        self.fill_particle_indices(Np, n)

        self.rng.standard_normal(size=(n, 3), out=self.eij[:n])
        np.multiply(self.eij[:n], self.eij[:n], out=self.prod[:n])
        np.sum(self.prod[:n], axis=1, out=self.norm[:n])
        np.sqrt(self.norm[:n], out=self.norm[:n])
        np.maximum(self.norm[:n], 1.0e-30, out=self.norm[:n])
        np.divide(self.eij[:n], self.norm[:n, None], out=self.eij[:n])

        np.take(vel, self.p1[:n], axis=0, out=self.v1[:n])
        np.take(vel, self.p2[:n], axis=0, out=self.v2[:n])
        np.subtract(self.v1[:n], self.v2[:n], out=self.vrel[:n])
        np.multiply(self.eij[:n], self.vrel[:n], out=self.prod[:n])
        np.sum(self.prod[:n], axis=1, out=self.cr[:n])
        np.abs(self.cr[:n], out=self.abs_cr[:n])

        vrmax_temp = float(np.max(self.abs_cr[:n]))
        self.rng.random(n, out=self.rand[:n])
        np.multiply(self.rand[:n], vrmax, out=self.rand[:n])
        np.greater_equal(self.abs_cr[:n], self.rand[:n], out=self.mask[:n])
        return vrmax_temp, np.nonzero(self.mask[:n])[0]


def candidate_count(Np, sigma_c, vrmax, volume, dt):
    """Return the stochastic NTC candidate count for one homogeneous cell."""
    mean = 2.0 * float(Np) * float(Np - 1) * sigma_c * vrmax * (0.5 * dt) / volume
    return int(np.floor(mean + np.random.rand()))
