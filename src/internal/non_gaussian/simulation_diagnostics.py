import json
import os

import numpy as np


DEFAULT_NON_GAUSSIAN_CONFIG = {
    "enabled": False,
    "sample_axis": "tau",
    "sample_start_t": None,
    "sample_end_t": None,
    "sample_delta_t": None,
    "sample_start_tau": None,
    "sample_end_tau": None,
    "sample_delta_tau": None,
    "start_tau": 20.0,
    "sample_every_outputs": 1,
    "hist_speed_bins": 256,
    "hist_speed_range": [0.0, 7.0],
    "hist_rot_speed_bins": 256,
    "hist_rot_speed_range": [0.0, 7.0],
    "hist_energy_tr_bins": 256,
    "hist_energy_tr_range": [0.0, 16.0],
    "hist_energy_rot_bins": 256,
    "hist_energy_rot_range": [0.0, 16.0],
    "hist_energy_coupling_bins": 256,
    "hist_energy_coupling_range": [0.0, 64.0],
    "write_time_series": True,
    "write_histograms": True,
    "write_progress": False,
}

OUTPUT_BUFFER_SIZE = 65536


def get_non_gaussian_config(config):
    """Return a complete non-Gaussian diagnostics config."""
    user_cfg = config.get("diagnostics", {}).get("non_gaussian", {})
    diag_cfg = dict(DEFAULT_NON_GAUSSIAN_CONFIG)
    diag_cfg.update(user_cfg or {})
    diag_cfg["sample_every_outputs"] = max(1, int(diag_cfg["sample_every_outputs"]))
    diag_cfg["hist_speed_bins"] = int(diag_cfg["hist_speed_bins"])
    diag_cfg["hist_rot_speed_bins"] = int(diag_cfg["hist_rot_speed_bins"])
    diag_cfg["hist_energy_tr_bins"] = int(diag_cfg["hist_energy_tr_bins"])
    diag_cfg["hist_energy_rot_bins"] = int(diag_cfg["hist_energy_rot_bins"])
    diag_cfg["hist_energy_coupling_bins"] = int(
        diag_cfg["hist_energy_coupling_bins"]
    )
    diag_cfg["sample_axis"] = str(diag_cfg["sample_axis"]).lower()
    if diag_cfg["sample_axis"] not in ("tau", "t"):
        raise ValueError(
            "diagnostics.non_gaussian.sample_axis must be 'tau' or 't'"
        )
    diag_cfg["start_tau"] = float(diag_cfg["start_tau"])
    if diag_cfg["sample_axis"] == "t":
        if diag_cfg["sample_start_t"] is None:
            raise ValueError(
                "diagnostics.non_gaussian.sample_start_t is required when "
                "sample_axis='t'"
            )
        diag_cfg["sample_start_t"] = float(diag_cfg["sample_start_t"])
        if diag_cfg["sample_end_t"] is not None:
            diag_cfg["sample_end_t"] = float(diag_cfg["sample_end_t"])
        if diag_cfg["sample_delta_t"] is not None:
            diag_cfg["sample_delta_t"] = float(diag_cfg["sample_delta_t"])
            if diag_cfg["sample_delta_t"] <= 0.0:
                raise ValueError(
                    "diagnostics.non_gaussian.sample_delta_t must be > 0"
                )
        if (
            diag_cfg["sample_end_t"] is not None
            and diag_cfg["sample_end_t"] < diag_cfg["sample_start_t"]
        ):
            raise ValueError(
                "diagnostics.non_gaussian.sample_end_t must be >= sample_start_t"
            )
        if diag_cfg["sample_start_tau"] is None:
            diag_cfg["sample_start_tau"] = diag_cfg["start_tau"]
        diag_cfg["sample_start_tau"] = float(diag_cfg["sample_start_tau"])
    else:
        if diag_cfg["sample_start_tau"] is None:
            diag_cfg["sample_start_tau"] = diag_cfg["start_tau"]
        diag_cfg["sample_start_tau"] = float(diag_cfg["sample_start_tau"])
        if diag_cfg["sample_end_tau"] is not None:
            diag_cfg["sample_end_tau"] = float(diag_cfg["sample_end_tau"])
        if diag_cfg["sample_delta_tau"] is not None:
            diag_cfg["sample_delta_tau"] = float(diag_cfg["sample_delta_tau"])
            if diag_cfg["sample_delta_tau"] <= 0.0:
                raise ValueError(
                    "diagnostics.non_gaussian.sample_delta_tau must be > 0"
                )
        if (
            diag_cfg["sample_end_tau"] is not None
            and diag_cfg["sample_end_tau"] < diag_cfg["sample_start_tau"]
        ):
            raise ValueError(
                "diagnostics.non_gaussian.sample_end_tau must be >= sample_start_tau"
            )
    return diag_cfg


def cumulants_from_moments(c4, c6, w4=None, c2w2=None):
    """Compute Brey-style reduced cumulants from reduced moments."""
    values = {
        "a2_tr": (4.0 / 15.0) * c4 - 1.0,
        "a3_tr": (8.0 / 105.0) * c6 - 1.0,
    }
    if w4 is None:
        values["a2_rot"] = np.nan
    else:
        values["a2_rot"] = 0.5 * w4 - 1.0
    if c2w2 is None:
        values["a11"] = np.nan
    else:
        values["a11"] = (2.0 / 3.0) * c2w2 - 1.0
    return values


class NonGaussianDiagnostics:
    """Streaming scalar HCS non-Gaussian diagnostics."""

    def __init__(self, config, output_path, seed, Np, sphere_mode, flow_mode,
                 mass, mI, t_end):
        self.cfg = get_non_gaussian_config(config)
        self.enabled = bool(self.cfg["enabled"]) and flow_mode == "hcs"
        self.output_path = output_path
        self.seed = int(seed)
        self.Np = int(Np)
        self.sphere_mode = bool(sphere_mode)
        self.mass = float(mass)
        self.mI = float(mI)
        self.t_end = float(t_end)
        self.output_index = 0
        self.sample_count = 0
        self.particle_sample_count = 0
        self.final_tau = 0.0
        self.final_t = 0.0
        self.final_NColl = 0
        self.moment_file = None
        sim_cfg = config.get("simulation", {})
        self.output_buffer_size = int(
            sim_cfg.get("output_buffer_size", OUTPUT_BUFFER_SIZE)
        )
        if self.output_buffer_size == 0 or self.output_buffer_size < -1:
            raise ValueError(
                "simulation.output_buffer_size must be -1, 1, or a positive "
                f"integer, got {self.output_buffer_size}"
            )
        self.hcs_rescale_enabled = bool(
            sim_cfg.get("hcs_rescale_temperature", False)
        )
        self.sample_axis = self.cfg["sample_axis"]
        self.sample_start_tau = float(self.cfg["sample_start_tau"])
        self.sample_end_tau = self.cfg["sample_end_tau"]
        self.sample_delta_tau = self.cfg["sample_delta_tau"]
        self.sample_start_t = self.cfg["sample_start_t"]
        self.sample_end_t = self.cfg["sample_end_t"]
        self.sample_delta_t = self.cfg["sample_delta_t"]
        if self.sample_axis == "t":
            self.sample_start_value = float(self.sample_start_t)
            self.sample_end_value = self.sample_end_t
            self.sample_delta_value = self.sample_delta_t
        else:
            self.sample_start_value = self.sample_start_tau
            self.sample_end_value = self.sample_end_tau
            self.sample_delta_value = self.sample_delta_tau
        self.next_sample_value = self.sample_start_value
        self.expected_sample_count = self._expected_sample_count()

        self.sum_c2 = 0.0
        self.sum_c4 = 0.0
        self.sum_c6 = 0.0
        self.sum_w2 = 0.0
        self.sum_w4 = 0.0
        self.sum_c2w2 = 0.0
        self.rot_sample_count = 0

        self.speed_edges = None
        self.rot_speed_edges = None
        self.energy_tr_edges = None
        self.energy_rot_edges = None
        self.energy_coupling_edges = None
        self.speed_counts = None
        self.rot_speed_counts = None
        self.energy_tr_counts = None
        self.energy_rot_counts = None
        self.energy_coupling_counts = None

        if self.enabled:
            self._init_outputs()

    def _derived_path(self, suffix):
        root, _ = os.path.splitext(self.output_path)
        return f"{root}{suffix}"

    def _clean(self, value):
        if isinstance(value, dict):
            return {k: self._clean(v) for k, v in value.items()}
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (np.floating, float)):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    def _expected_sample_count(self):
        if self.sample_end_value is None or self.sample_delta_value is None:
            return None
        span = self.sample_end_value - self.sample_start_value
        return int(np.floor(span / self.sample_delta_value + 1.0e-10)) + 1

    def _init_outputs(self):
        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        if self.cfg["write_histograms"]:
            self.speed_edges = np.linspace(
                self.cfg["hist_speed_range"][0], self.cfg["hist_speed_range"][1],
                self.cfg["hist_speed_bins"] + 1
            )
            self.rot_speed_edges = np.linspace(
                self.cfg["hist_rot_speed_range"][0],
                self.cfg["hist_rot_speed_range"][1],
                self.cfg["hist_rot_speed_bins"] + 1
            )
            self.energy_tr_edges = np.linspace(
                self.cfg["hist_energy_tr_range"][0],
                self.cfg["hist_energy_tr_range"][1],
                self.cfg["hist_energy_tr_bins"] + 1
            )
            self.energy_rot_edges = np.linspace(
                self.cfg["hist_energy_rot_range"][0],
                self.cfg["hist_energy_rot_range"][1],
                self.cfg["hist_energy_rot_bins"] + 1
            )
            self.energy_coupling_edges = np.linspace(
                self.cfg["hist_energy_coupling_range"][0],
                self.cfg["hist_energy_coupling_range"][1],
                self.cfg["hist_energy_coupling_bins"] + 1
            )
            self.speed_counts = np.zeros(self.cfg["hist_speed_bins"], dtype=np.float64)
            self.rot_speed_counts = np.zeros(
                self.cfg["hist_rot_speed_bins"], dtype=np.float64
            )
            self.energy_tr_counts = np.zeros(
                self.cfg["hist_energy_tr_bins"], dtype=np.float64
            )
            self.energy_rot_counts = np.zeros(
                self.cfg["hist_energy_rot_bins"], dtype=np.float64
            )
            self.energy_coupling_counts = np.zeros(
                self.cfg["hist_energy_coupling_bins"], dtype=np.float64
            )

        if self.cfg["write_time_series"]:
            self.moment_file = open(self._derived_path("_ng_moments.txt"), "w",
                                    buffering=self.output_buffer_size)
            self.moment_file.write(
                "# t tau Ttrans Trot theta a2_tr a3_tr a2_rot a11 "
                "c2 c4 c6 w2 w4 c2w2 sample_index n_samples_particles\n"
            )

    def close(self, final_NColl, final_tau, final_t=None):
        if not self.enabled:
            return
        self.final_NColl = int(final_NColl)
        self.final_tau = float(final_tau)
        self.final_t = self.t_end if final_t is None else float(final_t)
        if self.moment_file is not None:
            self.moment_file.close()
            self.moment_file = None
        if self.cfg["write_histograms"]:
            self._write_histograms()
        self._write_summary()

    def maybe_sample(self, t, tau, output_index, vel, Er, Ttrans, Trot):
        if not self.enabled:
            return False
        self.output_index = int(output_index)
        tol = 1.0e-10
        sample_value = t if self.sample_axis == "t" else tau
        if sample_value + tol < self.next_sample_value:
            return False
        if (
            self.sample_end_value is not None
            and self.next_sample_value > self.sample_end_value + tol
        ):
            return False
        if Ttrans <= 0.0:
            return False

        c = vel / np.sqrt(2.0 * Ttrans / self.mass)
        c2 = np.sum(c * c, axis=1)
        c2_mean = float(np.mean(c2))
        c4 = float(np.mean(c2 * c2))
        c6 = float(np.mean(c2 * c2 * c2))
        cumulants = cumulants_from_moments(c4, c6)

        w2_mean = np.nan
        w4 = np.nan
        c2w2 = np.nan
        w2 = None
        has_rot = (not self.sphere_mode) and Trot > 0.0
        if has_rot:
            # Er is the authoritative rotational state in this DSMC model.
            w2 = Er / Trot
            w2_mean = float(np.mean(w2))
            w4 = float(np.mean(w2 * w2))
            c2w2 = float(np.mean(c2 * w2))
            cumulants = cumulants_from_moments(c4, c6, w4=w4, c2w2=c2w2)

        self.sample_count += 1
        self.particle_sample_count += self.Np
        self.sum_c2 += c2_mean
        self.sum_c4 += c4
        self.sum_c6 += c6
        if has_rot:
            self.rot_sample_count += 1
            self.sum_w2 += w2_mean
            self.sum_w4 += w4
            self.sum_c2w2 += c2w2

        if self.cfg["write_time_series"] and self.moment_file is not None:
            theta = Ttrans / Trot if Trot > 0.0 else np.nan
            self.moment_file.write(
                f"{t:13.6f} {tau:13.6f} {Ttrans:13.6f} {Trot:13.6f} "
                f"{theta:13.6f} {cumulants['a2_tr']:13.8f} "
                f"{cumulants['a3_tr']:13.8f} {cumulants['a2_rot']:13.8f} "
                f"{cumulants['a11']:13.8f} {c2_mean:13.8f} "
                f"{c4:13.8f} {c6:13.8f} {w2_mean:13.8f} "
                f"{w4:13.8f} {c2w2:13.8f} {self.sample_count:d} "
                f"{self.particle_sample_count:d}\n"
            )

        if self.cfg["write_histograms"]:
            c_mag = np.sqrt(c2)
            self.speed_counts += np.histogram(c_mag, bins=self.speed_edges)[0]
            self.energy_tr_counts += np.histogram(c2, bins=self.energy_tr_edges)[0]
            if has_rot:
                w_mag = np.sqrt(w2)
                self.rot_speed_counts += np.histogram(
                    w_mag, bins=self.rot_speed_edges
                )[0]
                self.energy_rot_counts += np.histogram(
                    w2, bins=self.energy_rot_edges
                )[0]
                energy_coupling = c2 * w2
                self.energy_coupling_counts += np.histogram(
                    energy_coupling, bins=self.energy_coupling_edges
                )[0]

        if self.sample_delta_value is None:
            self.next_sample_value = np.inf
        else:
            while self.next_sample_value <= sample_value + tol:
                self.next_sample_value += self.sample_delta_value
        if self.cfg["write_progress"]:
            self._write_progress(t, tau, Ttrans, Trot)
        return True

    def _write_progress(self, t, tau, Ttrans, Trot):
        T_total = (
            Ttrans if self.sphere_mode else (3.0 * Ttrans + 2.0 * Trot) / 5.0
        )
        progress = {
            "enabled": True,
            "seed": self.seed,
            "sample_axis": self.sample_axis,
            "current_t": t,
            "current_tau": tau,
            "Ttrans": Ttrans,
            "Trot": None if self.sphere_mode else Trot,
            "T_total": T_total,
            "hcs_rescale_temperature": self.hcs_rescale_enabled,
            "expected_output_samples": self.expected_sample_count,
            "n_output_samples": self.sample_count,
            "sampling_complete": (
                self.expected_sample_count is not None
                and self.sample_count >= self.expected_sample_count
            ),
            "n_particle_samples": self.particle_sample_count,
        }
        with open(self._derived_path("_ng_progress.json"), "w") as f:
            json.dump(self._clean(progress), f, indent=2, sort_keys=True)
            f.write("\n")

    def _density(self, counts, edges):
        total = np.sum(counts)
        widths = np.diff(edges)
        if total <= 0.0:
            return np.zeros_like(counts, dtype=np.float64)
        return counts / (total * widths)

    def _write_histogram_file(self, path, edges, density, header):
        centers = 0.5 * (edges[:-1] + edges[1:])
        with open(path, "w") as f:
            f.write(header + "\n")
            for center, value in zip(centers, density):
                f.write(f"{center:14.8f} {value:16.10e}\n")

    def _write_histograms(self):
        self._write_histogram_file(
            self._derived_path("_ng_hist_speed.txt"),
            self.speed_edges,
            self._density(self.speed_counts, self.speed_edges),
            "# c phi_speed"
        )
        self._write_histogram_file(
            self._derived_path("_ng_hist_rot_speed.txt"),
            self.rot_speed_edges,
            self._density(self.rot_speed_counts, self.rot_speed_edges),
            "# w phi_rot_speed"
        )
        self._write_histogram_file(
            self._derived_path("_ng_hist_energy_tr.txt"),
            self.energy_tr_edges,
            self._density(self.energy_tr_counts, self.energy_tr_edges),
            "# epsilon_t phi_energy_tr"
        )
        self._write_histogram_file(
            self._derived_path("_ng_hist_energy_rot.txt"),
            self.energy_rot_edges,
            self._density(self.energy_rot_counts, self.energy_rot_edges),
            "# epsilon_r phi_energy_rot"
        )
        self._write_histogram_file(
            self._derived_path("_ng_hist_energy_coupling.txt"),
            self.energy_coupling_edges,
            self._density(self.energy_coupling_counts, self.energy_coupling_edges),
            "# epsilon_t_epsilon_r phi_energy_coupling"
        )

    def _write_summary(self):
        avg_c2 = self.sum_c2 / self.sample_count if self.sample_count else np.nan
        avg_c4 = self.sum_c4 / self.sample_count if self.sample_count else np.nan
        avg_c6 = self.sum_c6 / self.sample_count if self.sample_count else np.nan
        avg_w2 = self.sum_w2 / self.rot_sample_count if self.rot_sample_count else None
        avg_w4 = self.sum_w4 / self.rot_sample_count if self.rot_sample_count else None
        avg_c2w2 = (
            self.sum_c2w2 / self.rot_sample_count
            if self.rot_sample_count else None
        )
        cumulants = cumulants_from_moments(avg_c4, avg_c6, avg_w4, avg_c2w2)
        nu = (
            self.final_NColl / (2.0 * self.Np * self.final_t)
            if self.Np > 0 and self.final_t > 0.0 else np.nan
        )
        summary = {
            "enabled": True,
            "seed": self.seed,
            "sample_axis": self.sample_axis,
            "start_tau": self.cfg["start_tau"],
            "sample_start_tau": self.sample_start_tau,
            "sample_end_tau": self.sample_end_tau,
            "sample_delta_tau": self.sample_delta_tau,
            "sample_start_t": self.sample_start_t,
            "sample_end_t": self.sample_end_t,
            "sample_delta_t": self.sample_delta_t,
            "expected_output_samples": self.expected_sample_count,
            "n_output_samples": self.sample_count,
            "sampling_complete": (
                self.expected_sample_count is not None
                and self.sample_count >= self.expected_sample_count
            ),
            "n_particle_samples": self.particle_sample_count,
            "final_tau": self.final_tau,
            "final_t": self.final_t,
            "final_NColl": self.final_NColl,
            "collision_frequency": nu,
            "hcs_rescale_temperature": self.hcs_rescale_enabled,
            "moments": {
                "c2": avg_c2,
                "c4": avg_c4,
                "c6": avg_c6,
                "w2": avg_w2,
                "w4": avg_w4,
                "c2w2": avg_c2w2,
            },
            "cumulants": cumulants,
        }
        with open(self._derived_path("_ng_summary.json"), "w") as f:
            json.dump(self._clean(summary), f, indent=2, sort_keys=True)
            f.write("\n")
