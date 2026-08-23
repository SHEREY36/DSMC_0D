"""PoF-style paper figures for the HCS DSMC validation study."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


AR_VALUES = (1.5, 2.0, 2.5, 3.0)
AR_VALUES_WITH_LIMIT = (1.5, 2.0, 2.5, 3.0, 4.0)
ALPHA_SWEEP = (0.95, 0.90, 0.70, 0.50)
HIST_ALPHAS = (0.95, 0.80, 0.60)

DEFAULT_ROOT = Path("runs/paper_hcs_sweep")
DEFAULT_DEM_ROOT = Path(
    "/home/muhammed/Documents/Thesis/New Simulations/HCS_DEM/"
    "kT_1_kTr_1/Alpha_095"
)
DEFAULT_AR4_DSMC = Path(
    "/home/muhammed/Documents/Thesis/New Simulations/Py_scripts2/Results/T40_95.txt"
)

AR_COLORS = {
    1.5: "#1f77b4",
    2.0: "#2ca02c",
    2.5: "#9467bd",
    3.0: "#d62728",
    4.0: "#bf0040",
}
ALPHA_COLORS = {
    0.95: "#1f77b4",
    0.90: "#2ca02c",
    0.80: "#9467bd",
    0.70: "#d62728",
    0.60: "#ff7f0e",
    0.50: "#8c564b",
}
MARKERS = {
    1.5: "o",
    2.0: "s",
    2.5: "^",
    3.0: "D",
    4.0: "v",
    0.95: "o",
    0.90: "s",
    0.80: "^",
    0.70: "D",
    0.60: "v",
    0.50: "P",
}


@dataclass
class TemperatureSeries:
    t: np.ndarray
    tau: np.ndarray
    Ttrans: np.ndarray
    Trot: np.ndarray
    Ttotal: np.ndarray
    path: Path


@dataclass
class HistogramSeries:
    x: np.ndarray
    density_mean: np.ndarray
    density_stderr: np.ndarray
    paths: list[Path]


def ar_dir(ar: float) -> str:
    return f"AR{int(round(float(ar) * 100)):03d}"


def alpha_dir(alpha: float) -> str:
    return f"alpha_{int(round(float(alpha) * 100)):03d}"


def dem_ar_dir(ar: float) -> str:
    return f"AR{int(round(float(ar) * 10)):02d}"


def load_temperature_file(path: str | Path) -> TemperatureSeries:
    path = Path(path)
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 5:
        raise ValueError(f"Expected at least 5 columns in {path}, got {data.shape[1]}")
    return TemperatureSeries(
        t=data[:, 0],
        tau=data[:, 1],
        Ttrans=data[:, 2],
        Trot=data[:, 3],
        Ttotal=data[:, 4],
        path=path,
    )


def macro_result_dir(macro_root: str | Path, ar: float, alpha: float) -> Path:
    return Path(macro_root) / ar_dir(ar) / alpha_dir(alpha) / "results"


def hist_result_dir(hist_root: str | Path, ar: float, alpha: float) -> Path:
    return Path(hist_root) / ar_dir(ar) / alpha_dir(alpha) / "results"


def list_base_result_files(root: str | Path, ar: float, alpha: float) -> list[Path]:
    result_dir = macro_result_dir(root, ar, alpha)
    paths = sorted(result_dir.glob("*.txt"))
    return [
        p for p in paths
        if "_ng_" not in p.name and "_pressure" not in p.name
    ]


def load_macro_case(root: str | Path, ar: float, alpha: float) -> list[TemperatureSeries]:
    paths = list_base_result_files(root, ar, alpha)
    if not paths:
        raise FileNotFoundError(f"No DSMC macro files for AR={ar:g}, alpha={alpha:.2f}")
    return [load_temperature_file(p) for p in paths]


def load_dem_case(dem_root: str | Path, ar: float) -> TemperatureSeries:
    path = Path(dem_root) / dem_ar_dir(ar) / "T.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing DEM reference for AR={ar:g}: {path}")
    return load_temperature_file(path)


def load_ar4_dsmc(path: str | Path) -> TemperatureSeries:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing AR=4 DSMC reference: {path}")
    return load_temperature_file(path)


def _interp_trace(x_src, y_src, x_grid):
    mask = np.isfinite(x_src) & np.isfinite(y_src)
    if mask.sum() < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    x = x_src[mask]
    y = y_src[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    unique = np.r_[True, np.diff(x) > 0]
    x = x[unique]
    y = y[unique]
    if x.size < 2:
        return np.full_like(x_grid, np.nan, dtype=float)
    return np.interp(x_grid, x, y, left=np.nan, right=np.nan)


def ensemble_mean_on_grid(series_list, x_name, y_getter, x_grid):
    traces = []
    for ts in series_list:
        x_src = getattr(ts, x_name)
        y_src = y_getter(ts)
        traces.append(_interp_trace(x_src, y_src, x_grid))
    arr = np.vstack(traces)
    mean = np.nanmean(arr, axis=0)
    n = np.sum(np.isfinite(arr), axis=0)
    stderr = np.full_like(mean, np.nan, dtype=float)
    for j in np.where(n > 1)[0]:
        vals = arr[:, j]
        vals = vals[np.isfinite(vals)]
        stderr[j] = np.std(vals, ddof=1) / np.sqrt(vals.size)
    return mean, stderr, arr


def _temperature_log(ts: TemperatureSeries):
    T = np.asarray(ts.Ttotal, dtype=float)
    mask = np.isfinite(T) & (T > 0.0)
    if not np.any(mask):
        return np.full_like(T, np.nan)
    T0 = T[mask][0]
    return np.where(mask, np.log(T / T0), np.nan)


def _haff_linear(ts: TemperatureSeries):
    T = np.asarray(ts.Ttotal, dtype=float)
    mask = np.isfinite(T) & (T > 0.0)
    if not np.any(mask):
        return np.full_like(T, np.nan)
    T0 = T[mask][0]
    return np.where(mask, np.sqrt(T0 / T), np.nan)


def _theta(ts: TemperatureSeries):
    return np.where(ts.Trot > 0.0, ts.Ttrans / ts.Trot, np.nan)


def _apply_style(ax):
    ax.tick_params(axis="both", direction="in", which="both", right=True, top=True)
    ax.grid(False)


def set_aspect_from_limits(ax, aspect=1.0):
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    xrange = xhi - xlo
    yrange = yhi - ylo
    if xrange > 0.0 and yrange > 0.0:
        ax.set_aspect(aspect * (xrange / yrange), adjustable="box")


def _framed_legend(ax, *args, **kwargs):
    legend = ax.legend(*args, frameon=True, **kwargs)
    frame = legend.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    frame.set_alpha(1.0)
    return legend


def _save_all(fig, out_dir: str | Path, stem: str, formats=("pdf", "png")):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def _marker_step(n, target=18):
    return max(1, int(math.ceil(max(n, 1) / target)))


def maxwell_speed_pdf(c):
    c = np.asarray(c, dtype=float)
    return 4.0 / np.sqrt(np.pi) * c * c * np.exp(-c * c)


def maxwell_rot_energy_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-x)


def load_histogram_file(path: str | Path):
    data = np.loadtxt(path)
    data = np.atleast_2d(data)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")
    return data[:, 0], data[:, 1]


def aggregate_histogram(result_dir: str | Path, suffix: str) -> HistogramSeries:
    result_dir = Path(result_dir)
    paths = sorted(result_dir.glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No histogram files matching *{suffix} in {result_dir}")
    xs = []
    ys = []
    for path in paths:
        x, y = load_histogram_file(path)
        xs.append(x)
        ys.append(y)
    x0 = xs[0]
    arr = []
    for x, y in zip(xs, ys):
        if x.shape != x0.shape or not np.allclose(x, x0):
            y = np.interp(x0, x, y, left=np.nan, right=np.nan)
        arr.append(y)
    arr = np.vstack(arr)
    mean = np.nanmean(arr, axis=0)
    stderr = (
        np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
        if arr.shape[0] > 1 else np.full_like(mean, np.nan)
    )
    return HistogramSeries(x0, mean, stderr, paths)


def load_hist_cases(hist_root: str | Path):
    cases = {}
    for alpha in HIST_ALPHAS:
        rdir = hist_result_dir(hist_root, 2.0, alpha)
        cases[alpha] = {
            "speed": aggregate_histogram(rdir, "_ng_hist_speed.txt"),
            "rot_energy": aggregate_histogram(rdir, "_ng_hist_energy_rot.txt"),
        }
    return cases


def _hist_integral(hist: HistogramSeries):
    if hist.x.size < 2:
        return np.nan
    dx = float(np.nanmedian(np.diff(hist.x)))
    return float(np.nansum(hist.density_mean * dx))


def load_theta_targets(models_dir: str | Path, alpha_values=(0.90, 0.95)):
    models_dir = Path(models_dir)
    targets = {alpha: [] for alpha in alpha_values}
    for ar in AR_VALUES:
        path = models_dir / "targets" / f"theta_target_table_{ar_dir(ar)[:4]}.json"
        if not path.exists():
            path = models_dir / "targets" / f"theta_target_table_AR{int(round(ar * 10)):02d}.json"
        with open(path) as f:
            raw = json.load(f)
        parsed = {}
        for key, value in raw.items():
            alpha = round(float(key.strip("()").split(",")[0]), 4)
            parsed[alpha] = float(value)
        for alpha in alpha_values:
            if alpha in parsed:
                targets[alpha].append((ar, parsed[alpha]))
    return targets


def figure_collision_time(
    macro_root,
    dem_root,
    ar4_dsmc_file,
    out_dir,
    formats=("pdf", "png"),
):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax = axes[0]
    residual_ar = []
    residual = []

    for ar in AR_VALUES_WITH_LIMIT:
        color = AR_COLORS[ar]
        dem = load_dem_case(dem_root, ar)
        ax.plot(dem.t, dem.tau, color=color, lw=1.25, ls="-", label=fr"${ar:g}$")

        if ar == 4.0:
            dsmc_series = [load_ar4_dsmc(ar4_dsmc_file)]
        else:
            dsmc_series = load_macro_case(macro_root, ar, 0.95)
        t_end = min(float(dem.t[-1]), min(float(s.t[-1]) for s in dsmc_series))
        t_grid = np.linspace(0.0, t_end, 260)
        tau_mean, _, _ = ensemble_mean_on_grid(
            dsmc_series, "t", lambda ts: ts.tau, t_grid
        )
        step = _marker_step(t_grid.size, target=16)
        ax.plot(
            t_grid,
            tau_mean,
            color=color,
            lw=1.15,
            ls="--",
            marker=MARKERS[ar],
            markevery=step,
            ms=4.0,
            mfc="white",
            mec=color,
            alpha=0.95,
        )

        tau_dsmc_tf = float(np.nanmean([
            _interp_trace(s.t, s.tau, np.array([t_end]))[0]
            for s in dsmc_series
        ]))
        tau_dem_tf = float(_interp_trace(dem.t, dem.tau, np.array([t_end]))[0])
        residual_ar.append(ar)
        residual.append((tau_dsmc_tf - tau_dem_tf) / tau_dem_tf)

    ax.set_xlim(-10.0, 1050.0)
    ax.set_ylim(-2.0, 120.0)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_yticks([0, 40, 80, 120])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\tau=N_{\mathrm{coll}}/N_p$")
    _apply_style(ax)
    set_aspect_from_limits(ax, aspect=1.0)

    legend_ar = _framed_legend(ax, title=r"$\mathrm{AR}$", fontsize=8, loc="upper left")
    ax.add_artist(legend_ar)
    method_handles = [
        Line2D([0], [0], color="black", ls="-", lw=1.2, label="DEM"),
        Line2D([0], [0], color="black", ls="--", marker="o", mfc="white",
               mec="black", lw=1.2, ms=4, label="DSMC"),
    ]
    _framed_legend(ax, handles=method_handles, fontsize=8, loc="lower right")
    ax.text(0.03, 0.94, "(a)", transform=ax.transAxes, ha="left", va="top")

    ax = axes[1]
    residual_ar = np.array(residual_ar)
    residual = np.array(residual)
    colors = [AR_COLORS[float(ar)] for ar in residual_ar]
    ax.axhline(0.0, color="0.25", lw=0.9, ls="--")
    ax.plot(residual_ar, residual, color="0.25", lw=0.9, alpha=0.65)
    ax.scatter(
        residual_ar,
        residual,
        s=42,
        facecolors="white",
        edgecolors=colors,
        linewidths=1.4,
        zorder=3,
    )
    ax.scatter([4.0], [residual[residual_ar == 4.0][0]], s=58,
               facecolors=AR_COLORS[4.0], edgecolors="black", linewidths=0.6,
               zorder=4)
    pad = max(0.02, 0.15 * float(np.nanmax(np.abs(residual))))
    ax.set_xlim(1.35, 4.15)
    ax.set_ylim(float(np.nanmin(residual)) - pad, float(np.nanmax(residual)) + pad)
    ax.set_xticks(list(AR_VALUES_WITH_LIMIT))
    ax.set_xlabel(r"$\mathrm{AR}$")
    ax.set_ylabel(r"$[\tau_{\mathrm{DSMC}}-\tau_{\mathrm{DEM}}]/\tau_{\mathrm{DEM}}$")
    _apply_style(ax)
    set_aspect_from_limits(ax, aspect=1.0)
    ax.text(0.03, 0.94, "(b)", transform=ax.transAxes, ha="left", va="top")

    fig.tight_layout()
    return _save_all(fig, out_dir, "fig1_collision_time_scale", formats)


def figure_cooling_law(
    macro_root,
    dem_root,
    ar4_dsmc_file,
    out_dir,
    formats=("pdf", "png"),
):
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.6))
    ax_tau_ar, ax_t_ar, ax_tau_alpha, ax_t_alpha = axes.flat

    for ar in AR_VALUES_WITH_LIMIT:
        color = AR_COLORS[ar]
        dem = load_dem_case(dem_root, ar)
        mask = dem.tau > 0.0
        ax_tau_ar.plot(
            dem.tau[mask],
            _temperature_log(dem)[mask],
            color=color,
            lw=1.15,
            ls="-",
            label=fr"${ar:g}$",
        )
        if ar == 4.0:
            dsmc_series = [load_ar4_dsmc(ar4_dsmc_file)]
        else:
            dsmc_series = load_macro_case(macro_root, ar, 0.95)
        tau_end = min(float(np.nanmax(s.tau)) for s in dsmc_series)
        tau_grid = np.linspace(0.05, tau_end, 280)
        mean, _, _ = ensemble_mean_on_grid(
            dsmc_series, "tau", _temperature_log, tau_grid
        )
        ax_tau_ar.plot(tau_grid, mean, color=color, lw=1.15, ls="--")

        if ar != 4.0:
            t_end = min(float(np.nanmax(s.t)) for s in dsmc_series)
            t_grid = np.linspace(0.0, t_end, 280)
            mean, _, _ = ensemble_mean_on_grid(
                dsmc_series, "t", _haff_linear, t_grid
            )
            ax_t_ar.plot(t_grid, mean, color=color, lw=1.25, ls="--",
                         marker=MARKERS[ar], markevery=_marker_step(t_grid.size),
                         ms=3.8, mfc="white", mec=color)

    for alpha in ALPHA_SWEEP:
        color = ALPHA_COLORS[alpha]
        dsmc_series = load_macro_case(macro_root, 2.0, alpha)
        tau_end = min(float(np.nanmax(s.tau)) for s in dsmc_series)
        tau_grid = np.linspace(0.05, tau_end, 260)
        mean, _, _ = ensemble_mean_on_grid(
            dsmc_series, "tau", _temperature_log, tau_grid
        )
        ax_tau_alpha.plot(tau_grid, mean, color=color, lw=1.25, ls="-",
                          label=fr"${alpha:.2f}$")
        t_end = min(float(np.nanmax(s.t)) for s in dsmc_series)
        t_grid = np.linspace(0.0, t_end, 260)
        mean, _, _ = ensemble_mean_on_grid(
            dsmc_series, "t", _haff_linear, t_grid
        )
        ax_t_alpha.plot(t_grid, mean, color=color, lw=1.25, ls="-",
                        marker=MARKERS[alpha], markevery=_marker_step(t_grid.size),
                        ms=3.8, mfc="white", mec=color)

    ax_tau_ar.set_xlabel(r"$\tau$")
    ax_tau_ar.set_ylabel(r"$\ln[T/T(0)]$")
    ax_tau_ar.set_xlim(0.0, 115.0)
    ax_tau_ar.set_ylim(-1.55, 0.05)
    ax_tau_ar.text(0.03, 0.94, "(a)", transform=ax_tau_ar.transAxes, ha="left", va="top")
    legend_ar = _framed_legend(ax_tau_ar, title=r"$\mathrm{AR}$", fontsize=8, loc="lower left")
    ax_tau_ar.add_artist(legend_ar)
    _framed_legend(
        ax_tau_ar,
        handles=[
            Line2D([0], [0], color="black", ls="-", lw=1.2, label="DEM"),
            Line2D([0], [0], color="black", ls="--", lw=1.2, label="DSMC"),
        ],
        fontsize=8,
        loc="upper right",
    )

    ax_t_ar.set_xlabel(r"$t$")
    ax_t_ar.set_ylabel(r"$[T(0)/T(t)]^{1/2}$")
    ax_t_ar.set_xlim(-10.0, 1050.0)
    ax_t_ar.set_ylim(0.95, 2.05)
    ax_t_ar.text(0.03, 0.94, "(b)", transform=ax_t_ar.transAxes, ha="left", va="top")

    ax_tau_alpha.set_xlabel(r"$\tau$")
    ax_tau_alpha.set_ylabel(r"$\ln[T/T(0)]$")
    ax_tau_alpha.set_xlim(0.0, 105.0)
    ax_tau_alpha.set_ylim(-1.55, 0.05)
    ax_tau_alpha.text(0.03, 0.94, "(c)", transform=ax_tau_alpha.transAxes, ha="left", va="top")
    _framed_legend(ax_tau_alpha, title=r"$\alpha$", fontsize=8, loc="lower left")

    ax_t_alpha.set_xlabel(r"$t$")
    ax_t_alpha.set_ylabel(r"$[T(0)/T(t)]^{1/2}$")
    ax_t_alpha.set_xlim(-10.0, 1050.0)
    ax_t_alpha.set_ylim(0.95, 2.2)
    ax_t_alpha.text(0.03, 0.94, "(d)", transform=ax_t_alpha.transAxes, ha="left", va="top")

    for ax in axes.flat:
        _apply_style(ax)
        set_aspect_from_limits(ax, aspect=0.82)

    fig.tight_layout()
    return _save_all(fig, out_dir, "fig2_hcs_cooling_law", formats)


def figure_temperature_partition(
    macro_root,
    models_dir,
    out_dir,
    formats=("pdf", "png"),
):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax = axes[0]
    for alpha in ALPHA_SWEEP:
        color = ALPHA_COLORS[alpha]
        series = load_macro_case(macro_root, 2.0, alpha)
        t_end = min(float(np.nanmax(s.t)) for s in series)
        t_grid = np.linspace(0.0, min(360.0, t_end), 240)
        mean, stderr, traces = ensemble_mean_on_grid(series, "t", _theta, t_grid)
        for trace in traces:
            ax.plot(t_grid, trace, color=color, lw=0.45, alpha=0.22)
        ax.plot(t_grid, mean, color=color, lw=1.45, label=fr"${alpha:.2f}$")
        good = np.isfinite(stderr)
        ax.fill_between(
            t_grid[good],
            mean[good] - stderr[good],
            mean[good] + stderr[good],
            color=color,
            alpha=0.10,
            lw=0,
        )
    ax.axhline(1.0, color="0.25", lw=0.9, ls="--")
    ax.set_xlim(-5.0, 365.0)
    ax.set_ylim(0.80, 1.10)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\theta(t)=T_{\mathrm{tr}}/T_{\mathrm{rot}}$")
    ax.text(0.03, 0.94, "(a)", transform=ax.transAxes, ha="left", va="top")
    _framed_legend(ax, title=r"$\alpha$ ($\mathrm{AR}=2$)", fontsize=8, loc="lower right")
    _apply_style(ax)
    set_aspect_from_limits(ax, aspect=1.0)

    ax = axes[1]
    targets = load_theta_targets(models_dir, alpha_values=(0.90, 0.95))
    for alpha, rows in targets.items():
        rows = sorted(rows)
        ars = np.array([r[0] for r in rows])
        theta = np.array([r[1] for r in rows])
        color = ALPHA_COLORS[alpha]
        ax.plot(
            ars,
            theta,
            color=color,
            lw=1.25,
            marker=MARKERS[alpha],
            ms=4.5,
            mfc="white",
            mec=color,
            label=fr"$\alpha={alpha:.2f}$",
        )
    ax.axhline(1.0, color="0.25", lw=0.9, ls="--", label=r"$\theta=1$")
    ax.set_xlim(1.35, 3.15)
    ax.set_ylim(0.88, 1.04)
    ax.set_xticks(list(AR_VALUES))
    ax.set_xlabel(r"$\mathrm{AR}$")
    ax.set_ylabel(r"$\theta^*$")
    ax.text(0.03, 0.94, "(b)", transform=ax.transAxes, ha="left", va="top")
    _framed_legend(ax, fontsize=8, loc="lower right")
    _apply_style(ax)
    set_aspect_from_limits(ax, aspect=1.0)

    fig.tight_layout()
    return _save_all(fig, out_dir, "fig3_temperature_partition", formats)


def figure_speed_distribution(hist_root, out_dir, formats=("pdf", "png")):
    cases = load_hist_cases(hist_root)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax_p, ax_ratio = axes
    c_ref = np.linspace(0.0, 6.5, 500)
    ax_p.plot(c_ref, maxwell_speed_pdf(c_ref), color="black", lw=1.25,
              ls="--", label="Maxwellian")
    ax_ratio.axhline(1.0, color="black", lw=1.0, ls="--")

    for alpha in HIST_ALPHAS:
        hist = cases[alpha]["speed"]
        color = ALPHA_COLORS[alpha]
        x = hist.x
        p = hist.density_mean
        ref = maxwell_speed_pdf(x)
        valid = (p > 0.0) & np.isfinite(p)
        ax_p.plot(x[valid], p[valid], color=color, lw=1.2,
                  marker=MARKERS[alpha], markevery=_marker_step(valid.sum(), 14),
                  ms=3.5, mfc="white", mec=color, label=fr"$\alpha={alpha:.2f}$")
        ratio = np.where(ref > 1.0e-10, p / ref, np.nan)
        ratio_valid = valid & np.isfinite(ratio) & (x <= 5.0) & (ref > 1.0e-5)
        ax_ratio.plot(x[ratio_valid], ratio[ratio_valid], color=color, lw=1.2,
                      marker=MARKERS[alpha],
                      markevery=_marker_step(ratio_valid.sum(), 14),
                      ms=3.5, mfc="white", mec=color,
                      label=fr"$\alpha={alpha:.2f}$")

    ax_p.set_xlim(0.0, 5.2)
    ax_p.set_ylim(0.0, 0.95)
    ax_p.set_xlabel(r"$c=|\mathbf{v}-\mathbf{U}|/\sqrt{2T_{\mathrm{tr}}/m}$")
    ax_p.set_ylabel(r"$P(c)$")
    ax_p.text(0.03, 0.94, "(a)", transform=ax_p.transAxes, ha="left", va="top")
    _framed_legend(ax_p, fontsize=8, loc="upper right")

    ax_ratio.set_xlim(0.0, 5.0)
    ax_ratio.set_ylim(0.55, 3.2)
    ax_ratio.set_xlabel(r"$c$")
    ax_ratio.set_ylabel(r"$P(c)/P_M(c)$")
    ax_ratio.text(0.03, 0.94, "(b)", transform=ax_ratio.transAxes, ha="left", va="top")
    _framed_legend(ax_ratio, title=r"$\mathrm{AR}=2$", fontsize=8, loc="upper left")

    for ax in axes:
        _apply_style(ax)
        set_aspect_from_limits(ax, aspect=1.0)

    fig.tight_layout()
    return _save_all(fig, out_dir, "fig4_reduced_speed_distribution", formats)


def figure_rot_energy_distribution(hist_root, out_dir, formats=("pdf", "png")):
    cases = load_hist_cases(hist_root)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax_p, ax_ratio = axes
    x_ref = np.linspace(0.0, 12.0, 500)
    ax_p.semilogy(x_ref, maxwell_rot_energy_pdf(x_ref), color="black", lw=1.25,
                  ls="--", label=r"$e^{-x}$")
    ax_ratio.axhline(1.0, color="black", lw=1.0, ls="--")

    for alpha in HIST_ALPHAS:
        hist = cases[alpha]["rot_energy"]
        color = ALPHA_COLORS[alpha]
        x = hist.x
        p = hist.density_mean
        ref = maxwell_rot_energy_pdf(x)
        valid = (p > 0.0) & np.isfinite(p) & (x <= 12.0)
        ax_p.semilogy(x[valid], p[valid], color=color, lw=1.2,
                      marker=MARKERS[alpha], markevery=_marker_step(valid.sum(), 14),
                      ms=3.5, mfc="white", mec=color,
                      label=fr"$\alpha={alpha:.2f}$")
        ratio = np.where(ref > 1.0e-8, p / ref, np.nan)
        ratio_valid = valid & np.isfinite(ratio) & (x <= 10.0) & (ref > 1.0e-5)
        ax_ratio.plot(x[ratio_valid], ratio[ratio_valid], color=color, lw=1.2,
                      marker=MARKERS[alpha],
                      markevery=_marker_step(ratio_valid.sum(), 14),
                      ms=3.5, mfc="white", mec=color,
                      label=fr"$\alpha={alpha:.2f}$")

    ax_p.set_xlim(0.0, 10.0)
    ax_p.set_ylim(1.0e-5, 1.5)
    ax_p.set_xlabel(r"$x=E_{\mathrm{rot}}/T_{\mathrm{rot}}$")
    ax_p.set_ylabel(r"$P(x)$")
    ax_p.text(0.03, 0.94, "(a)", transform=ax_p.transAxes, ha="left", va="top")
    _framed_legend(ax_p, fontsize=8, loc="upper right")

    ax_ratio.set_xlim(0.0, 10.0)
    ax_ratio.set_ylim(0.45, 4.5)
    ax_ratio.set_xlabel(r"$x$")
    ax_ratio.set_ylabel(r"$P(x)/e^{-x}$")
    ax_ratio.text(0.03, 0.94, "(b)", transform=ax_ratio.transAxes, ha="left", va="top")
    _framed_legend(ax_ratio, title=r"$\mathrm{AR}=2$", fontsize=8, loc="upper left")

    for ax in axes:
        _apply_style(ax)
    set_aspect_from_limits(ax_p, aspect=1.0)
    set_aspect_from_limits(ax_ratio, aspect=1.0)

    fig.tight_layout()
    return _save_all(fig, out_dir, "fig5_rotational_energy_distribution", formats)


FIGURE_NAMES = ("collision_time", "cooling", "theta", "speed", "rot-energy", "all")


def generate_pof_figures(
    figure,
    root=DEFAULT_ROOT,
    macro_root=None,
    hist_root=None,
    dem_root=DEFAULT_DEM_ROOT,
    ar4_dsmc_file=DEFAULT_AR4_DSMC,
    models_dir="models",
    out_dir=None,
    formats=("pdf", "png"),
):
    root = Path(root)
    macro_root = Path(macro_root) if macro_root is not None else root / "macro"
    hist_root = Path(hist_root) if hist_root is not None else root / "hist"
    out_dir = Path(out_dir) if out_dir is not None else root / "pof_figures"
    formats = tuple(formats)
    written = []

    if figure in ("collision_time", "all"):
        written += figure_collision_time(
            macro_root, dem_root, ar4_dsmc_file, out_dir, formats
        )
    if figure in ("cooling", "all"):
        written += figure_cooling_law(
            macro_root, dem_root, ar4_dsmc_file, out_dir, formats
        )
    if figure in ("theta", "all"):
        written += figure_temperature_partition(
            macro_root, models_dir, out_dir, formats
        )
    if figure in ("speed", "all"):
        written += figure_speed_distribution(hist_root, out_dir, formats)
    if figure in ("rot-energy", "all"):
        written += figure_rot_energy_distribution(hist_root, out_dir, formats)
    return written


def validate_available_data(
    root=DEFAULT_ROOT,
    macro_root=None,
    hist_root=None,
    dem_root=DEFAULT_DEM_ROOT,
    ar4_dsmc_file=DEFAULT_AR4_DSMC,
):
    root = Path(root)
    macro_root = Path(macro_root) if macro_root is not None else root / "macro"
    hist_root = Path(hist_root) if hist_root is not None else root / "hist"
    macro_counts = {}
    for ar, alpha in [
        *[(ar, 0.95) for ar in AR_VALUES],
        *[(2.0, alpha) for alpha in (0.90, 0.70, 0.50)],
    ]:
        macro_counts[(ar, alpha)] = len(list_base_result_files(macro_root, ar, alpha))
    hist_counts = {}
    for alpha in HIST_ALPHAS:
        rdir = hist_result_dir(hist_root, 2.0, alpha)
        hist_counts[(2.0, alpha)] = len(list(rdir.glob("*_ng_hist_speed.txt")))
    external = {
        "ar4_dsmc": Path(ar4_dsmc_file).exists(),
        "dem": {ar: (Path(dem_root) / dem_ar_dir(ar) / "T.txt").exists()
                for ar in AR_VALUES_WITH_LIMIT},
    }
    return {"macro_counts": macro_counts, "hist_counts": hist_counts, "external": external}
