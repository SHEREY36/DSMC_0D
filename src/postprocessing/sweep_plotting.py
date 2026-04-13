import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from .analysis import load_dsmc_results, compute_temperature_ratio


ALPHA_DIR_PATTERN = re.compile(r"^alpha_(\d{3})$")


def _discover_case_dirs(sweep_root):
    cases = []
    if not os.path.isdir(sweep_root):
        return cases

    for name in sorted(os.listdir(sweep_root)):
        match = ALPHA_DIR_PATTERN.match(name)
        if not match:
            continue
        alpha = int(match.group(1)) / 100.0
        case_dir = os.path.join(sweep_root, name)
        if os.path.isdir(case_dir):
            cases.append((alpha, case_dir))
    return cases


def _collect_seed_files(case_dir):
    results_dir = os.path.join(case_dir, "results")
    if not os.path.isdir(results_dir):
        return []
    return sorted(glob.glob(os.path.join(results_dir, "*.txt")))


def _case_statistics(
    result_files,
    n_time_points=1200,
    asymptotic_tail_fraction=0.2,
    min_realizations=1,
    reference_realization_index=0,
):
    if not result_files or len(result_files) < int(min_realizations):
        return None

    series = []
    valid_files = []
    t_start = -np.inf
    t_end = np.inf
    for path in result_files:
        try:
            t, tau, T_trans, T_rot, T_total = load_dsmc_results(path)
        except (OSError, ValueError) as exc:
            print(f"Skipping invalid realization file: {path} ({exc})")
            continue
        theta = compute_temperature_ratio(T_trans, T_rot)
        series.append((t, theta, T_total))
        valid_files.append(path)
        t_start = max(t_start, t[0])
        t_end = min(t_end, t[-1])

    if len(valid_files) < int(min_realizations):
        return None

    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end < t_start:
        return None

    if np.isclose(t_end, t_start):
        t_grid = np.array([t_start], dtype=float)
    else:
        t_grid = np.linspace(t_start, t_end, n_time_points)
    theta_curves = []
    asymptotic_theta_per_realization = []

    for t, theta, _ in series:
        theta_curves.append(np.interp(t_grid, t, theta))

        t_tail = t[-1] * (1.0 - asymptotic_tail_fraction)
        mask = t >= t_tail
        if not np.any(mask):
            mask = np.ones_like(t, dtype=bool)
        asymptotic_theta_per_realization.append(float(np.mean(theta[mask])))

    theta_arr = np.vstack(theta_curves)
    theta_mean = np.mean(theta_arr, axis=0)

    # Asymptotic metric requested: tail-mean of the mean theta(t) curve.
    t_tail_mean_curve = t_grid[-1] * (1.0 - asymptotic_tail_fraction)
    mean_curve_tail_mask = t_grid >= t_tail_mean_curve
    if not np.any(mean_curve_tail_mask):
        mean_curve_tail_mask = np.ones_like(t_grid, dtype=bool)
    theta_asymptotic_from_mean = float(
        np.mean(theta_mean[mean_curve_tail_mask])
    )

    # Keep per-realization asymptotic spread for error bars.
    asymptotic_arr = np.array(asymptotic_theta_per_realization, dtype=float)

    ref_idx = int(reference_realization_index) % len(valid_files)
    t_ref, _, _, _, T_total_ref = load_dsmc_results(valid_files[ref_idx])

    return {
        "t": t_grid,
        "theta_mean": theta_mean,
        "theta_std": np.std(theta_arr, axis=0),
        "theta_asymptotic_from_mean": theta_asymptotic_from_mean,
        "theta_asymptotic_std": float(np.std(asymptotic_arr)),
        "n_realizations": len(valid_files),
        "t_ref": t_ref,
        "T_total_ref": T_total_ref,
        "reference_file": valid_files[ref_idx],
    }


def build_sweep_statistics(
    sweep_root,
    alpha_filter=None,
    alpha_exclude=None,
    n_time_points=1200,
    asymptotic_tail_fraction=0.2,
    min_realizations=1,
    reference_realization_index=0,
):
    cases = _discover_case_dirs(sweep_root)
    if alpha_filter:
        alpha_set = {round(float(a), 3) for a in alpha_filter}
        cases = [(a, d) for a, d in cases if round(a, 3) in alpha_set]
    if alpha_exclude:
        alpha_exclude_set = {round(float(a), 3) for a in alpha_exclude}
        cases = [(a, d) for a, d in cases if round(a, 3) not in alpha_exclude_set]

    stats = []
    for alpha, case_dir in cases:
        files = _collect_seed_files(case_dir)
        case_stats = _case_statistics(
            files,
            n_time_points=n_time_points,
            asymptotic_tail_fraction=asymptotic_tail_fraction,
            min_realizations=min_realizations,
            reference_realization_index=reference_realization_index,
        )
        if case_stats is None:
            continue
        case_stats["alpha"] = alpha
        case_stats["case_dir"] = case_dir
        stats.append(case_stats)

    stats.sort(key=lambda s: s["alpha"])
    return stats


def plot_theta_means_over_time(stats, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in stats:
        ax.plot(
            s["t"], s["theta_mean"],
            label=f"e={s['alpha']:.2f} (n={s['n_realizations']})"
        )

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$\langle T_{tr}/T_{rot}\rangle$")
    ax.set_title(r"Mode B: $T_{tr}/T_{rot}$ vs Time")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep plot saved: {output_path}")


def plot_asymptotic_theta_vs_alpha(stats, output_path):
    alphas = np.array([s["alpha"] for s in stats], dtype=float)
    means = np.array([s["theta_asymptotic_from_mean"] for s in stats], dtype=float)
    stds = np.array([s["theta_asymptotic_std"] for s in stats], dtype=float)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(alphas, means, yerr=stds, marker="o", capsize=3, lw=1.2)
    ax.set_xlabel(r"Coefficient of restitution $e$")
    ax.set_ylabel(r"Asymptotic $T_{tr}/T_{rot}$")
    ax.set_title("HCS Spherocylinders (Mode B)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep plot saved: {output_path}")


def plot_total_temperature_loglog(
    stats, output_path, normalize_total_temperature=True
):
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in stats:
        t = s["t_ref"]
        temp = s["T_total_ref"].copy()
        if normalize_total_temperature and temp[0] != 0.0:
            temp = temp / temp[0]
        mask = t > 0.0
        ax.loglog(t[mask], temp[mask], label=f"e={s['alpha']:.2f}")

    ylabel = r"$T_{total}(t)/T_{total}(0)$" \
        if normalize_total_temperature else r"$T_{total}(t)$"
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title("HCS Haff-Law Overlay (Mode B)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Sweep plot saved: {output_path}")


def run_sweep_postprocessing(
    sweep_root,
    figures_dir,
    alpha_filter=None,
    alpha_exclude=None,
    n_time_points=1200,
    asymptotic_tail_fraction=0.2,
    normalize_total_temperature=True,
    min_realizations=1,
    reference_realization_index=0,
):
    os.makedirs(figures_dir, exist_ok=True)
    stats = build_sweep_statistics(
        sweep_root=sweep_root,
        alpha_filter=alpha_filter,
        alpha_exclude=alpha_exclude,
        n_time_points=n_time_points,
        asymptotic_tail_fraction=asymptotic_tail_fraction,
        min_realizations=min_realizations,
        reference_realization_index=reference_realization_index,
    )

    if not stats:
        print(f"No sweep results found under {sweep_root}")
        return False

    plot_theta_means_over_time(
        stats, os.path.join(figures_dir, "theta_mean_vs_time.png")
    )
    plot_asymptotic_theta_vs_alpha(
        stats, os.path.join(figures_dir, "theta_asymptotic_vs_alpha.png")
    )
    plot_total_temperature_loglog(
        stats, os.path.join(figures_dir, "total_temperature_loglog.png"),
        normalize_total_temperature=normalize_total_temperature
    )
    return True
