#!/usr/bin/env python3
"""Calibrate the f_tr correction factor C(alpha) for a given AR by bisection (Illinois).

For each alpha, runs single-seed DSMC with varying C until the steady-state
theta* = T_trans/T_rot matches the reference theta* from a pre-extracted table.

The theta* target table must be generated first:
    python extract_theta_target.py --config config/default.yaml --AR 2.0

Output: models/C_alpha_table_AR{ar_int}.json  (e.g. AR20 for AR=2.0)

Parallelism:
  --workers 1   Serial loop with incremental JSON saves (default, safe for debugging)
  --workers N   ProcessPoolExecutor: N alphas calibrate simultaneously, each worker
                writes its own temp JSON; results merged into main table at the end.

HPC (preferred): Use hpc/job_calibrate_C_alpha.slurm (SLURM array, one task per alpha).
"""
import argparse
import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import yaml

from src.simulation.collision import CollisionModels
from src.simulation.dsmc import run_simulation
from src.postprocessing.analysis import load_dsmc_results


# ---------------------------------------------------------------------------
# θ* helpers
# ---------------------------------------------------------------------------

def _dsmc_theta_star(result_path, tail_fraction=0.3):
    """Extract steady-state theta* from a DSMC output file."""
    t, tau, T_trans, T_rot, T_total = load_dsmc_results(result_path)
    theta = T_trans / T_rot
    n_tail = max(10, int(tail_fraction * len(theta)))
    return float(np.mean(theta[-n_tail:]))


# ---------------------------------------------------------------------------
# Single DSMC run with a given C_alpha override
# ---------------------------------------------------------------------------

def _run_with_C(base_config, models, alpha, C_alpha, seed, output_dir):
    """Run one DSMC realization and return its steady-state theta*."""
    cfg = copy.deepcopy(base_config)
    cfg['system']['alpha'] = float(alpha)
    cfg['system']['C_alpha'] = float(C_alpha)
    cfg['simulation']['seeds'] = [seed]

    os.makedirs(output_dir, exist_ok=True)
    alpha_int = int(round(alpha * 100))
    fname = f"calib_C{C_alpha:.5f}_a{alpha_int:03d}_s{seed}.txt"
    out_path = os.path.join(output_dir, fname)
    pressure_path = os.path.join(
        output_dir, f"calib_C{C_alpha:.5f}_a{alpha_int:03d}_s{seed}_pressure.txt"
    )

    run_simulation(cfg, models, seed, out_path, pressure_path)
    return _dsmc_theta_star(out_path)


# ---------------------------------------------------------------------------
# Illinois root-finder (replaces pure bisection — ~2x fewer evaluations)
# ---------------------------------------------------------------------------

def _illinois_C(base_config, models, alpha, theta_target, seed, output_dir,
                C_lo=1.0, C_hi=1.5, tol=5e-3, max_iter=20, verbose=True):
    """Find C such that theta*_DSMC(C) ≈ theta_target using the Illinois algorithm.

    Physics: larger C → more translational dissipation → lower theta*.
    So theta*(C) is monotonically decreasing in C.

    The Illinois algorithm is a bracketed root-finder that uses secant
    interpolation to converge faster than bisection while retaining the
    bracketing safety guarantee.  Typical saving: ~40-50% fewer evaluations.
    """
    def eval_C(C):
        th = _run_with_C(base_config, models, alpha, C, seed, output_dir)
        if verbose:
            print(f"    C={C:.5f}  theta*_DSMC={th:.5f}  target={theta_target:.5f}")
        return th

    # Evaluate bracket endpoints
    th_lo = eval_C(C_lo)
    th_hi = eval_C(C_hi)

    if th_lo < theta_target:
        print(f"  Warning: theta*(C_lo={C_lo}) = {th_lo:.4f} < target {theta_target:.4f}. "
              f"Widening bracket.")
        C_lo = 0.01
        th_lo = eval_C(C_lo)
    if th_hi > theta_target:
        print(f"  Warning: theta*(C_hi={C_hi}) = {th_hi:.4f} > target {theta_target:.4f}. "
              f"Widening bracket.")
        C_hi = 5.0
        th_hi = eval_C(C_hi)

    # Illinois iteration
    # f(C) = theta*(C) - theta_target; root at f=0, f decreasing in C
    f_lo = th_lo - theta_target
    f_hi = th_hi - theta_target
    side = 0   # 0 = last update was lo-side; 1 = hi-side

    residual = float('inf')
    C_new = (C_lo + C_hi) / 2.0   # fallback if loop exits early

    for i in range(max_iter):
        # Secant step
        C_new = C_hi - f_hi * (C_hi - C_lo) / (f_hi - f_lo)
        th_new = eval_C(C_new)
        f_new  = th_new - theta_target
        residual = abs(f_new)

        if residual < tol:
            if verbose:
                print(f"  Converged after {i + 3} evaluations: "
                      f"C={C_new:.5f}, residual={f_new:.5f}")
            return C_new

        # Update bracket
        if f_new * f_hi < 0:
            # Root between C_new and C_hi: replace lo
            C_lo, f_lo = C_hi, f_hi
            side = 0
        else:
            # Root between C_lo and C_new: replace hi, apply Illinois modifier
            if side == 1:
                f_lo *= 0.5   # deflate stale lo endpoint to force progress
            side = 1
        C_hi, f_hi = C_new, f_new

    print(f"  Warning: Illinois did not converge for alpha={alpha:.2f} "
          f"(final residual={residual:.5f})")
    return C_new


# ---------------------------------------------------------------------------
# Worker function for parallel mode
# ---------------------------------------------------------------------------

def _calibrate_one_alpha(task):
    """Top-level worker: load models independently, run Illinois, write temp JSON.

    Args:
        task (dict): serialisable task bundle with keys:
            base_config, alpha, AR, theta_target, seed, output_dir,
            tmp_json_path, tol, max_iter, model_dir, gmm_path, ftr_path
    Returns:
        (alpha, C_opt)
    """
    alpha      = task['alpha']
    AR         = task['AR']
    seed       = task['seed']
    tol        = task['tol']
    max_iter   = task['max_iter']
    output_dir = task['output_dir']

    # Each worker loads its own CollisionModels (not picklable across processes)
    models = CollisionModels(
        task['model_dir'],
        gmm_npz_path=task['gmm_path'],
        ftr_params_path=task.get('ftr_path'),
    )

    C_opt = _illinois_C(
        task['base_config'], models, alpha, task['theta_target'],
        seed=seed, output_dir=output_dir,
        tol=tol, max_iter=max_iter, verbose=True,
    )

    # Write per-worker temp JSON (no contention)
    key = f"({alpha:.3f}, {AR:.1f})"
    os.makedirs(os.path.dirname(task['tmp_json_path']) or ".", exist_ok=True)
    with open(task['tmp_json_path'], 'w') as f:
        json.dump({key: float(C_opt)}, f, indent=2)

    return alpha, C_opt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate C(alpha, AR) correction factor for f_tr from LAMMPS HCS data"
    )
    parser.add_argument("--config",  default="config/default.yaml")
    parser.add_argument("--AR",      type=float, default=None,
                        help="Aspect ratio to calibrate. Default: from config.")
    parser.add_argument(
        "--alphas", default=None,
        help="Comma-separated alpha values, e.g. 0.65,0.70. Default: all in config."
    )
    parser.add_argument("--seed",     type=int,   default=42)
    parser.add_argument("--tol",      type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int,   default=20)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel workers for local multiprocessing (default 1 = serial). "
             "For HPC, use the SLURM array job instead."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: models/C_alpha_table_AR{ar_int}.json"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory for DSMC output files. Default: runs/calib_C_alpha"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-calibrate even if the alpha is already in the table."
    )
    parser.add_argument(
        "--theta-table", default=None,
        help="Path to theta_target JSON table produced by extract_theta_target.py. "
             "Default: calibration.theta_target_table_file from config."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # AR: CLI > config
    AR = args.AR if args.AR is not None else float(
        config['preprocessing']['zr_eff'].get('AR', 2.0)
    )
    ar_int = int(round(AR * 10))

    # Alpha values
    if args.alphas:
        alpha_values = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    else:
        alpha_values = config['preprocessing']['zr_eff']['alpha_values']
        alpha_values = [a for a in alpha_values if a < 1.0]

    # Theta target table
    theta_table_path = args.theta_table or config.get('calibration', {}).get(
        'theta_target_table_file',
        f"models/theta_target_table_AR{ar_int:02d}.json"
    )
    if not os.path.exists(theta_table_path):
        raise FileNotFoundError(
            f"theta_target table not found: {theta_table_path}\n"
            f"Run first:  python extract_theta_target.py --config {args.config} --AR {AR}"
        )
    with open(theta_table_path) as f:
        theta_table = json.load(f)
    print(f"Loaded theta_target table: {theta_table_path} ({len(theta_table)} entries)")

    # Paths — all derived from AR
    output_path = args.output or \
        config.get('calibration', {}).get(
            'C_alpha_table_file',
            f"models/C_alpha_table_AR{ar_int:02d}.json"
        )
    output_dir = args.output_dir or "runs/calib_C_alpha"

    # Model paths
    model_dir = config['preprocessing']['model_output_dir']
    # GMM path is always derived from the AR being calibrated, not from
    # config's gmm_cond_file (which is the default for normal simulations
    # and may point to a different AR).
    gmm_path = f"models/gmm_cond_AR{ar_int:02d}.npz"
    ftr_path  = config['preprocessing'].get('ftr', {}).get('ftr_params_file')

    # Build calibration config
    calib_cfg = copy.deepcopy(config)
    calib_cfg['particle']['AR'] = AR
    calib_cfg['time']['t_end'] = config.get('calibration_sweep', {}).get('t_end', 300)
    calib_cfg['time']['equilibration_time'] = config.get(
        'calibration_sweep', {}).get('equilibration_time', 2.0)
    calib_cfg['simulation']['output_dir'] = output_dir
    calib_cfg['system']['C_alpha'] = None

    # Load existing table (resume support)
    if os.path.exists(output_path):
        with open(output_path) as f:
            C_table = json.load(f)
        print(f"Resuming from existing table: {output_path} ({len(C_table)} entries)")
    else:
        C_table = {}

    print(f"\nCalibrating C(alpha) for AR={AR}, alphas={alpha_values}")
    print(f"Theta table: {theta_table_path}")
    print(f"Seed: {args.seed}, tol: {args.tol}, max_iter: {args.max_iter}, "
          f"workers: {args.workers}\n")

    # Determine which alphas actually need running
    tasks_to_run = []
    for alpha in alpha_values:
        key = f"({alpha:.3f}, {AR:.1f})"
        if key in C_table and not args.force:
            print(f"[{alpha:.2f}] Already in table (C={C_table[key]:.5f}), skipping.")
            continue
        if key not in theta_table:
            print(f"[{alpha:.2f}] Skipping: theta_target not found in {theta_table_path} for key {key}")
            continue
        theta_target = float(theta_table[key])
        print(f"[{alpha:.2f}] theta*_target = {theta_target:.5f}")

        alpha_int = int(round(alpha * 100))
        tasks_to_run.append({
            'alpha':        alpha,
            'AR':           AR,
            'theta_target': theta_target,
            'seed':         args.seed,
            'tol':          args.tol,
            'max_iter':     args.max_iter,
            'base_config':  calib_cfg,
            'output_dir':   output_dir,
            'model_dir':    model_dir,
            'gmm_path':     gmm_path,
            'ftr_path':     ftr_path,
            # temp JSON path used in parallel mode
            'tmp_json_path': os.path.join(
                output_dir,
                f"C_alpha_AR{ar_int:02d}_{alpha_int:03d}.json"
            ),
        })

    if not tasks_to_run:
        print("Nothing to calibrate.")
        _print_summary(C_table, output_path)
        return

    # -----------------------------------------------------------------------
    # Serial mode (workers == 1): incremental saves, models loaded once
    # -----------------------------------------------------------------------
    if args.workers <= 1:
        print("Loading collision models...")
        models = CollisionModels(model_dir, gmm_npz_path=gmm_path,
                                 ftr_params_path=ftr_path)

        for task in tasks_to_run:
            alpha = task['alpha']
            print(f"\n[{alpha:.2f}] Running Illinois calibration...")
            C_opt = _illinois_C(
                calib_cfg, models, alpha, task['theta_target'],
                seed=args.seed, output_dir=output_dir,
                tol=args.tol, max_iter=args.max_iter,
            )
            key = f"({alpha:.3f}, {AR:.1f})"
            C_table[key] = float(C_opt)
            print(f"  --> C({alpha:.2f}) = {C_opt:.5f}\n")

            # Save after each alpha in case of interruption
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(C_table, f, indent=2)

    # -----------------------------------------------------------------------
    # Parallel mode (workers > 1): each worker loads its own models
    # -----------------------------------------------------------------------
    else:
        print(f"Launching {min(args.workers, len(tasks_to_run))} parallel workers "
              f"for {len(tasks_to_run)} alpha(s)...\n")
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_calibrate_one_alpha, t): t['alpha']
                       for t in tasks_to_run}
            for fut in as_completed(futures):
                alpha, C_opt = fut.result()
                results[alpha] = C_opt
                print(f"  [done] alpha={alpha:.2f}  C={C_opt:.5f}")

        # Merge temp JSONs into main table
        for task in tasks_to_run:
            alpha = task['alpha']
            key   = f"({alpha:.3f}, {AR:.1f})"
            if alpha in results:
                C_table[key] = float(results[alpha])

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dict(sorted(C_table.items())), f, indent=2)

    _print_summary(C_table, output_path)


def _print_summary(C_table, output_path):
    print(f"\nCalibration complete. Table saved to: {output_path}")
    print("Summary:")
    for key, val in sorted(C_table.items()):
        print(f"  {key}: C = {val:.5f}")


if __name__ == "__main__":
    main()
