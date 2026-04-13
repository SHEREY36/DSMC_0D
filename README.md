# DSMC_0D — 0D DSMC for Spherocylinder Granular Gases

A spatially homogeneous (0D) Direct Simulation Monte Carlo (DSMC) simulation for granular gases of non-spherical (spherocylinder) particles. Supports two flow modes:

- **HCS** (Homogeneous Cooling State): inelastic cooling, temperatures decay monotonically.
- **USF** (Uniform Simple Shear Flow): shear-driven steady state, temperatures plateau.

All collision sub-models are data-driven, fitted from DEM/CTC trajectory data.

---

## Installation

**Requirements:** Python 3.8+, `numpy`, `scipy`, `matplotlib`, `pyyaml`, `scikit-learn`

```bash
git clone https://github.com/SHEREY36/DSMC_0D.git
cd DSMC_0D
pip install numpy scipy matplotlib pyyaml scikit-learn
```

Pre-trained models are included in `models/`. No pre-processing is needed to run simulations.

---

## Quick Start: Run a Single Case

The fastest way to run one specific (AR, alpha) case is to set the parameters directly in `config/default.yaml` and run:

```bash
# Edit config/default.yaml: set particle.AR, system.alpha, simulation.seeds
python run_simulation.py
```

Or pass any modified config file:

```bash
python run_simulation.py --config config/default.yaml
```

**Key parameters to set for a single case:**

```yaml
particle:
  AR: 2.0          # Aspect ratio (1.1, 1.25, 1.5, 2.0, 2.5, 3.0 available)

system:
  alpha: 0.95      # Coefficient of restitution (0.60–1.00)
  kTt: 1.0         # Initial translational temperature
  kTr: 1.0         # Initial rotational temperature
  eta: 1.0         # Prefactor in Zr(theta) = (0.39*theta^2 + 0.09*theta + 1.67)*eta
  phi: 0.01        # Volume fraction

time:
  t_end: 200       # Total simulation time
  equilibration_time: 0.0  # Elastic warm-up period before inelastic collisions begin

simulation:
  seeds: [42]      # One seed = one realization
  output_dir: results/

preprocessing:
  gmm:
    gmm_cond_file: models/gmm_cond_AR20.npz  # Match your AR (see table below)
  model_output_dir: models/
```

Output is written to `results/AR{AR}_COR{alpha*100}_R{realization}.txt`.

**Make sure `gmm_cond_file` matches your chosen AR** — see the [Available Models](#available-models) table.

---

## Flow Modes

### HCS (default)

```yaml
flow:
  mode: hcs
  shear_rate: 0.0
```

Temperatures decay to zero. `theta = T_trans / T_rot` evolves toward a steady-state ratio determined by `alpha` and `eta`.

### USF (Uniform Simple Shear Flow)

```yaml
flow:
  mode: usf
  shear_rate: 0.1   # gdot — shear rate in simulation units
```

A velocity shear gradient `dVx/dy = gdot` is applied each timestep. Translational and rotational temperatures reach nonzero plateaus. All collision machinery (GMM sampling, dissipation, Zr model) is unchanged — only the drift step is added.

---

## Step-by-Step Workflows

### 1. Single simulation (one AR, one alpha)

```bash
# Edit config/default.yaml to set AR, alpha, seeds, output_dir, gmm_cond_file
python run_simulation.py --config config/default.yaml
```

### 2. Post-process a single simulation

```bash
python run_postprocessing.py --config config/default.yaml
```

Generates three plots in `postprocessing.figures_dir`:
- `temperature_evolution.png` — total temperature vs time
- `temperature_components.png` — T_trans and T_rot separately
- `temperature_ratio.png` — `theta = T_trans / T_rot` vs time

### 3. Alpha sweep (batch over multiple alpha values)

Prepares per-alpha case folders under `calibration_sweep.output_root` (default: `runs/AR2_eta_sweep/`).

```bash
# Step 1: Create case folders and per-case config.yaml files (no simulation yet)
python run_alpha_sweep.py --config config/default.yaml --prepare-only

# Step 2: Run all alpha cases
python run_alpha_sweep.py --config config/default.yaml

# Step 3: Run a specific subset of alpha values
python run_alpha_sweep.py --config config/default.yaml --alphas 0.80,0.85,0.90

# Step 4: Run with parallel workers
python run_alpha_sweep.py --config config/default.yaml --workers 4

# Step 5: Run without generating sweep plots afterwards
python run_alpha_sweep.py --config config/default.yaml --skip-post
```

Sweep folder structure:

```
runs/AR2_eta_sweep/
  alpha_050/config.yaml, results/, figures/
  alpha_060/config.yaml, results/, figures/
  ...
  alpha_100/config.yaml, results/, figures/
  eta_profile.csv
```

Sweep post-processing produces (in `postprocessing.sweep_figures_dir`):
- `theta_mean_vs_time.png`
- `theta_asymptotic_vs_alpha.png`
- `total_temperature_loglog.png`

### 4. Post-process a sweep

```bash
# Uses sweep_root from config
python run_postprocessing.py --config config/default.yaml

# Override sweep folder
python run_postprocessing.py --config config/default.yaml \
    --sweep-root runs/AR2_eta_sweep

# Filter to specific alphas
python run_postprocessing.py --config config/default.yaml \
    --alphas 0.65,0.70,0.75,0.80

# Exclude low alphas, require at least 3 realizations
python run_postprocessing.py --config config/default.yaml \
    --exclude-alphas 0.50,0.55,0.60 \
    --min-realizations 3

# Save figures to a different directory
python run_postprocessing.py --config config/default.yaml \
    --figures-dir runs/AR2_eta_sweep/my_figures
```

### 5. Calibrate C(alpha) correction factor

`C_alpha` corrects the translational fraction of dissipated energy to match LAMMPS HCS reference data. Requires `LAMMPS_data/` to be present.

```bash
# Serial (default): calibrates all alpha values, saves incrementally
python calibrate_C_alpha.py --config config/default.yaml

# Calibrate only specific alpha values
python calibrate_C_alpha.py --config config/default.yaml --alphas 0.80,0.85,0.90

# Parallel: 4 alpha values simultaneously
python calibrate_C_alpha.py --config config/default.yaml --workers 4

# Override AR (default: from config)
python calibrate_C_alpha.py --config config/default.yaml --AR 2.0

# Force re-calibration even if already in table
python calibrate_C_alpha.py --config config/default.yaml --force

# Adjust convergence tolerance and iteration limit
python calibrate_C_alpha.py --config config/default.yaml --tol 5e-4 --max-iter 25
```

Output: `models/C_alpha_table_AR20.json`. Intermediate DSMC runs are saved in `runs/calib_C_alpha/`.

### 6. Validate calibration against LAMMPS

```bash
python validate_C_alpha.py
```

Generates `figures/validation_C_alpha_AR2.png` — a three-panel comparison (collision frequency, temperature ratio `theta`, normalised total temperature) between DSMC and LAMMPS for all calibrated alpha values.

### 7. Re-train collision models from CTC data (optional)

Only needed if you have new CTC data. Computationally expensive.

```bash
python run_preprocessing.py --config config/default.yaml
```

Fits and saves to `models/`:
- `gmm_cond_AR*.npz` — conditional GMM for energy redistribution
- `scattering_coeffs.npz` — polynomial scattering angle PDF
- `gamma_max_table.json` — max dissipation fraction lookup
- `one_hit_table.json` — single-contact probability lookup

---

## Custom Configuration

To run a case without touching `config/default.yaml`:

```bash
cp config/default.yaml runs/my_case/config.yaml
# Edit runs/my_case/config.yaml
python run_simulation.py --config runs/my_case/config.yaml
python run_postprocessing.py --config runs/my_case/config.yaml
```

Key sections to edit:

```yaml
particle:
  AR: 2.0

system:
  alpha: 0.80
  eta: 1.0

time:
  t_end: 200

flow:
  mode: hcs          # or 'usf'
  shear_rate: 0.0    # gdot, only used when mode=usf

simulation:
  seeds: [42, 123, 999]
  output_dir: runs/my_case/results/

preprocessing:
  gmm:
    gmm_cond_file: models/gmm_cond_AR20.npz
  model_output_dir: models/

postprocessing:
  results_dir: runs/my_case/results/
  figures_dir: runs/my_case/figures/
```

---

## Available Models

| Aspect Ratio | GMM file | AR label |
|---|---|---|
| AR = 1.1 | `gmm_cond_AR11.npz` | AR11 |
| AR = 1.25 | `gmm_cond_AR125.npz` | AR125 |
| AR = 1.5 | `gmm_cond_AR15.npz` | AR15 |
| AR = 2.0 | `gmm_cond_AR20.npz` | AR20 |
| AR = 2.5 | `gmm_cond_AR25.npz` | AR25 |
| AR = 3.0 | `gmm_cond_AR30.npz` | AR30 |

Dissipation tables (`gamma_max_table.json`, `one_hit_table.json`) cover:
- **alpha**: 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
- **AR**: 1.0, 1.5, 2.0, 2.5, 3.0, 4.0

Values outside this grid are linearly interpolated/extrapolated along the alpha axis at fixed AR.

---

## Output Format

Results are written as space-separated text files:

```
# t           tau          T_trans      T_rot        T_total
  0.000000     0.000000     1.001760     0.999853     1.000997
  0.700000     0.100062     0.999100     0.998702     0.999559
  ...
```

| Column | Description |
|---|---|
| `t` | Simulation time |
| `tau` | Cumulative collisions per particle (NColl / Np) |
| `T_trans` | Translational granular temperature |
| `T_rot` | Rotational granular temperature |
| `T_total` | Total temperature `(3*T_trans + 2*T_rot) / 5` |

USF output files are tagged with `_USF` in the filename: `AR2_COR95_USF_R1.txt`.

---

## Project Structure

```
DSMC_0D/
├── config/
│   └── default.yaml              # All simulation + model parameters
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py        # Load CTC data
│   │   ├── gmm_energy.py         # ConditionalGMM class (.npz load + sampling)
│   │   ├── scattering_angle.py   # p(chi|AR, alpha) polynomial model
│   │   ├── dissipation.py        # gamma_max / one_hit lookup + interpolation
│   │   ├── relaxation.py         # Zr(theta, eta) model, prepare_theta()
│   │   ├── ftr_distribution.py   # f_tr Laplace table loading/lookup
│   │   ├── zr_eff_table.py       # Z_R_eff fixed-point table
│   │   └── fit_all.py            # Preprocessing orchestrator
│   ├── simulation/
│   │   ├── particle.py           # SpherocylinderParams, sigma_c, volume, mI
│   │   ├── collision.py          # CollisionModels loader, chi/dissipation sampling
│   │   ├── dsmc.py               # Main DSMC loop (NTC, T-R exchange, dissipation, USF)
│   │   └── alpha_sweep.py        # Batch sweep: case preparation + parallel dispatch
│   └── postprocessing/
│       ├── analysis.py           # load_dsmc_results(), load_dem_results()
│       ├── plotting.py           # Single-case temperature plots
│       └── sweep_plotting.py     # Sweep aggregation and comparison plots
├── models/                       # Serialized model artifacts (pre-computed)
├── CTC_data/                     # DEM collision trajectory data
├── LAMMPS_data/                  # LAMMPS HCS reference data (for calibration)
├── results/                      # Default simulation output
├── run_simulation.py             # Entry point: run DSMC
├── run_alpha_sweep.py            # Entry point: alpha sweep
├── run_postprocessing.py         # Entry point: analyze and plot
├── run_preprocessing.py          # Entry point: fit models from CTC data
├── calibrate_C_alpha.py          # Calibrate C(alpha) against LAMMPS reference
└── validate_C_alpha.py           # Validation plots: DSMC vs LAMMPS
```

---

## Physics Reference

### Spherocylinder Geometry
- Cylinder length: `l_cyl = (AR - 1) * d`
- Caps: two hemispheres of diameter `d`
- Collision cross-section: `sigma_c = (0.32*AR^2 + 0.694*AR - 0.0213) * pi`

### Collision Sub-models
| Sub-model | Description |
|---|---|
| **GMM energy redistribution** | Conditional GMM: `p(epsilon_tr_f, epsilon_r1_f | theta, epsilon_tr_i, epsilon_r1_i)` |
| **Scattering angle** | `p(chi | AR, alpha)` polynomial PDF, rejection-sampled |
| **Dissipation** | `gamma ~ Beta(1.21, 3.67) * gamma_max(alpha, AR) * p_one_hit(alpha, AR)` |
| **Rotational relaxation** | `Zr(theta, eta) = (0.39*theta^2 + 0.09*theta + 1.67) * eta`; `P_r = min(1/Zr, 0.5)` |
| **f_tr correction** | `f_tr = C_alpha * 3*theta / (3*theta + 2)` splits dissipated energy between T and R |

### USF Drift Step
Applied once per timestep, before NTC collision selection:
```
vel[:, 0] -= gdot * vel[:, 1] * dt
vel -= vel.mean(axis=0)          # remove bulk momentum
```

---

## Troubleshooting

**`KeyError` for (alpha, AR) pair** — the dissipation tables do not cover that combination. Check [Available Models](#available-models) and ensure your alpha/AR values are within range (or close enough for interpolation).

**GMM file not found** — set `preprocessing.gmm.gmm_cond_file` to the `.npz` matching your AR. The file name encodes AR (e.g. `gmm_cond_AR20.npz` for AR=2.0).

**Simulation runs but temperatures don't decay (HCS)** — check that `system.alpha < 1.0` and `flow.mode: hcs`.

**USF temperatures grow without bound** — `gdot` is too large relative to the dissipation rate. Reduce `flow.shear_rate` or increase `system.alpha` toward 1.0.

**Slow simulation** — reduce `system.phi` or `system.domain` (fewer particles), reduce `time.t_end`, or use a single seed (`seeds: [42]`).
