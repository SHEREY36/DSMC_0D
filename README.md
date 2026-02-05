# DSMC_0D - 0D Direct Simulation Monte Carlo for Spherocylinder Particles

A spatially homogeneous (0D) Direct Simulation Monte Carlo (DSMC) simulation for granular gas cooling of non-spherical particles (spherocylinders). The simulation models the Homogeneous Cooling State (HCS) where particle translational and rotational temperatures decay over time through inelastic collisions.

## Features

- **Data-driven collision models**: Trained from DEM (Discrete Element Method) collision trajectory computation (CTC) data
- **Efficient conditional GMM sampling**: Pre-computed matrices avoid runtime inversions (~200μs per sample)
- **NTC collision selection**: No-Time-Counter method for efficient collision partner selection
- **Flexible configuration**: YAML-based parameter management
- **Multiple realizations**: Run multiple seeds in parallel for statistical averaging

## Physics Background

### Particle Geometry
- **Spherocylinders**: Cylinder of length `l_cyl = (AR - 1) × d` capped by two hemispheres of diameter `d`
- **Aspect Ratio**: `AR = total_length / d`

### Key Parameters
- **Coefficient of restitution** `α` (0.60 - 1.00): Controls energy dissipation per collision
- **Temperature ratio** `θ = T_trans / T_rot`: Ratio of translational to rotational granular temperature
- **Volume fraction** `φ`: Packing density of particles in the domain

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SHEREY36/DSMC_0D.git
cd DSMC_0D
```

2. Install dependencies:
```bash
pip install numpy scipy matplotlib pyyaml scikit-learn
```

## Quick Start

### Running a Simulation

The project includes pre-trained collision models, so you can run simulations immediately:

```bash
python run_simulation.py --config config/default.yaml
```

This will:
- Load pre-computed collision models from `models/`
- Initialize particles with Maxwell-Boltzmann distribution
- Run the DSMC simulation for all seeds specified in config
- Save results to `results/AR{AR}_COR{alpha}_R{seed}.txt`

### Analyzing Results

```bash
python run_postprocessing.py --config config/default.yaml
```

This generates temperature evolution plots in `figures/`.

## Running Custom Cases

### 1. Create a Run Directory

```bash
mkdir -p runs/my_case
```

### 2. Create a Custom Configuration

Copy and modify the default config:

```bash
cp config/default.yaml runs/my_case/config.yaml
```

Edit `runs/my_case/config.yaml`:

```yaml
particle:
  AR: 2.0              # Aspect ratio (1.1, 1.25, 1.5, 2.0, 2.5, 3.0 available)
  radius: 0.5          # Hemisphere radius
  mass: 1.0            # Particle mass

system:
  kTt: 1.0             # Initial translational temperature
  kTr: 1.0             # Initial rotational temperature
  alpha: 0.95          # Coefficient of restitution (0.60-1.00)
  phi: 0.01            # Volume fraction
  domain: [200, 200, 200]  # Domain size [Lx, Ly, Lz]

time:
  dt: 0.01             # Timestep for collision detection
  dtau: 0.1            # Output interval
  t_end: 200           # Total simulation time

simulation:
  seeds: [42, 123, 999]  # Random seeds for multiple realizations
  output_dir: runs/my_case/results/

preprocessing:
  gmm:
    gmm_cond_file: models/gmm_cond_AR20.npz  # Match your AR
  model_output_dir: models/

postprocessing:
  results_dir: runs/my_case/results/
  figures_dir: runs/my_case/figures/
```

### 3. Create Output Directories

```bash
mkdir -p runs/my_case/results runs/my_case/figures
```

### 4. Run the Simulation

```bash
python run_simulation.py --config runs/my_case/config.yaml
```

### 5. Post-process Results

```bash
python run_postprocessing.py --config runs/my_case/config.yaml
```

## Available Pre-trained Models

The repository includes pre-trained collision models for:

### Aspect Ratios
- AR = 1.1 (`gmm_cond_AR11.npz`)
- AR = 1.25 (`gmm_cond_AR125.npz`)
- AR = 1.5 (`gmm_cond_AR15.npz`)
- AR = 2.0 (`gmm_cond_AR20.npz`)
- AR = 2.5 (`gmm_cond_AR25.npz`)
- AR = 3.0 (`gmm_cond_AR30.npz`)

### Coefficients of Restitution
α ∈ {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00}

**Important**: Make sure to set `gmm_cond_file` in your config to match your chosen aspect ratio.

## Output Format

Results are saved as space-separated text files with columns:

```
t          tau        Tt         Tr         Teff
0.000000   0.000000   1.001760   0.999853   1.000997
0.700000   0.100062   1.000130   0.998702   0.999559
...
```

Where:
- `t`: Simulation time
- `tau`: Collision time (rescaled)
- `Tt`: Translational temperature (normalized by initial)
- `Tr`: Rotational temperature (normalized by initial)
- `Teff`: Effective temperature

## Project Structure

```
DSMC_0D/
├── config/
│   └── default.yaml              # Default configuration
├── src/
│   ├── preprocessing/            # Collision model fitting (optional)
│   ├── simulation/               # Core DSMC engine
│   └── postprocessing/           # Analysis and plotting
├── models/                       # Pre-trained collision models
│   ├── gmm_cond_AR*.npz         # Conditional GMM for energy redistribution
│   ├── scattering_coeffs.npz    # Scattering angle polynomials
│   ├── gamma_max_table.json     # Max dissipation lookup
│   └── one_hit_table.json       # Single-hit probability lookup
├── CTC_data/                     # DEM collision trajectory data (large)
├── run_preprocessing.py          # Fit models from CTC data (optional)
├── run_simulation.py             # Run DSMC simulation
├── run_postprocessing.py         # Analyze and plot results
└── README.md                     # This file
```

## Advanced: Re-training Collision Models

If you have new CTC data and want to re-train the collision models:

```bash
python run_preprocessing.py --config config/default.yaml
```

This will:
1. Train a Gaussian Mixture Model (GMM) for energy redistribution
2. Export conditional GMM parameters to `.npz` files
3. Fit polynomial models for scattering angle distribution
4. Build lookup tables for dissipation parameters

**Note**: This is computationally expensive (GMM training) and not required for running simulations with existing models.

## Performance

- **Conditional GMM sampling**: ~200 μs per sample (NumPy-only, no sklearn dependency)
- **Typical simulation**: ~350k collisions for 25k particles in ~10-30 seconds (domain-dependent)
- **Memory efficient**: Pre-computed matrices stored in compact `.npz` format

## Physical Results

Expected behavior for HCS (Homogeneous Cooling State):
- Smooth monotonic temperature decay
- Both translational and rotational temperatures cool together
- Temperature ratio `θ = Tt/Tr` evolves toward equilibrium
- Decay rate depends on coefficient of restitution `α`

Example for AR=2.0, α=0.95 over 100 time units:
- Translational temperature: 1.00 → 0.82 (18% decay)
- Rotational temperature: 1.00 → 0.83 (17% decay)

## Troubleshooting

### KeyError for (alpha, AR) pair
The lookup tables only contain pre-computed values for specific combinations. Check that your `alpha` and `AR` values are in the available sets listed above.

### Slow simulation
- Reduce domain size (fewer particles)
- Reduce `t_end`
- Increase `dt` (with caution - affects collision detection accuracy)
- Run fewer seeds initially for testing

### Out of memory
- Reduce domain size
- Use a smaller aspect ratio (fewer components in GMM)

## Citation

If you use this code in your research, please cite the relevant publications describing the collision models and DSMC methodology.

## License

[Specify your license here]

## Contact

For questions or issues, please open an issue on GitHub or contact the repository maintainers.
