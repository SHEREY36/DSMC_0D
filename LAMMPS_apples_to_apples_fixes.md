# LAMMPS → DSMC Apples-to-Apples Alignment: Required Fixes

**Prepared by:** DSMC_0D project analysis
**Date:** 2026-03-25
**For:** Claude instance working in `/home/muhammed/Documents/LAMMPS/src/GRANULAR/`

This document lists all identified discrepancies between the LAMMPS HCS (Homogeneous Cooling State) simulations and the 0D DSMC model. It describes exactly what changes to make in LAMMPS so that a direct, unambiguous comparison is possible. **Do not modify the DSMC code** — the goal is to bring LAMMPS into alignment with the DSMC model's physical assumptions.

---

## Summary of Discrepancies

| # | Discrepancy | Impact | Priority |
|---|-------------|--------|----------|
| 1 | V_rel includes rotational surface velocity in normal damping | Changes energy exchange rate, coupling T_rot to collision rate | **Critical** |
| 2 | Particle mass mismatch (`m_LAMMPS=1.309` vs `m_DSMC=1.0`) | 14.4% faster collision clock in DSMC | High |
| 3 | Collision counter factor of 2 | τ_DSMC = 2 × cpp_LAMMPS by definition | Medium (post-processing) |
| 4 | Initial condition modes (Mode B vs Mode C) | Different θ₀ and T_total₀ → different relaxation trajectories | Medium |

---

## Fix 1 (Critical): Normal Relative Velocity in `pair_gran_spherocyl_history.cpp`

### Problem

File: `/home/muhammed/Documents/LAMMPS/src/GRANULAR/pair_gran_spherocyl_history.cpp`

**Lines 502–504** compute the relative velocity at the contact point, including the rotational surface velocity:

```cpp
// CURRENT (WRONG for DSMC comparison):
vr1 = (v[i][0] + wxc_i[0]) - (v[j][0] + wxc_j[0]);
vr2 = (v[i][1] + wxc_i[1]) - (v[j][1] + wxc_j[1]);
vr3 = (v[i][2] + wxc_i[2]) - (v[j][2] + wxc_j[2]);
```

where `wxc_i = omega_i × rho_i` is the surface velocity due to rotation at the contact point (computed on lines 491–500).

This `vr` is then projected onto the contact normal to get `vnnr` (line 507), which drives the **normal damping force** on line 537:

```cpp
vnnr = vr1 * delx + vr2 * dely + vr3 * delz;  // line 507 — includes rotational contribution
...
damp = meff * gamman * vnnr * rsqinv;           // line 537
```

**Effect:** The normal restitution coefficient effectively couples to rotational velocity, creating an implicit T_trans/T_rot cross-coupling that is absent in the DSMC model. This changes the steady-state temperature ratio θ* and the apparent collision dissipation rate.

### Reference: MFIX Gets It Right

The equivalent code in MFIX (`/home/muhammed/Documents/Thesis/mfix-spherocylinder/model/des/cfrelvel.f`, line 106) explicitly comments out the rotational contribution to translational relative velocity:

```fortran
VRELTRANS(:) = VRELTRANS(:)! - V_ROT(:)
```

MFIX uses **only translational velocity** for the normal relative velocity — consistent with the DSMC NTC model where `vr = |v_i - v_j|` (centre-of-mass velocities only).

### The Fix

**Replace lines 502–504** with translational-only relative velocity for the normal component:

```cpp
// FIXED (translational-only, consistent with DSMC NTC and MFIX):
vr1 = v[i][0] - v[j][0];
vr2 = v[i][1] - v[j][1];
vr3 = v[i][2] - v[j][2];
```

Keep `wxc_i` and `wxc_j` intact — they are still needed for the **tangential** relative velocity (friction/shear). The tangential path (`vt1/vt2/vt3`, lines 513–515, and `vtr1/vtr2/vtr3`, lines 544–546) already adds the rotational contribution correctly for the shear history update. Only the **normal** component needs the translational-only version.

**Specifically:**
- Lines 502–504: change to translational only (as above)
- Lines 506–510 (`vnnr`, `vn1/vn2/vn3`): unchanged — they correctly use the now-fixed `vr`
- Lines 512–515 (`vt1/vt2/vt3`): these become `vr - vn`, which is now the translational tangential component
- Lines 544–546: if tangential shear history should still include rotational velocity, add `wxc` back here:
  ```cpp
  vtr1 = vt1 - (wxc_i[0] - wxc_j[0]) + ...; // see note below
  ```

> **Note on tangential treatment:** For a pure inelastic granular gas in the DSMC model, there is no tangential spring force (no shear history, no `kt`). The LAMMPS tangential channel is a physical enrichment. The critical fix for matching DSMC is solely the **normal channel** (lines 502–504). If you wish to keep the tangential channel physically correct (for its own purposes), re-add `wxc` only to the `vtr` variables used in the shear spring/damping, but NOT to the normal `vnnr`.

A clean implementation that fixes normal and preserves tangential correctly:

```cpp
// Step 1: translational-only relative velocity (for normal channel)
double vr1_tr = v[i][0] - v[j][0];
double vr2_tr = v[i][1] - v[j][1];
double vr3_tr = v[i][2] - v[j][2];

// Step 2: full contact-point velocity (for tangential channel)
vr1 = (v[i][0] + wxc_i[0]) - (v[j][0] + wxc_j[0]);
vr2 = (v[i][1] + wxc_i[1]) - (v[j][1] + wxc_j[1]);
vr3 = (v[i][2] + wxc_i[2]) - (v[j][2] + wxc_j[2]);

// Step 3: normal component uses translational-only
vnnr = vr1_tr * delx + vr2_tr * dely + vr3_tr * delz;
vn1 = delx * vnnr * rsqinv;
vn2 = dely * vnnr * rsqinv;
vn3 = delz * vnnr * rsqinv;

// Step 4: tangential uses full contact velocity
vt1 = vr1 - vn1;   // full vr minus translational normal
vt2 = vr2 - vn2;
vt3 = vr3 - vn3;
```

Then `damp = meff * gamman * vnnr * rsqinv` (line 537) uses only translational normal velocity — matching DSMC.

---

## Fix 2 (High): Particle Mass — Set `rho_p` in LAMMPS Input Scripts

### Problem

DSMC uses `mass = 1.0` for all particles (hardcoded in `config/default.yaml`). LAMMPS uses `rho_p = 1.0` (unit density), giving:

```
V_particle(AR=2, d=1) = π/6 · d³ · (AR - 1/3·AR·...)
                       = π · [R²(4/3·R + lcyl)]
                       = π · [0.25 · (4/3·0.5 + 1.0)]
                       ≈ 1.309  (exact: π/4 · 4/3 · 0.5 + π/4·1 = π·(1/6+1/4) ≈ 1.30900)
```

So `m_LAMMPS = rho_p × V_p = 1.0 × 1.309 = 1.309`, while `m_DSMC = 1.0`.

**Effect on collision frequency:**
The mean relative speed scales as `√(kT/m)`. With equal temperatures, DSMC particles are faster by:

```
sqrt(m_LAMMPS / m_DSMC) = sqrt(1.309 / 1.0) = 1.1441
```

→ DSMC collision clock runs **14.4% faster** than LAMMPS. This means all τ vs time curves from DSMC and LAMMPS will diverge even if everything else is matched.

### The Fix

In the LAMMPS input scripts (e.g. `in.hcs`), change the mass density so that `m = 1.0`:

```
# Exact formula: rho_p = 1 / V_p
# For AR=2, d=1 (R=0.5, lcyl=1.0):
#   V_p = (4/3)π R³ (two hemispheres) + π R² lcyl (cylinder)
#       = π [ (4/3)(0.5)³ + (0.5)²(1.0) ]
#       = π [ 1/6 + 1/4 ] = π · 5/12 ≈ 1.308997
#
# rho_p = 1 / 1.308997 ≈ 0.763944

density 0.763944
```

**Check:** with this density, `m = rho_p × V_p = 0.763944 × 1.308997 ≈ 1.000`. Collision frequency will then match DSMC up to the `σ_c` (cross-section) factor.

> **Inertia moment note:** The moment of inertia scales with mass, so `I_perp = m × (I_perp/m)`. The ratio `I_perp/m` is a geometric quantity (independent of density). At AR=2:
> - DSMC: `I_perp/m = 0.3025` (from `particle.py` formula)
> - LAMMPS: `I_perp/m = 0.395972 / 1.309 = 0.3025` (same ratio — correct)
>
> So fixing the mass automatically fixes the inertia moment. No separate inertia change needed.

---

## Fix 3 (Post-processing): Collision Counter Factor of 2

### Problem

DSMC increments `NColl += 2` for each collision event (one count per particle involved). Therefore:

```
τ_DSMC = NColl / Np = 2 × (number_of_unique_collision_events) / Np
       = 2 × cpp_LAMMPS
```

LAMMPS `collision_events.dat` counts each unique contact once (`events_cum_local` is incremented with `tag[i] < tag[j]` guard at line 551 of `pair_gran_spherocyl_history.cpp`):

```cpp
if (touch[jj] == 0 && tag[i] < tag[j]) events_new_local_step++;
```

So `cpp_LAMMPS = cumulative_events / N_part` = number of unique collision events per particle.

### The Fix Options

**Option A (Recommended):** Multiply LAMMPS cpp by 2 in post-processing:

```python
# In any comparison script:
lmp_tau = 2.0 * lmp_cpp   # now directly comparable to DSMC tau
```

**Option B:** Modify `collision_events.dat` output in LAMMPS to count both `i` and `j` (i.e., remove the `tag[i] < tag[j]` guard and divide by 2 in the output, or count both directions):

```cpp
// Remove the tag guard:
if (touch[jj] == 0) events_new_local_step++;
// Then cpp = cumulative_events / N_part will equal DSMC tau directly
```

> Option A is less invasive. The current `validate_C_alpha.py` in DSMC_0D uses option A implicitly by plotting DSMC `tau` vs LAMMPS `cpp` on separate twin axes and noting the factor-of-2 in alignment. If you want a single shared x-axis, use `2 × cpp_LAMMPS`.

---

## Fix 4 (Medium): Initial Conditions — Use Mode B Only

### Problem

Two LAMMPS datasets exist:
- **Mode B** (`modeB_e_sweep2a/`): `T_trans = T_rot = 1.0` at t=0 → `θ₀ ≈ 1.0`, `T_total₀ = 1.0`. This is the correct dataset for DSMC comparison.
- **Mode C / Nequal** (`LAMMPS_data/Nequal/`): After elastic equilibration, T_rot is rescaled to 0.1 → `θ₀ ≈ 10`, `T_total₀ = 0.64`. This is a strongly out-of-equilibrium initial condition not used in DSMC calibration.

The DSMC calibration (in `calibrate_C_alpha.py`) was performed against **Mode B** data. The parameter `C_alpha` in `models/C_alpha_table_AR20.json` was fitted to match θ* (asymptotic temperature ratio) between DSMC and LAMMPS Mode B.

### The Fix

Ensure all LAMMPS comparison runs use **Mode B initial conditions**:
- Input script key: `mode string B` (not `C`)
- `T_trans0 = T_rot0 = 1.0` (no rescaling of T_rot)
- This produces `θ₀ ≈ 1.0` at the start of cooling

If Mode C data is desired for separate validation, use it as a distinct dataset with its own DSMC runs starting from `T_trans=1, T_rot=0.1`.

---

## Verification Checklist After Applying Fixes

After making the changes above, run new LAMMPS HCS simulations and compare against DSMC using `validate_C_alpha.py`. Expected outcomes:

### Panel 1: Collision Frequency (τ vs physical time)
- After Fix 2 (mass), the DSMC and LAMMPS collision clocks should align within ~1–5% (residual from σ_c differences)
- After applying the factor-of-2 (Fix 3), `τ_DSMC(t)` should overlay `2×cpp_LAMMPS(t)` closely
- If they still diverge, compute the cross-section ratio:
  ```
  σ_c_DSMC / σ_c_LAMMPS = ?
  ```
  DSMC uses: `σ_c = (0.32·AR² + 0.694·AR - 0.0213) × π` from the fitted formula in `particle.py`
  LAMMPS uses: geometric projection of the spherocylinder shape (depends on orientation sampling)

### Panel 2: Temperature Ratio θ vs τ
- After Fix 1 (V_rel), the θ* values should converge with DSMC
- The shape of the θ relaxation curve (transient) should match
- If θ* still differs, this could indicate residual V_rot coupling elsewhere

### Panel 3: Cooling Rate T(t)/T(0) vs τ
- This should be largely unaffected by Fix 1 (the total dissipation rate is controlled by `gamman` and the **normal** restitution, not by f_tr or the T_rot coupling)
- After Fix 2 (mass), the curves should fall on top of each other when plotted vs τ (not vs physical time)
- Any remaining offset here indicates a σ_c mismatch

---

## Reference Values and Formulae

### Spherocylinder geometry (AR=2, d=1, R=0.5, l_cyl=1.0)

| Quantity | Formula | Value |
|---|---|---|
| Volume | `π[4/3 R³ + R² l_cyl]` | 1.308997 |
| I_perp/m | `(1/12)(3R² + l_cyl²) + R(3R+l_cyl)²/12/...` | 0.3025 |
| σ_c (DSMC) | `(0.32·AR² + 0.694·AR - 0.0213)·π` | ~4.5 |
| rho_p for m=1 | `1 / V_p` | **0.763944** |

### DSMC model equations (from `src/simulation/dsmc.py`)
```
f_tr(θ) = C(α) · 3θ / (3θ + 2)      # mode-neutral energy fraction
Zr(θ, η) = (0.39θ² + 0.09θ + 1.67) · η
P_r = min(1/Zr, 0.5)
gamma ~ Beta(1.211, 3.672) · gamma_max · prob_one_hit
τ = NColl / Np  (NColl += 2 per collision event)
```

### LAMMPS → DSMC correspondence
```
τ_DSMC = 2 × cpp_LAMMPS           (after Fix 3)
t_LAMMPS × sqrt(m_DSMC/m_LAMMPS) = t_DSMC_equivalent   (after Fix 2)
```

---

## Files to Modify in LAMMPS Project

| File | Change | Fix # |
|---|---|---|
| `src/GRANULAR/pair_gran_spherocyl_history.cpp` | Lines 502–504: translational-only V_rel for normal component | 1 |
| All `in.hcs` input scripts | `density 0.763944` (instead of `1.0`) | 2 |
| Post-processing scripts | Multiply cpp by 2 when comparing to DSMC τ | 3 |
| All `in.hcs` input scripts | Confirm `mode string B`, `T_trans0=T_rot0=1.0` | 4 |

---

## What NOT to Change in DSMC

The DSMC model is considered the reference. Do not change:
- `mass = 1.0` in `config/default.yaml`
- `NColl += 2` counting in `src/simulation/dsmc.py`
- The `f_tr(θ) = C(α)·3θ/(3θ+2)` formula
- The `C_alpha_table_AR20.json` calibration values (these were fitted to Mode B LAMMPS data; re-calibrate only after LAMMPS fixes are applied)

After all LAMMPS fixes are applied and new simulations run, the C(α) calibration should be **re-run** using `calibrate_C_alpha.py` with the corrected LAMMPS data as reference. The existing table was fitted against the buggy LAMMPS runs and will need updating.
