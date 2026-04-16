# hyperion
## Pipeline overview

### Prerequisites
All scripts import shared data and constants from `exp_data.py`, which contains all the experimental data and necessary global variables.

---

### Step 1 — Dry-run (Ni permeability) baseline (run in any order)

| Script | Purpose |
|---|---|
| `dry_run.py` | Forward simulation of the dry-run cases; compares model flux against experimental measurements |
| `dry_run_fitting.py` | Recovers effective Ni permeability φ(T) from dry-run data via linear scaling; fits Arrhenius parameters |

**Output:** `results/dry_run_phi_arrhenius_fits.txt`

---

### Step 2 — FLiBe permeability inversion
Reads `dry_run_phi_arrhenius_fits.txt` for Ni parameters, then inverts
each SWAP experimental point to find the FLiBe φ that matches the measured flux.

**Output:** `results/fitted_params.csv`

---

### Step 3 — Sim vs exp comparison (run in any order, after Step 2)

| Script | Purpose |
|---|---|
| `comparison.py` | 2D FESTIM forward run for all cases using fitted φ, output the fluxes at all surface, compute the loss and contribution through the sidewall |
| `para_1d.py` | 1D equivalent of HYPERION; faster, used for comparison with 2D results |

**Outputs:** `results/master_summary.csv`, `jsim_jexp.csv`, `percentage_metrics.csv`, `surface_breakdown.csv`, `all_results_1d.csv`

---

### Step 4 — Plotting (run after Steps 2–3)

| Script | Reads | Produces |
|---|---|---|
| `plot_comparison.py` | `jsim_jexp.csv`, `all_results_1d.csv` | 1D vs 2D vs experiment panels |
| `plot_sidewall_metrics.py` | `percentage_metrics.csv` | Sidewall leakage / contribution figures |
| `plot_perm_fits.py` | `fitted_params.csv`, `inverted_points.csv` | Arrhenius plot with literature overlay |

- **t3.py** — Transient simulation for FLiBe permeability with adaptive time-stepping enabled.  
- **t5.py** — Input file for the flow-direction swap test.
