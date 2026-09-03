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
| `dry_run_sidewall.py` | Side-wall bypass of the empty cell; solves and plots (Fig. 12) |

**Outputs:** `results/dry_run_phi_arrhenius_fits.txt`, `dry_run_sidewall_metrics.csv`

---

### Step 2 — FLiBe permeability inversion
| Script | Purpose |
|---|---|
| `para_swap_pure.py` | Reads `dry_run_phi_arrhenius_fits.txt` for Ni parameters, then inverts each SWAP experimental point to find the FLiBe φ that matches the measured flux. |

**Output:** `results/fitted_params.csv`

---

### Step 3 — Sim vs exp comparison (run in any order, after Step 2)

| Script | Purpose |
|---|---|
| `comparison.py` | 2D FESTIM forward run for all cases using fitted φ, output the fluxes at all surface, compute the loss and contribution through the sidewall |
| `para_1d.py` | 1D equivalent of HYPERION; faster, used for comparison with 2D results |
| `identifiability.py` | Sweeps D and re-inverts φ, to show which parameters the steady flux can separate |
| `compare_bc_effect.py` | Writes the concentration fields for the two outer-wall limits |

**Outputs:** `results/master_summary.csv`, `jsim_jexp.csv`, `percentage_metrics.csv`, `surface_breakdown.csv`, `all_results_1d.csv`, `identifiability_*.csv`, `out-species_vol_*.bp`

---

### Step 4 — Plotting (run after Steps 2–3)

Plotting scripts live in `plotting/`. They resolve `results/` from the repository
root, so they can be run from any working directory.

| Script | Reads | Produces |
|---|---|---|
| `plotting/plot_framework_diagram.py` | — | Fig. 1, framework diagram |
| `plotting/plot_cell_schematic.py` | — | Fig. 2, cell and interface schematic |
| `plotting/plot_domain.py` | — | Fig. 3, computational domain |
| `plotting/plot_inversion_workflow.py` | — | Fig. 4, inversion flowchart |
| `plotting/plot_identifiability.py` | `identifiability_*.csv` | Fig. 5, recovered parameters |
| `plotting/plot_concentration_fields.py` | `out-species_vol_*.bp` | Fig. 6, concentration fields |
| `plotting/plot_comparison.py` | `jsim_jexp.csv`, `all_results_1d.csv` | Figs. 8, 13, 14, 1D vs 2D vs experiment |
| `plotting/plot_perm_fits_atom.py` | `fitted_params.csv`, `inverted_points.csv` | Fig. 9, Arrhenius plot with literature overlay |
| `plotting/plot_compare_sidewall.py` | `percentage_metrics.csv` | Figs. 10, 11, side-wall leakage and contribution |

The first four take no input and can be run at any time. Figs. 7 and 12 come
from `dry_run.py` and `dry_run_sidewall.py`, which solve as well as plot and so
stay with the simulation scripts in Step 1.

`plot_perm_fits.py`, still at the repository root, is the earlier version of the
Arrhenius figure and converts every literature source with N_A.
`plotting/plot_perm_fits_atom.py` supersedes it for Fig. 9: each source is
converted on its own mole basis, and each curve is clipped to the temperature
range it was measured over.

- **t3.py** — Transient simulation for FLiBe permeability with adaptive time-stepping enabled.  
- **t5.py** — Input file for the flow-direction swap test.
