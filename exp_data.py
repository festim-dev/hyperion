# Central store for all experimental data and material constants.
# All flux_err values are k=2 (95% CI); divide by 2 for 1-sigma before use.

from pathlib import Path
import re
import h_transport_materials as htm

# ── Shared Ni diffusivity ─────────────────────────────────────────────────────
# Used in both dry_run_fitting.py and para_swap_pure.py to ensure S = phi / D
# is computed consistently with the same D_nickel everywhere.

D_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")[-1]
D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

# ── Dry-run downstream flux data ──────────────────────────────────────────────
# Columns: (T [C], run label, P_up [Pa], P_down [Pa], flux [H/s], flux_err [H/s])

dry_run = [
    (500.0, "Run 1", 1.30e5, 1.98e2, 4.52e16, 3.06e15),
    (500.0, "Run 2", 1.10e5, 1.79e2, 4.08e16, 2.76e15),
    (600.0, "Run 1", 1.30e5, 4.59e2, 1.03e17, 6.97e15),
    (600.0, "Run 2", 1.10e5, 4.02e2, 9.19e16, 6.22e15),
    (700.0, "Run 1", 1.30e5, 8.16e2, 1.85e17, 1.25e16),
    (700.0, "Run 2", 1.10e5, 7.36e2, 1.65e17, 1.12e16),
]

# ── Normal experimental flux data ────────────────────────────────────────────
# normal_conditions holds shared pressure and flux values for both BC cases.
# normal_transparent adds a per-run P_gb (outer surface H2 pressure [Pa]);
# normal_infinite has no outer pressure BC.
# Columns per run: {P_up [Pa], P_down [Pa], J_exp [H/s]}

normal_conditions = {
    500.0: {
        "runs": {
            "Run 1": {"P_up": 1.11e5, "P_down": 4.55, "J_exp": 1.04e15},
            "Run 2": {"P_up": 1.05e5, "P_down": 4.84, "J_exp": 1.09e15},
        }
    },
    550.0: {
        "runs": {
            "Run 1": {"P_up": 1.10e5, "P_down": 7.10, "J_exp": 1.52e15},
            "Run 2": {"P_up": 1.05e5, "P_down": 7.80, "J_exp": 1.69e15},
        }
    },
    600.0: {
        "runs": {
            "Run 1": {"P_up": 1.05e5, "P_down": 9.36, "J_exp": 2.01e15},
            "Run 2": {"P_up": 1.31e5, "P_down": 1.07e1, "J_exp": 2.38e15},
        }
    },
    650.0: {
        "runs": {
            "Run 1": {"P_up": 1.05e5, "P_down": 1.51e1, "J_exp": 3.30e15},
            "Run 2": {"P_up": 1.04e5, "P_down": 1.40e1, "J_exp": 3.00e15},
        }
    },
    700.0: {
        "runs": {
            "Run 1": {"P_up": 1.03e5, "P_down": 1.97e1, "J_exp": 4.34e15},
            "Run 2": {"P_up": 1.02e5, "P_down": 2.04e1, "J_exp": 4.38e15},
        }
    },
}

# P_gb differs per run and temperature for normal_transparent.
# 550 Run 2 has a slightly different P_down (7.08 vs 7.80) — stored here as an override.
_normal_pgb = {
    500.0: {"Run 1": 5.0, "Run 2": 3.0},
    550.0: {"Run 1": 5.0, "Run 2": 5.0},
    600.0: {"Run 1": 5.0, "Run 2": 5.0},
    650.0: {"Run 1": 5.0, "Run 2": 5.0},
    700.0: {"Run 1": 7.0, "Run 2": 7.0},
}

# P_down for 550 Run 2 differs slightly between the two cases (measurement variation).
_normal_transparent_pdown_override = {550.0: {"Run 2": 7.08}}

normal_infinite = {
    Tc: {"runs": {run: dict(cond) for run, cond in block["runs"].items()}}
    for Tc, block in normal_conditions.items()
}

normal_transparent = {
    Tc: {
        "runs": {
            run: {
                **cond,
                "P_down": _normal_transparent_pdown_override.get(Tc, {}).get(
                    run, cond["P_down"]
                ),
                "P_gb": _normal_pgb[Tc][run],
            }
            for run, cond in block["runs"].items()
        }
    }
    for Tc, block in normal_conditions.items()
}

# Normal flux uncertainties (k=2, 95% CI). Both cases share the same measurements.
_normal_flux_err_base = {
    500.0: {"runs": {"Run 1": 2.72e13, "Run 2": 4.49e13}},
    550.0: {"runs": {"Run 1": 4.62e13, "Run 2": 5.50e13}},
    600.0: {"runs": {"Run 1": 5.04e13, "Run 2": 7.63e13}},
    650.0: {"runs": {"Run 1": 7.87e13, "Run 2": 6.96e13}},
    700.0: {"runs": {"Run 1": 9.65e13, "Run 2": 9.34e13}},
}

normal_flux_err = {
    "normal_infinite": _normal_flux_err_base,
    "normal_transparent": _normal_flux_err_base,
}

# ── SWAP experimental flux data ───────────────────────────────────────────────
# swap_conditions holds the shared pressure and flux values for both BC cases.
# swap_transparent adds P_gb per entry; swap_infinite has no outer pressure.
# Run 3 entries are D2 measurements, kept for reference but excluded from H fits.
# Columns per run: {P_up [Pa], P_down [Pa], J_exp [H/s]}

swap_conditions = {
    500.0: {
        "runs": {
            "Run 1": {"P_up": 1.31e5, "P_down": 1.77e1, "J_exp": 3.89e15},
            "Run 2": {"P_up": 1.31e5, "P_down": 1.99e1, "J_exp": 4.34e15},
            "Run 3": {"P_up": 1.31e5, "P_down": 8.66, "J_exp": 1.91e15},  # D2
        }
    },
    550.0: {
        "runs": {
            "Run 1": {"P_up": 1.31e5, "P_down": 3.21e1, "J_exp": 7.20e15},
            "Run 2": {"P_up": 1.31e5, "P_down": 3.89e1, "J_exp": 8.58e15},
        }
    },
    600.0: {
        "runs": {
            "Run 1": {"P_up": 1.33e5, "P_down": 3.57e1, "J_exp": 7.64e15},
            "Run 2": {"P_up": 1.32e5, "P_down": 4.62e1, "J_exp": 1.01e16},
            "Run 3": {"P_up": 1.33e5, "P_down": 2.10e1, "J_exp": 4.50e15},  # D2
        }
    },
    650.0: {
        "runs": {
            "Run 2": {"P_up": 1.32e5, "P_down": 5.02e1, "J_exp": 1.10e16},
        }
    },
    700.0: {
        "runs": {
            "Run 1": {"P_up": 1.32e5, "P_down": 4.07e1, "J_exp": 9.04e15},
            "Run 2": {"P_up": 1.32e5, "P_down": 4.78e1, "J_exp": 1.04e16},
            "Run 3": {"P_up": 1.31e5, "P_down": 3.23e1, "J_exp": 7.12e15},  # D2
        }
    },
}

# Build per-case dicts by injecting P_gb only for swap_transparent.
# Both share the same J_exp / pressure values from swap_conditions.
P_GB_TRANSPARENT = 1e-30  # outer surface H2 partial pressure forced to ~0 [Pa]

swap_infinite = {
    Tc: {"runs": {run: dict(cond) for run, cond in block["runs"].items()}}
    for Tc, block in swap_conditions.items()
}

swap_transparent = {
    Tc: {
        "runs": {
            run: {**cond, "P_gb": P_GB_TRANSPARENT}
            for run, cond in block["runs"].items()
        }
    }
    for Tc, block in swap_conditions.items()
}

# ── SWAP flux uncertainties (k=2, 95% CI) ────────────────────────────────────
# Both BC cases share the same raw measurement uncertainties.
# Nested as {T_C: {runs: {run_label: flux_err [H/s]}}}.

_swap_flux_err_base = {
    500.0: {"runs": {"Run 1": 2.66e14, "Run 2": 2.96e14, "Run 3": 1.33e14}},
    550.0: {"runs": {"Run 1": 4.88e14, "Run 2": 5.81e14}},
    600.0: {"runs": {"Run 1": 5.25e14, "Run 2": 6.84e14, "Run 3": 3.08e14}},
    650.0: {"runs": {"Run 2": 7.43e14}},
    700.0: {"runs": {"Run 1": 6.17e14, "Run 2": 7.08e14, "Run 3": 4.84e14}},
}

swap_flux_err = {
    "swap_infinite": _swap_flux_err_base,
    "swap_transparent": _swap_flux_err_base,
}

# ── Ni permeability from dry-run Arrhenius fits ───────────────────────────────
# Loaded at import time from results/dry_run_phi_arrhenius_fits.txt, which is
# produced by dry_run_fitting.py. The mapping from BC mode to surface condition is:
#   flux0  ->  particle_flux_zero  (ideal coating, zero outer flux)
#   conc0  ->  sieverts            (uncoated, zero outer concentration)

_MODE_TO_BC = {
    "flux0": "particle_flux_zero",
    "conc0": "sieverts",
}

_FLIBE_FITS_FILE = Path("results") / "fitted_params.csv"


def load_flibe_permeability(
    fits_file: Path = _FLIBE_FITS_FILE,
) -> dict:
    """
    Parse fitted_params.csv produced by para_swap_pure.py and return FLiBe
    permeability objects keyed by (case, run).

    Returns
    -------
    dict
        {(case_name, run_name): htm.Permeability}
    """
    import csv as _csv

    if not fits_file.exists():
        raise FileNotFoundError(
            f"FLiBe permeability file not found: {fits_file}\n"
            "Run para_swap_pure.py first to generate it."
        )

    result = {}
    with open(fits_file, newline="") as f:
        for row in _csv.DictReader(f):
            key = (row["case"].strip(), row["run"].strip())
            result[key] = htm.Permeability(
                pre_exp=float(row["phi0"]),
                act_energy=float(row["E_eV"]),
                law="henry",
            )
    return result


def load_ni_permeability(
    fits_file: Path = Path("results") / "dry_run_phi_arrhenius_fits.txt",
    run_id: str = "Run 1",
) -> dict:
    """
    Parse dry_run_phi_arrhenius_fits.txt and return Ni permeability parameters
    keyed by outer BC type.

    Each line in the file has the form:
        <mode> | <run>: phi_0=<value>, E_phi_kJmol=<value>

    Parameters
    ----------
    fits_file : Path
        Path to the Arrhenius fits text file produced by dry_run_fitting.py.
    run_id : str
        Which run's fit to use (default 'Run 1').

    Returns
    -------
    dict
        {"particle_flux_zero": {"phi_0": ..., "E_phi_kJmol": ...},
         "sieverts":           {"phi_0": ..., "E_phi_kJmol": ...}}
    """
    if not fits_file.exists():
        raise FileNotFoundError(
            f"Ni permeability file not found: {fits_file}\n"
            "Run dry_run_fitting.py first to generate it."
        )

    pattern = re.compile(
        r"^(?P<mode>\w+)\s*\|\s*(?P<run>[^:]+):\s*"
        r"phi_0=(?P<phi_0>[0-9eE+\-.]+),\s*E_phi_kJmol=(?P<E_phi>[0-9eE+\-.]+)"
    )

    result = {}
    with open(fits_file) as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            if m.group("run").strip() != run_id:
                continue
            mode = m.group("mode").strip()
            bc_type = _MODE_TO_BC.get(mode)
            if bc_type is None:
                continue
            result[bc_type] = {
                "phi_0": float(m.group("phi_0")),
                "E_phi_kJmol": float(m.group("E_phi")),
            }

    missing = [bc for bc in _MODE_TO_BC.values() if bc not in result]
    if missing:
        raise ValueError(
            f"Could not find entries for BC types {missing} with run='{run_id}' "
            f"in {fits_file}."
        )

    return result
