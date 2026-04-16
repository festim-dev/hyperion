"""
Arrhenius plot of FLiBe permeability: fitted curves from inversion output
overlaid with literature data points.

Reads from results/inverted_points.csv and results/fitted_params.csv,
produced by invert_phi_flibe.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import morethemes as mt

mt.set_theme("lumen")

# ── Constants ─────────────────────────────────────────────────────────────────

kB_eV = 8.617333262e-5  # Boltzmann constant [eV/K]
KJ_MOL_TO_EV = 1.0 / 96.485
N_A = 6.02214076e23


def phi_arrhenius(T, phi0, E):
    return phi0 * np.exp(-E / (kB_eV * T))


def mol_to_particles(x):
    return x * N_A


def kjmol_to_ev(x):
    return x * KJ_MOL_TO_EV


def ev_to_kjmol(x):
    return x / KJ_MOL_TO_EV


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUTDIR = RESULTS
OUTDIR.mkdir(parents=True, exist_ok=True)

INV_CSV = RESULTS / "inverted_points.csv"
FIT_CSV = RESULTS / "fitted_params.csv"

# ── Case configuration ────────────────────────────────────────────────────────

CASES = {
    "swap_infinite": {"label": "Ideal coating"},
    "swap_transparent": {"label": "Uncoated"},
}

RUNS = ["Run 2", "Run 3"]

RUN_LABEL = {
    "Run 2": "H",
    "Run 3": "D",
}

ISOTOPE_COLOR = {
    "Run 2": "red",
    "Run 3": "black",
}

CASE_MARKERS = {
    "swap_infinite": "s",
    "swap_transparent": "^",
}

CASE_SYMBOL = {
    "swap_infinite": "□",
    "swap_transparent": "△",
}

LABEL_OFFSET = {
    ("swap_transparent", "Run 2"): (0.01, 1.25),
    ("swap_transparent", "Run 3"): (0.01, 0.75),
    ("swap_infinite", "Run 2"): (0.01, 0.80),
    ("swap_infinite", "Run 3"): (0.01, 1.25),
}

LABEL_ROTATION = {
    ("swap_transparent", "Run 2"): -7,
    ("swap_transparent", "Run 3"): -10,
    ("swap_infinite", "Run 2"): -4,
    ("swap_infinite", "Run 3"): -6,
}

# ── Literature data ───────────────────────────────────────────────────────────
# (name, phi_0 [mol-based], E [kJ/mol])

permeability_data_flibe = [
    ("Nakamura_H (2015)", 1.37e-5, 129.7),
    ("Calderoni_T (2008)", 7.34e-8, 77.6),
    ("Nishiumi_H (2016)", 3.76e-1, 194.0),
]
permeability_data_flibe_converted = [
    (name, mol_to_particles(phi0), kjmol_to_ev(E))
    for name, phi0, E in permeability_data_flibe
]

permeability_data_two_points = [
    ("Anderl_D (2004)", [873, 923], [1.49296e11, 1.806e11])  # T in K
]

# ── Plot ──────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 18,
    }
)

fig, ax = plt.subplots(figsize=(7.5, 5.5))

T_bg = np.linspace(773.15, 973.15, 400)
x_bg = 1000 / T_bg

LIT_COLORS = ["#4C72B0", "#DD8452", "#55A868"]
LIT_STYLES = ["--", "-.", ":"]

for i, (name, phi0, E) in enumerate(permeability_data_flibe_converted):
    ax.plot(
        x_bg,
        phi_arrhenius(T_bg, phi0, E),
        color=LIT_COLORS[i],
        linestyle=LIT_STYLES[i],
        lw=2,
        alpha=0.6,
        label=name,
        zorder=1,
    )

for name, T_points, phi_points in permeability_data_two_points:
    ax.plot(
        1000 / np.array(T_points),
        np.array(phi_points),
        color="purple",
        linestyle="--",
        lw=2,
        alpha=0.8,
        label=name,
        zorder=1,
    )

# Fitted curves and inverted points from inversion output
inv = pd.read_csv(INV_CSV)
fit = pd.read_csv(FIT_CSV)
inv.columns = inv.columns.str.strip()
fit.columns = fit.columns.str.strip()

for case_key, case in CASES.items():
    for run in RUNS:
        data = inv[(inv["case"] == case_key) & (inv["run"] == run)].sort_values("T_K")
        if data.empty:
            continue

        T = data["T_K"].values
        phi = data["phi"].values
        x = 1000 / T

        iso_color = ISOTOPE_COLOR[run]
        cond_marker = CASE_MARKERS[case_key]

        ax.plot(
            x,
            phi,
            linestyle="",
            marker=cond_marker,
            ms=6,
            mfc="white",
            mec=iso_color,
            mew=1.5,
            color=iso_color,
            zorder=4,
        )

        row = fit[(fit["case"] == case_key) & (fit["run"] == run)]
        if row.empty:
            continue

        phi0 = float(row["phi0"].iloc[0])
        E = float(row["E_eV"].iloc[0])
        print(
            f"P-{case['label']}-{RUN_LABEL[run]} ({CASE_SYMBOL[case_key]}) = "
            f"{phi0 / N_A:.2e} * exp(-{ev_to_kjmol(E):.2f}/(RT))"
        )

        T_fit = np.linspace(T.min(), T.max(), 300)
        phi_fit = phi_arrhenius(T_fit, phi0, E)
        ax.plot(1000 / T_fit, phi_fit, color=iso_color, lw=2, zorder=3)

        dx, mult = LABEL_OFFSET.get((case_key, run), (0.002, 1.0))
        rot = LABEL_ROTATION.get((case_key, run), 0)
        ax.text(
            1000 / T_fit[-1] + dx,
            phi_fit[-1] * mult,
            f"{case['label']}-{RUN_LABEL[run]}",
            fontsize=11,
            color=iso_color,
            rotation=rot,
            rotation_mode="anchor",
            va="center",
        )

ax.set_yscale("log")
ax.set_xlabel("1000 / T  [1/K]")
ax.set_ylabel(r"Permeability  [particle·m$^{-1}$·s$^{-1}$·Pa$^{-1}$]")
ax.legend(loc="upper right", frameon=True, fontsize=10)
ax.grid(True, axis="y", which="major", alpha=0.2)
ax.grid(False, axis="x")
ax.minorticks_off()

# Twin x-axis showing temperature in K
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
T_ticks = [773, 823, 873, 923, 973]
ax2.set_xticks([1000 / T for T in T_ticks])
ax2.set_xticklabels([str(T) for T in T_ticks])
ax2.set_xlabel("Temperature [K]", labelpad=10)
ax2.grid(True, axis="x", which="major", alpha=0.2)
ax2.spines["top"].set_visible(True)
ax2.spines["top"].set_linewidth(1.0)

out_path = OUTDIR / "fitted_phi_flibe.pdf"
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out_path}")
