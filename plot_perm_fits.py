import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

kB_eV = 8.617333262e-5


def phi_arrhenius(T, phi0, E):
    return phi0 * np.exp(-E / (kB_eV * T))


ROOT = Path(__file__).resolve().parent

EXPORT_BASE = ROOT / "exports" / "figs" / "calibration_A" / "SWAP_bundle_pure"

CASES = {
    "swap_infinite": {
        "label": "Ideal coating",
        "dir": EXPORT_BASE / "swap_infinite_Ni_from_perm",
    },
    "swap_transparent": {
        "label": "No coating",
        "dir": EXPORT_BASE / "swap_transparent_Ni_from_perm",
    },
}

LABEL_OFFSET = {
    ("swap_transparent", "Run 2"): (0.01, 0.85),
    ("swap_transparent", "Run 3"): (0.01, 0.8),
    ("swap_infinite", "Run 2"): (0.01, 0.86),
    ("swap_infinite", "Run 3"): (0.01, 0.85),
}

LABEL_ROTATION = {
    ("swap_transparent", "Run 2"): -14,
    ("swap_transparent", "Run 3"): -18,
    ("swap_infinite", "Run 2"): -5,
    ("swap_infinite", "Run 3"): -12,
}

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

RUNS = ["Run 2", "Run 3"]


def angle_along_curve(ax, x, y, idx, k=8, max_angle=60):
    n = len(x)
    i0 = max(0, idx - k)
    i1 = min(n - 1, idx + k)
    if i1 == i0:
        return 0.0

    p0 = ax.transData.transform((x[i0], y[i0]))
    p1 = ax.transData.transform((x[i1], y[i1]))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))

    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180

    return float(np.clip(ang, -max_angle, max_angle))


OUT_DIR = ROOT / "plots"
OUT_DIR.mkdir(exist_ok=True)

# KJ_MOL_TO_EV = 1.0 / 96.485
# N_A = 6.02214076e23
# K_B_EV_PER_K = 8.617333262145e-5


# def mol_to_particles(x):
#     return x * N_A


# def kjmol_to_ev(x):
#     return x * KJ_MOL_TO_EV


# def ev_to_kjmol(x):
#     return x / KJ_MOL_TO_EV


# permeability_data = [
#     ("Lee", 4.52e-7, 55.3),
#     ("Gorman & Nardella", 4.65e-7, 55.2),
#     ("Ebisuzaki et al.", 4.05e-7, 55.1),  # 200 -420C
#     ("Robertson", 3.22e-7, 54.6),
#     ("Louthan et al.", 3.90e-7, 55.3),
#     ("Yaminish et al.", 7.08e-7, 54.8),
#     ("Masui et al.", 5.21e-7, 54.4),
#     (
#         "Altunoglu",
#         3.35e-7,
#         54.24,
#     ),  # this is good, but it seems that only work in 373K and 623K, which is outside our T range. So we will not use this for now.
#     (
#         "Masui",
#         4.87e-7,
#         54.0,
#     ),  # this is from the remi h dash data, which seems to have a different E than the original Masui paper?? unit converge issue??? double check
#     (
#         "Shiraishi et al.",
#         4.44e-7,
#         54.8,
#     ),
# ]

# permeability_data_converted = [
#     (name, mol_to_particles(phi0_mol), kjmol_to_ev(E_kJmol))
#     for name, phi0_mol, E_kJmol in permeability_data
# ]
# =========================================================
# Figure 1: Arrhenius plot with fitted curves + inline text
# =========================================================
plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 18,
    }
)
fig, ax = plt.subplots(figsize=(7.5, 5.5))

T_bg = np.linspace(750, 1000, 400)  # K range covering experiments
x_bg = 1000 / T_bg

# for name, phi0, E in permeability_data_converted:
#     phi_bg = phi_arrhenius(T_bg, phi0, E)
#     ax.plot(
#         x_bg,
#         phi_bg,
#         color="gray",
#         lw=1.2,
#         alpha=0.35,
#         zorder=1,
#     )

for case_key, case in CASES.items():
    inv = pd.read_csv(case["dir"] / "inverted_points.csv")
    fit = pd.read_csv(case["dir"] / "fitted_params.csv")

    inv.columns = inv.columns.str.strip()
    fit.columns = fit.columns.str.strip()

    for run in RUNS:
        data = inv[(inv["case"] == case_key) & (inv["run"] == run)].copy()
        if data.empty:
            continue

        data = data.sort_values("T_K")

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

        T_fit = np.linspace(T.min(), T.max(), 300)
        x_fit = 1000 / T_fit
        phi_fit = phi_arrhenius(T_fit, phi0, E)

        ax.plot(x_fit, phi_fit, color=iso_color, lw=2, zorder=3)

        label = f"{case['label']}-{RUN_LABEL[run]} ({CASE_SYMBOL[case_key]})"
        dx, mult = LABEL_OFFSET.get((case_key, run), (0.002, 1.0))
        rot = LABEL_ROTATION.get((case_key, run), 0)

        ax.text(
            x_fit[-1] + dx,
            phi_fit[-1] * mult,
            label,
            fontsize=11,
            color=iso_color,
            rotation=rot,
            rotation_mode="anchor",
            va="center",
        )

ax.set_yscale("log")
ax.set_xlabel("1000 / T  [1/K]")
ax.set_ylabel("Permeability  [particle·m⁻¹·s⁻¹·Pa⁻¹]")
ax.grid(True, alpha=0.3)

fig.savefig(
    OUT_DIR / "arrhenius_permeability_combined.svg",
    bbox_inches="tight",
)
plt.close(fig)
print("Saved:", OUT_DIR / "arrhenius_permeability_combined.svg")


# =========================================================
# Figure 2 + 3: linear y, x = T, one figure per run
# =========================================================
for run in RUNS:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "legend.fontsize": 18,
        }
    )
    fig_lin, ax_lin = plt.subplots(figsize=(7.5, 5.5))
    iso_color = ISOTOPE_COLOR[run]

    for case_key, case in CASES.items():
        inv = pd.read_csv(case["dir"] / "inverted_points.csv")
        inv.columns = inv.columns.str.strip()

        data = inv[(inv["case"] == case_key) & (inv["run"] == run)].copy()
        if data.empty:
            continue

        data = data.sort_values("T_K")
        T = data["T_K"].values
        phi = data["phi"].values

        cond_marker = CASE_MARKERS[case_key]

        ax_lin.plot(
            T,
            phi,
            linestyle="",
            marker=cond_marker,
            ms=12,
            mfc="white",
            mec=iso_color,
            mew=1.5,
            color=iso_color,
            label=case["label"],
        )

    ax_lin.set_xticks([773, 823, 873, 923, 973])
    ax_lin.set_xlim(768, 978)

    ax_lin.set_xlabel("T  [K]")
    ax_lin.set_ylabel("Permeability  [particle·m⁻¹·s⁻¹·Pa⁻¹]")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(frameon=True, fontsize=18)

    outname = f"{run.lower().replace(' ', '_')}_phi_linear_T_vs_phi.svg"
    fig_lin.savefig(
        OUT_DIR / outname,
        bbox_inches="tight",
    )
    plt.close(fig_lin)
    print("Saved:", OUT_DIR / outname)


# =========================================================
# Figure 4 + 5: log y, x = 1000/T, one figure per run
# =========================================================
for run in RUNS:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "legend.fontsize": 18,
        }
    )
    fig_log, ax_log = plt.subplots(figsize=(7.5, 5.5))
    iso_color = ISOTOPE_COLOR[run]

    for case_key, case in CASES.items():
        inv = pd.read_csv(case["dir"] / "inverted_points.csv")
        inv.columns = inv.columns.str.strip()

        data = inv[(inv["case"] == case_key) & (inv["run"] == run)].copy()
        if data.empty:
            continue

        data = data.sort_values("T_K")
        T = data["T_K"].values
        phi = data["phi"].values
        x = 1000 / T

        cond_marker = CASE_MARKERS[case_key]

        ax_log.plot(
            x,
            phi,
            linestyle="",
            marker=cond_marker,
            ms=12,
            mfc="white",
            mec=iso_color,
            mew=1.5,
            color=iso_color,
            label=case["label"],
        )

    ax_log.set_yscale("log")
    ax_log.set_xlabel("1000 / T  [1/K]")
    ax_log.set_ylabel("Permeability  [particle·m⁻¹·s⁻¹·Pa⁻¹]")
    ax_log.set_ylim(ymin=1e10, ymax=1e12)
    ax_log.grid(True, alpha=0.3)
    ax_log.legend(frameon=True, fontsize=18)

    outname = f"{run.lower().replace(' ', '_')}_phi_log_1000overT_vs_phi.svg"
    fig_log.savefig(
        OUT_DIR / outname,
        bbox_inches="tight",
    )
    plt.close(fig_log)
    print("Saved:", OUT_DIR / outname)
