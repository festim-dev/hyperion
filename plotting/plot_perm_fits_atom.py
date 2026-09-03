"""
FLiBe permeability from the inversion, against the literature on an atom basis.

Each literature source is converted on its own mole basis rather than
uniformly with N_A, and each curve is clipped to the temperature range its
source measured.

Mole basis of each source:

  Nakamura (2015)   reports P_H2 in [mol/(m s Pa)] with the flux linear in
                    pressure and sorption obeying Henry's law, so a mole is a
                    mole of H2 and the conversion is 2*N_A.
  Nishiumi (2016)   same group and apparatus, same molecular basis; the values
                    used are the FLiBe set, not the Flinabe or Fnabe sets the
                    paper also reports.
  Calderoni (2008)  reports a flux proportional to sqrt(p_T2), carried by "the
                    molar flux of T", with tritium diffusing "in the atomic
                    form".  A mole is a mole of atoms, so N_A, and the
                    permeability carries Pa^-1/2 rather than Pa^-1.  Eq. (6) of
                    that paper prints [mol/m3 Pa], which the sqrt dependence
                    cannot give; h_transport_materials records the printed unit
                    and flags the inconsistency.  N_A and Pa^-1/2 are used here
                    because they are what the measurement supports.
  Anderl (2004)     two measured points, already atom-based in the source.

Measured ranges: Nakamura 500-600 C (stated twice in the JSPF paper); Nishiumi
not stated, taken as Nakamura's; Calderoni 550-700 C, the range its apparatus
and diffusivity data give, though its abstract says 500-700.

The palette is plot_perm_fits.py's, so the two figures can be read side by
side.  Its third literature colour #55A868 is close to #DD8452 under
protanopia, so those two series are told apart by dash pattern rather than hue.

Outputs (saved to results/):
    fitted_phi_flibe_atom.pdf/.png  -- Arrhenius plot with literature overlay
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

N_A = 6.02214076e23
KJ_MOL_TO_EV = 1.0 / 96.485332123
K_B = 8.617333262145e-5

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INV_CSV = RESULTS / "inverted_points.csv"
FIT_CSV = RESULTS / "fitted_params.csv"

# (name, phi_0 as printed in the source, E [kJ/mol], basis, colour, style,
#  measured range [K])
# basis: "H2" -> a mole is a mole of molecules, so 2*N_A;  "H" -> N_A.
# No source covers the 773-973 K the inversion spans, so each curve is clipped
# to its own range; extrapolating them to a common temperature would compare
# extrapolations rather than measurements.
LIT = [
    ("Nakamura (2015), H", 1.37e-5, 129.7, "H2", "#4C72B0", "--", (773.15, 873.15)),
    ("Calderoni (2008), T", 7.34e-8, 77.6, "H", "#DD8452", "-.", (823.15, 973.15)),
    ("Nishiumi (2016), H", 3.76e-1, 194.0, "H2", "#55A868", ":", (773.15, 873.15)),
]
BASIS = {"H2": 2.0 * N_A, "H": N_A}
ANDERL = ("Anderl (2004), D", [873.0, 923.0], [1.49296e11, 1.806e11], "purple")

CASES = {"swap_infinite": {"label": "Ideal coating", "marker": "s", "sym": "□"},
         "swap_transparent": {"label": "Uncoated", "marker": "^", "sym": "△"}}
RUNS = ["Run 2", "Run 3"]
RUN_LABEL = {"Run 2": "H", "Run 3": "D"}
ISOTOPE_COLOR = {"Run 2": "red", "Run 3": "black"}
LABEL_OFFSET = {("swap_transparent", "Run 2"): (0.01, 1.25),
                ("swap_transparent", "Run 3"): (0.01, 0.75),
                ("swap_infinite", "Run 2"): (0.01, 0.80),
                ("swap_infinite", "Run 3"): (0.01, 1.25)}
LABEL_ROTATION = {("swap_transparent", "Run 2"): -7,
                  ("swap_transparent", "Run 3"): -10,
                  ("swap_infinite", "Run 2"): -4,
                  ("swap_infinite", "Run 3"): -6}


def phi_arrhenius(T, phi0, E_eV):
    return phi0 * np.exp(-E_eV / (K_B * np.asarray(T, float)))


def main() -> None:
    plt.rcParams.update({"font.size": 18, "axes.labelsize": 18,
                         "axes.titlesize": 18, "legend.fontsize": 10})
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    T_bg = np.linspace(773.15, 973.15, 400)
    x_bg = 1000 / T_bg

    print(f"{'source':>34} {'basis':>6} {'Phi_0 [H/m/s/Pa]':>18} {'E [eV]':>8}")
    for name, phi0, E_kj, basis, color, ls, rng in LIT:
        p0, E = phi0 * BASIS[basis], E_kj * KJ_MOL_TO_EV
        print(f"{name:>34} {basis:>6} {p0:>18.4e} {E:>8.4f}"
              f"   {rng[0] - 273.15:.0f}-{rng[1] - 273.15:.0f} C")
        T_src = np.linspace(*rng, 200)
        ax.plot(1000 / T_src, phi_arrhenius(T_src, p0, E), color=color,
                linestyle=ls, lw=2, alpha=0.6, label=name, zorder=1)

    name, Ts, phis, color = ANDERL
    ax.plot(1000 / np.array(Ts), np.array(phis), color=color, linestyle="--",
            lw=2, alpha=0.8, label=name, zorder=1)

    inv = pd.read_csv(INV_CSV)
    fit = pd.read_csv(FIT_CSV)
    inv.columns = inv.columns.str.strip()
    fit.columns = fit.columns.str.strip()

    for case_key, case in CASES.items():
        for run in RUNS:
            d = inv[(inv["case"] == case_key) & (inv["run"] == run)].sort_values("T_K")
            row = fit[(fit["case"] == case_key) & (fit["run"] == run)]
            if d.empty or row.empty:
                continue
            c = ISOTOPE_COLOR[run]
            ax.plot(1000 / d["T_K"].values, d["phi"].values, linestyle="",
                    marker=case["marker"], ms=6, mfc="white", mec=c, mew=1.5,
                    zorder=4)
            phi0, E = float(row["phi0"].iloc[0]), float(row["E_eV"].iloc[0])
            T_fit = np.linspace(d["T_K"].min(), d["T_K"].max(), 300)
            y = phi_arrhenius(T_fit, phi0, E)
            ax.plot(1000 / T_fit, y, color=c, lw=2, zorder=3)
            dx, mult = LABEL_OFFSET.get((case_key, run), (0.002, 1.0))
            ax.text(1000 / T_fit[-1] + dx, y[-1] * mult,
                    f"{case['label']}-{RUN_LABEL[run]}", fontsize=11, color=c,
                    rotation=LABEL_ROTATION.get((case_key, run), 0),
                    rotation_mode="anchor", va="center")

    ax.set_yscale("log")
    ax.set_xlabel("1000 / T  [1/K]")
    ax.set_ylabel(r"Permeability  [atom(H or D)$\cdot$m$^{-1}\cdot$s$^{-1}\cdot$Pa$^{-1}$]")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True, axis="y", which="major", alpha=0.2)
    ax.grid(False, axis="x")
    ax.minorticks_off()

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    T_ticks = [773, 823, 873, 923, 973]
    ax2.set_xticks([1000 / T for T in T_ticks])
    ax2.set_xticklabels([str(T) for T in T_ticks])
    ax2.set_xlabel("Temperature [K]", labelpad=10)
    ax2.grid(True, axis="x", which="major", alpha=0.2)
    ax2.spines["top"].set_visible(True)
    ax2.spines["top"].set_linewidth(1.0)

    stem = RESULTS / "fitted_phi_flibe_atom"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"\n[saved] {stem}.pdf (+ .png)")


if __name__ == "__main__":
    main()
