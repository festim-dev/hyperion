"""Model-free Arrhenius anchor for the high-temperature bubble-free flux.

The criticism: R_bulk at 650/700 C rests entirely on the FESTIM model
extrapolated from the 500-600 C calibration; the whole over-prediction is
then attributed to the interface.

The strongest check available inside this data set (no full bubble-removal
moment exists; max measured flux reaches only 77 % / 53 % of J0 at 650/700)
is a MODEL-FREE extrapolation: the three bubble-free steady fluxes
J_ss(500/550/600) follow an Arrhenius law (J_ss ~ Phi_F(T) x geometry, and
the geometry factor is T-independent).  Fitting ln J_ss vs 1/T on the three
bubble-free points and extrapolating to 650/700 C requires NO transport
model at all.  If the extrapolation lands near the FESTIM J0 and far above
the measured plateaus, the high-T flux deficit is real and cannot be an
artifact of the FESTIM extrapolation.

Uncertainty: 3-point fit -> report the least-squares line plus the spread
of the three leave-one-out pairwise slopes as an honest band.

Output: results_kint_inversion/anchor_extrapolation.png + printed table.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "results_kint_inversion"

# FESTIM bubble-free steady fluxes (J0_ss, from the inversion run)
J0_MODEL = {500: 0.795e-6, 550: 1.253e-6, 600: 1.867e-6,
            650: 2.641e-6, 700: 3.637e-6}


def steady_flux(T):
    d = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    t, J = d["time_s"].to_numpy(), d["flux_mol_m2_s"].to_numpy()
    return float(np.median(J[t > t[-1] - 1800]))


def main():
    T_cal = np.array([500, 550, 600])
    T_K = T_cal + 273.15
    J_ss = np.array([steady_flux(T) for T in T_cal])
    x = 1.0 / T_K
    y = np.log(J_ss)

    # least-squares Arrhenius line
    b, a = np.polyfit(x, y, 1)          # y = a + b*x  (b<0)
    # leave-one-out pairwise slopes (honest 3-point spread)
    pair_slopes = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        pair_slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    bs = np.array(sorted(pair_slopes))

    KB = 8.617333262e-5
    print(f"LSQ slope: {b:.0f} K  ->  E_app = {-b*KB:.3f} eV")
    print(f"pairwise slopes: {[f'{s:.0f}' for s in bs]}  "
          f"(E = {[f'{-s*KB:.2f}' for s in bs]} eV)")

    def extrap(slope, T):
        # anchor each line at the 600 C point
        return np.exp(y[2] + slope * (1 / (T + 273.15) - x[2]))

    print(f"\n{'T':>5} {'J_extrap (LSQ)':>15} {'band':>22} "
          f"{'J0 FESTIM':>10} {'J_meas':>18}")
    meas = {650: (1.6, 2.04), 700: (1.75, 1.91)}   # plateau range, max
    for T in (650, 700):
        je = extrap(b, T) * 1e6
        lo = min(extrap(s, T) for s in bs) * 1e6
        hi = max(extrap(s, T) for s in bs) * 1e6
        print(f"{T:>5} {je:>12.2f}e-6 [{lo:.2f}, {hi:.2f}]e-6"
              f" {J0_MODEL[T]*1e6:>9.2f}e-6"
              f"   plateau {meas[T][0]:.2f}, max {meas[T][1]:.2f} e-6")

    # figure
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    xx = np.linspace(1 / (730 + 273.15), 1 / (480 + 273.15), 200)
    ax.plot(xx * 1e3, np.exp(y[2] + b * (xx - x[2])) * 1e6, "-",
            color="#1f77b4", lw=1.8,
            label="Arrhenius extrapolation of bubble-free data")
    lo_line = [min(np.exp(y[2] + s * (xv - x[2])) for s in bs) * 1e6
               for xv in xx]
    hi_line = [max(np.exp(y[2] + s * (xv - x[2])) for s in bs) * 1e6
               for xv in xx]
    ax.fill_between(xx * 1e3, lo_line, hi_line, color="#1f77b4", alpha=0.15,
                    label="pairwise-slope band")
    ax.plot(1e3 / T_K, J_ss * 1e6, "o", ms=8, color="#1f77b4",
            label="measured steady flux, bubble-free (500-600 °C)")
    for T in (650, 700):
        xT = 1e3 / (T + 273.15)
        ax.plot(xT, J0_MODEL[T] * 1e6, "s", ms=9, mfc="none", mec="#d62728",
                mew=2, label="FESTIM bubble-free $J_0$" if T == 650 else None)
        ax.plot(xT, meas[T][0] * 1, "v", ms=9, color="0.25",
                label="measured plateau (bubble)" if T == 650 else None)
        ax.plot(xT, meas[T][1] * 1, "^", ms=8, color="0.55",
                label="measured max (during/after shake)" if T == 650 else None)
    for T in (500, 550, 600, 650, 700):
        ax.annotate(f"{T}°C", (1e3 / (T + 273.15), 0.55), fontsize=8,
                    ha="center", color="0.4")
    ax.set_yscale("log")
    ax.set_xlabel(r"$10^3/T$  [K$^{-1}$]")
    ax.set_ylabel(r"Steady flux [$10^{-6}$ mol H$_2$/m$^2$/s]")
    ax.set_title("Model-free Arrhenius anchor of the bubble-free flux")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "anchor_extrapolation.png", dpi=180)
    print(f"\nsaved {OUT_DIR / 'anchor_extrapolation.png'}")


if __name__ == "__main__":
    main()
