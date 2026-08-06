"""Sensitivity of the k_int inversion to the bulk-transient mask.

The inversion formula R_int(t) = c_top (1/J_meas - 1/J0) is evaluated at
every time point; the mask threshold theta (valid where J0 > theta * J0_ss)
only decides which points we TRUST.  This script varies theta in
{0.5 ... 0.95} and reports, for 650 and 700 C:

  - the opening time of the valid window,
  - alpha_eff at the window opening,
  - the window medians (pre-shake / post-shake / plateau),
  - the implied bound on the 700 C formation time tau_g.

If the trusted plateau values are theta-independent and only the earliest
(transient-biased) points move, the conclusions do not rest on the mask
choice.  The quantitative bias of those early points is calibrated
separately in check_quasisteady_error.py.

Output: results_kint_inversion/mask_sensitivity.png + printed table.
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE_FILE = HERE / "results_baseline_2d" / "fitted_baseline.txt"
GRID_DIR = HERE / "results_2d_alpha_grid"
KINT_DIR = HERE / "results_kint_inversion"

KB_EV = 8.617333262e-5
N_A = 6.02214076e23
P_UP = 1.32e5
THETAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
SHAKE_H = {650: [(3.0, 6.0)], 700: [(0.0, 0.08)]}


def read_kv(p):
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([\w_]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def c_top_molH2(T_C, base):
    T_K = T_C + 273.15
    K_H = (base["Phi_F0_atoms"] / base["D_F0"]) * np.exp(
        -(base["E_PhiF"] - base["E_DF"]) / (KB_EV * T_K))
    return K_H * P_UP / (2.0 * N_A)


def main():
    base = read_kv(BASE_FILE)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    for ax, T in zip(axes, (650, 700)):
        df = pd.read_csv(KINT_DIR / f"kint_{T}C.csv")
        t = df["t_s"].to_numpy()
        R_int = df["R_int_s_per_m"].to_numpy()
        J0 = df["J0_model"].to_numpy()
        th = t / 3600.0

        z = np.load(GRID_DIR / f"grid_{T}C.npz")
        alphas_g, J_g = z["alphas"], z["J"][:, -1]
        c_top = c_top_molH2(T, base)
        R_g = c_top * (1.0 / J_g - 1.0 / J_g[0])
        a_eff = np.interp(np.clip(R_int, 0.0, R_g[-1]), R_g, alphas_g)

        print(f"\n=== {T} °C ===")
        print(f"{'theta':>6} {'t_open [h]':>11} {'a_eff(open)':>12} "
              f"{'median window A':>16} {'median window B':>16}")
        cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(THETAS)))
        for c, theta in zip(cmap, THETAS):
            valid = J0 > theta * J0[-1]
            i0 = int(np.argmax(valid))
            # window A: pre-shake (650) / 1-3 h (700); window B: post-shake / late
            if T == 650:
                wA = (th >= th[i0]) & (th < 3.0) & valid
                wB = (th >= 6.0) & valid
            else:
                wA = (th >= 1.0) & (th < 3.0) & valid
                wB = (th >= 3.0) & valid
            mA = np.median(a_eff[wA]) if wA.any() else np.nan
            mB = np.median(a_eff[wB]) if wB.any() else np.nan
            print(f"{theta:>6.2f} {th[i0]:>11.2f} {a_eff[i0]:>12.3f} "
                  f"{mA:>16.3f} {mB:>16.3f}")
            sel = valid.copy()
            ax.plot(th[sel], a_eff[sel], ".", ms=2.6, color=c,
                    label=fr"$\theta$ = {theta:.2f}" if T == 650 else None)
        for a, b in SHAKE_H[T]:
            ax.axvspan(a, b, color="orange", alpha=0.12)
        ax.axhline(0.26, color="0.4", ls=":", lw=1)
        ax.set_xlabel("Time [h]")
        ax.set_title(f"{T} °C")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel(r"$\alpha_{\rm eff}$")
    axes[0].set_ylim(0, 0.75)
    axes[0].legend(fontsize=8, ncol=2, title="mask threshold")
    fig.suptitle("Mask-threshold sensitivity of the inverted coverage")
    fig.tight_layout()
    fig.savefig(KINT_DIR / "mask_sensitivity.png", dpi=180)
    print(f"\nsaved {KINT_DIR / 'mask_sensitivity.png'}")


if __name__ == "__main__":
    main()
