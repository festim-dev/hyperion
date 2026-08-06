"""Fit FLiBe permeability (Arrhenius) from the 500/550/600 C curves, assuming
no bubble effect.

We fit:
  Phi_F(T) = Phi_F0 * exp(-E_PhiF / (kB T))      [mol H2 / m / s / Pa]
  D_F(T)   = D_F0   * exp(-E_DF   / (kB T))      [m^2/s]
with Ni permeability fixed at the values in bubble_model_v2.Materials.

Four parameters: (Phi_F0, E_PhiF, D_F0, E_DF).

Each candidate set is evaluated by simulating each T in (500, 550, 600) C
with alpha == 0 (no bubble) for the same duration as the experimental data,
interpolating the model onto the experimental times, and summing the relative
squared error over all three curves.
"""

from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import bubble_model_v2 as bm

DATA_DIR = Path(__file__).with_name("data")
OUT_DIR = Path(__file__).with_name("results_baseline")
OUT_DIR.mkdir(exist_ok=True)

P_UP = 1.32e5
TS = (500, 550, 600)


def load(T_C: int):
    df = pd.read_csv(DATA_DIR / f"T{T_C}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


data = {T: load(T) for T in TS}


def run_no_bubble(T_C, t_end, Phi_F0, E_PhiF, D_F0, E_DF, n_x=40, dt=20.0):
    mat = bm.Materials()
    mat.Phi_F0 = Phi_F0
    mat.E_PhiF = E_PhiF
    mat.D_F0 = D_F0
    mat.E_DF = E_DF
    bub = bm.BubbleParams(alpha0=0.0, alpha_max=0.0, tau_grow=1e30, tau_shake=1e30)
    return bm.simulate(T_C=T_C, P_up=P_UP, t_end_s=t_end, mat=mat, bub=bub,
                       shake=lambda t: 0.0, n_x=n_x, dt_target=dt)


def loss(x):
    log_Phi_F0, E_PhiF, log_D_F0, E_DF = x
    Phi_F0 = 10 ** log_Phi_F0
    D_F0 = 10 ** log_D_F0
    if not (0.05 <= E_PhiF <= 1.5):
        return 1e6
    if not (0.05 <= E_DF <= 1.0):
        return 1e6
    if not (-15 <= log_Phi_F0 <= -8):
        return 1e6
    if not (-9 <= log_D_F0 <= -4):
        return 1e6

    err = 0.0
    for T in TS:
        t, J = data[T]
        try:
            res = run_no_bubble(T, t[-1] + 60, Phi_F0, E_PhiF, D_F0, E_DF)
        except Exception:
            return 1e6
        Jm = np.interp(t, res.t, res.flux)
        scale = max(np.max(np.abs(J)), 1e-12)
        # de-emphasize the very first points (initial measurement transient)
        w = np.where(t / 3600 > 0.3, 1.0, 0.3)
        err += np.sum(w * (Jm - J) ** 2) / (scale ** 2) / len(t)
    return err


# initial guess from current Materials defaults
mat0 = bm.Materials()
x0 = np.array([
    math.log10(mat0.Phi_F0),  # log_Phi_F0  ~ -10.5
    mat0.E_PhiF,              # 0.466
    math.log10(mat0.D_F0),    # log_D_F0    ~ -6.6
    mat0.E_DF,                # 0.24
])
print("initial loss:", loss(x0))

opt = minimize(loss, x0, method="Nelder-Mead",
               options=dict(xatol=1e-4, fatol=1e-6, maxiter=1500, disp=True))
xf = opt.x
print("final loss:", opt.fun)

Phi_F0 = 10 ** xf[0]
E_PhiF = xf[1]
D_F0 = 10 ** xf[2]
E_DF = xf[3]

print("\nFitted FLiBe parameters (baseline, no bubble):")
print(f"  Phi_F0 = {Phi_F0:.4e}  mol H2 / m / s / Pa")
print(f"  E_PhiF = {E_PhiF:.4f}  eV")
print(f"  D_F0   = {D_F0:.4e}  m^2/s")
print(f"  E_DF   = {E_DF:.4f}  eV")

# ---- plots ----
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
for ax, T in zip(axes, TS):
    t, J = data[T]
    res = run_no_bubble(T, t[-1] + 60, Phi_F0, E_PhiF, D_F0, E_DF, n_x=80)
    ax.plot(t / 3600, J * 1e6, "o", ms=3, color="#1f77b4", label="experiment")
    ax.plot(res.t / 3600, res.flux * 1e6, "-", color="#d62728", lw=1.6, label="model")
    ax.axhline(res.J_ss_clean * 1e6, color="0.5", ls="--", lw=1,
               label=f"J_ss = {res.J_ss_clean*1e6:.2f}")
    ax.set_title(f"{T} °C")
    ax.set_xlabel("Time [h]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
axes[0].set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
fig.suptitle("FLiBe baseline fit  (no bubble) -- 500/550/600 °C")
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_fit.png", dpi=180)
print("saved", OUT_DIR / "baseline_fit.png")

# ---- Arrhenius plot of fitted Phi_F (with the three single-T inversions) ----
T_K_arr = np.array([T + 273.15 for T in TS])
Phi_F_pred = np.array([Phi_F0 * math.exp(-E_PhiF / (bm.KB_EV * Tk)) for Tk in T_K_arr])

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.semilogy(1000 / T_K_arr, Phi_F_pred, "o-", color="#d62728",
             label=f"$\\Phi_F = {Phi_F0:.2e} \\exp(-{E_PhiF:.3f}/k_BT)$")
ax2.set_xlabel("1000 / T  [1/K]")
ax2.set_ylabel(r"$\Phi_{FLiBe}$  [mol H$_2$/m/s/Pa]")
ax2.set_title("FLiBe permeability (Arrhenius)")
ax2.grid(True, which="both", alpha=0.3)
ax2.legend(loc="upper right", fontsize=9)
# annotate temps
for Tk, ph in zip(T_K_arr, Phi_F_pred):
    ax2.annotate(f"{int(Tk-273.15)} °C", xy=(1000/Tk, ph), xytext=(2, 6),
                 textcoords="offset points", fontsize=8)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "phi_F_arrhenius.png", dpi=180)
print("saved", OUT_DIR / "phi_F_arrhenius.png")

# ---- save params for downstream use ----
with (OUT_DIR / "fitted_baseline.txt").open("w") as f:
    f.write(f"Phi_F0 = {Phi_F0:.6e}    # mol H2/m/s/Pa\n")
    f.write(f"E_PhiF = {E_PhiF:.6f}    # eV\n")
    f.write(f"D_F0   = {D_F0:.6e}    # m^2/s\n")
    f.write(f"E_DF   = {E_DF:.6f}    # eV\n")
    f.write(f"final_loss = {opt.fun:.4e}\n")
print("saved", OUT_DIR / "fitted_baseline.txt")
