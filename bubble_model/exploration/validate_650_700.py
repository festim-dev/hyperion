"""Validate the bubble model on the 650 / 700 C curves.

The FLiBe permeability / diffusivity Arrhenius coefficients are LOCKED to the
baseline obtained from the 500 / 550 / 600 C data (fit_baseline.py output).
Only bubble-model parameters are fit per temperature:
    alpha0    : initial bubble coverage
    alpha_max : asymptotic coverage when not shaking
    tau_grow  : growth time-constant (s)
and one shared schedule timing for 650 C (when shaking starts).
"""
from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import bubble_model_v2 as bm

DATA_DIR = Path(__file__).with_name("data")
BASELINE_FILE = Path(__file__).with_name("results_baseline") / "fitted_baseline.txt"
OUT_DIR = Path(__file__).with_name("results_validation")
OUT_DIR.mkdir(exist_ok=True)

P_UP = 1.32e5


# ---------------------------------------------------------------------------
# Load baseline FLiBe parameters
# ---------------------------------------------------------------------------
def read_baseline(path: Path) -> dict:
    out = {}
    pat = re.compile(r"^\s*([A-Za-z_0-9]+)\s*=\s*([-+0-9.eE]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out

base = read_baseline(BASELINE_FILE)
PHI_F0 = base["Phi_F0"]
E_PHIF = base["E_PhiF"]
D_F0 = base["D_F0"]
E_DF = base["E_DF"]
print("Loaded baseline:")
for k in ("Phi_F0", "E_PhiF", "D_F0", "E_DF"):
    print(f"  {k} = {base[k]:.4e}")


# ---------------------------------------------------------------------------
# Build a Materials object with the baseline locked in
# ---------------------------------------------------------------------------
def make_mat() -> bm.Materials:
    m = bm.Materials()
    m.Phi_F0 = PHI_F0
    m.E_PhiF = E_PHIF
    m.D_F0 = D_F0
    m.E_DF = E_DF
    return m


# ---------------------------------------------------------------------------
# Print predicted clean steady-state values
# ---------------------------------------------------------------------------
def J_ss_clean(T_C: float) -> float:
    mat = make_mat()
    bub = bm.BubbleParams(alpha0=0.0, alpha_max=0.0, tau_grow=1e30, tau_shake=1e30)
    # very short sim, we only need the analytic field
    res = bm.simulate(T_C=T_C, P_up=P_UP, t_end_s=60.0, mat=mat, bub=bub,
                      shake=lambda t: 0.0, n_x=20, dt_target=60.0)
    return res.J_ss_clean

print(f"\nClean (alpha=0) steady-state from baseline:")
print(f"  J_ss(650 C) = {J_ss_clean(650):.3e}  mol H2/m^2/s")
print(f"  J_ss(700 C) = {J_ss_clean(700):.3e}  mol H2/m^2/s")


# ---------------------------------------------------------------------------
# Load experimental data
# ---------------------------------------------------------------------------
def load(T_C: int):
    df = pd.read_csv(DATA_DIR / f"T{T_C}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()

t650, J650 = load(650)
t700, J700 = load(700)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    alpha0: float
    alpha_max: float
    tau_grow: float
    extras: dict


def fit_one(T_C: int, t_exp, J_exp, shake, x0, bounds):
    """x = [alpha0, alpha_max, tau_grow, alpha_residual]"""
    mat = make_mat()

    def loss(x):
        alpha0, alpha_max, tau_grow, alpha_res = x
        for v, (lo, hi) in zip(x, bounds):
            if not (lo <= v <= hi):
                return 1e6
        if alpha_res > alpha_max:
            return 1e6  # residual cannot exceed asymptote
        bub = bm.BubbleParams(alpha0=alpha0, alpha_max=alpha_max,
                              tau_grow=tau_grow, tau_shake=300.0,
                              alpha_residual=alpha_res)
        try:
            res = bm.simulate(T_C=T_C, P_up=P_UP, t_end_s=t_exp[-1] + 60,
                              mat=mat, bub=bub, shake=shake, n_x=40, dt_target=20.0)
        except Exception:
            return 1e6
        Jm = np.interp(t_exp, res.t, res.flux)
        scale = max(np.max(np.abs(J_exp)), 1e-12)
        w = np.where(t_exp / 3600 > 0.5, 1.0, 0.2)
        return float(np.sum(w * (Jm - J_exp) ** 2) / (scale ** 2) / len(t_exp))

    opt = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-6, maxiter=500))
    return opt.x, opt.fun


# 650 C: shake on at user-stated "after first plateau" -- we try a few start
# times in the 2.5 .. 3.5 h window and pick the best.
print("\nFitting 650 C ...")
best_650 = (None, 1e9, None)
for shake_on in [2.5, 2.8, 3.0, 3.2, 3.5]:
    shake = bm.make_shake_schedule([(shake_on, 6.0)])
    x_, fun = fit_one(
        650, t650, J650, shake,
        x0=np.array([0.30, 0.30, 4000.0, 0.15]),
        bounds=[(0.0, 0.95), (0.0, 0.6), (300, 5e4), (0.0, 0.4)],
    )
    print(f"  shake_on={shake_on:.1f} h  loss={fun:.4e}  "
          f"a0={x_[0]:.3f} amax={x_[1]:.3f} tg={x_[2]:.0f} a_res={x_[3]:.3f}")
    if fun < best_650[1]:
        best_650 = (x_, fun, shake_on)

x650, loss650, shake_on_650 = best_650
print(f"best: shake_on={shake_on_650} h  loss={loss650:.4e}")

print("\nFitting 700 C ...")
shake700 = bm.SHAKE_700
x700, loss700 = fit_one(
    700, t700, J700, shake700,
    x0=np.array([0.02, 0.35, 5000.0, 0.0]),
    bounds=[(0.0, 0.5), (0.0, 0.7), (300, 5e4), (0.0, 0.3)],
)
print(f"loss={loss700:.4e}  a0={x700[0]:.3f}  amax={x700[1]:.3f}  "
      f"tg={x700[2]:.0f}  a_res={x700[3]:.3f}")


# ---------------------------------------------------------------------------
# Run final simulations and plot
# ---------------------------------------------------------------------------
def simulate_final(T_C, t_end, x, shake):
    mat = make_mat()
    bub = bm.BubbleParams(alpha0=x[0], alpha_max=x[1], tau_grow=x[2],
                          tau_shake=300.0, alpha_residual=x[3])
    return bm.simulate(T_C=T_C, P_up=P_UP, t_end_s=t_end, mat=mat, bub=bub,
                       shake=shake, n_x=120, dt_target=10.0)

shake650 = bm.make_shake_schedule([(shake_on_650, 6.0)])
res650 = simulate_final(650, t650[-1] + 60, x650, shake650)
res700 = simulate_final(700, t700[-1] + 60, x700, bm.SHAKE_700)


fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex="col")

ax = axes[0, 0]
ax.plot(t650 / 3600, J650 * 1e6, "o", ms=3, color="#1f77b4", label="experiment")
ax.plot(res650.t / 3600, res650.flux * 1e6, "-", color="#d62728", lw=1.6, label="model")
ax.axhline(res650.J_ss_clean * 1e6, color="0.5", ls="--", lw=1,
           label=f"$J_{{ss}}^{{clean}}$ = {res650.J_ss_clean*1e6:.2f}")
ax.axvspan(shake_on_650, 6.0, color="orange", alpha=0.12, label="shaking")
ax.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
ax.set_title(f"650 °C    "
             f"$\\alpha_0$={x650[0]:.2f}, $\\alpha_\\max$={x650[1]:.2f}, "
             f"$\\alpha_{{res}}$={x650[3]:.2f}, $\\tau_g$={x650[2]:.0f} s")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 0]
ax.plot(res650.t / 3600, res650.alpha, color="#2ca02c", lw=1.6)
ax.axvspan(shake_on_650, 6.0, color="orange", alpha=0.12)
ax.set_xlabel("Time [h]")
ax.set_ylabel(r"$\alpha$ (bubble coverage)")
ax.set_ylim(-0.02, 1.0)
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.plot(t700 / 3600, J700 * 1e6, "o", ms=3, color="#1f77b4", label="experiment")
ax.plot(res700.t / 3600, res700.flux * 1e6, "-", color="#d62728", lw=1.6, label="model")
ax.axhline(res700.J_ss_clean * 1e6, color="0.5", ls="--", lw=1,
           label=f"$J_{{ss}}^{{clean}}$ = {res700.J_ss_clean*1e6:.2f}")
ax.axvspan(0.0, 0.08, color="orange", alpha=0.12, label="initial shake")
ax.set_title(f"700 °C    "
             f"$\\alpha_0$={x700[0]:.2f}, $\\alpha_\\max$={x700[1]:.2f}, "
             f"$\\alpha_{{res}}$={x700[3]:.2f}, $\\tau_g$={x700[2]:.0f} s")
ax.legend(loc="lower right", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1, 1]
ax.plot(res700.t / 3600, res700.alpha, color="#2ca02c", lw=1.6)
ax.axvspan(0.0, 0.08, color="orange", alpha=0.12)
ax.set_xlabel("Time [h]")
ax.set_ylim(-0.02, 1.0)
ax.grid(alpha=0.3)

fig.suptitle("Bubble model validation -- FLiBe parameters locked from 500/550/600 °C fit")
fig.tight_layout()
fig.savefig(OUT_DIR / "validation.png", dpi=180)
print("\nsaved", OUT_DIR / "validation.png")

# ---- write summary ----
with (OUT_DIR / "validation_summary.txt").open("w") as f:
    f.write("Baseline (locked from 500/550/600 C fit):\n")
    for k in ("Phi_F0", "E_PhiF", "D_F0", "E_DF"):
        f.write(f"  {k} = {base[k]:.6e}\n")
    f.write("\nFitted bubble params:\n")
    f.write(f"  650 C: alpha0={x650[0]:.4f}, alpha_max={x650[1]:.4f}, "
            f"alpha_residual={x650[3]:.4f}, tau_grow={x650[2]:.1f} s, "
            f"shake_on={shake_on_650:.2f} h, shake_off=6.00 h\n")
    f.write(f"  700 C: alpha0={x700[0]:.4f}, alpha_max={x700[1]:.4f}, "
            f"alpha_residual={x700[3]:.4f}, tau_grow={x700[2]:.1f} s, "
            f"initial-shake 0..0.08 h\n")
    f.write(f"\nClean steady-state:\n")
    f.write(f"  J_ss(650 C, alpha=0) = {res650.J_ss_clean:.4e}\n")
    f.write(f"  J_ss(700 C, alpha=0) = {res700.J_ss_clean:.4e}\n")
    f.write(f"\nFit losses: 650 = {loss650:.4e}, 700 = {loss700:.4e}\n")
print("saved", OUT_DIR / "validation_summary.txt")
