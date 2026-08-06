"""Arrhenius-constrained reduced-order bubble-coverage model.

Replaces the per-temperature ODE fit (fit_dynamic_alpha.py) with a single
global parameter set that is shared across 650 °C and 700 °C.  The
temperature dependence of the bubble dynamics is forced through Arrhenius
forms — i.e. the two temperatures are described by the *same* physical
mechanism, only with the rates re-evaluated at each T.

Model
-----
    α_max(T)    = α_max,∞ · exp(−E_max / (k_B T))    [equilibrium coverage]
    τ_grow(T)  = τ_g,0   · exp(+E_τ   / (k_B T))    [growth time constant]
    α_res       (T-independent — surface-attachment property of Ni)
    τ_shake    = 300 s (fixed)

Physical expectations
---------------------
- α_max should grow with T because the H₂ supply rate to the bubble
  scales with the FLiBe permeability Φ_F ∝ exp(−E_Φ_F/k_BT).
  Expected E_max ≈ E_Φ_F ≈ 0.5 eV.
- τ_grow should decrease with T (faster filling at higher T) because the
  supply rate increases.  If τ_grow is set purely by Φ_F-limited supply,
  expected E_τ ≈ E_Φ_F.  Significantly larger fitted E_τ would point at
  additional T-activated steps (e.g. nucleation).
- α_res ≈ 0.26 (fitted previously) reflects the geometric trapping of gas
  pockets in Ni-surface roughness.  Assumed T-independent.

Initial coverage from run history (no free parameter):
    α₀(650) = α_max(600)  — inherited from the prior 600 °C steady state
                              (the cell sat at 600 °C without shaking
                               before the 650 °C run).
    α₀(700) = α_res        — the initial shake at t=0 drives α down to
                              the residual; from there it grows back.
"""
from __future__ import annotations
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import minimize

KB_EV = 8.617333262e-5   # eV/K

DATA_DIR = Path(__file__).with_name("data")
GRID_DIR = Path(__file__).with_name("results_2d_alpha_grid")
OUT_DIR = Path(__file__).with_name("results_alpha_arrhenius")
OUT_DIR.mkdir(exist_ok=True)


def load_exp(T):
    df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


def load_grid(T):
    z = np.load(GRID_DIR / f"grid_{T}C.npz")
    return z["alphas"], z["t"], z["J"]


# ── Arrhenius parameterisation ─────────────────────────────────────────────
def alpha_max_T(T_K, amax_inf, E_max):
    return amax_inf * math.exp(-E_max / (KB_EV * T_K))


def tau_grow_T(T_K, tg0, E_tau):
    return tg0 * math.exp(E_tau / (KB_EV * T_K))


# ── α(t) ODE integration ───────────────────────────────────────────────────
def alpha_history(t_arr, alpha0, alpha_max, alpha_res, tau_grow,
                  tau_shake, shake_intervals_s):
    t_arr = np.asarray(t_arr, dtype=float)
    n = len(t_arr)
    alpha = np.empty(n)
    alpha[0] = alpha0

    def is_shake(t):
        for a, b in shake_intervals_s:
            if a <= t < b:
                return True
        return False

    for k in range(1, n):
        dt = t_arr[k] - t_arr[k - 1]
        a = alpha[k - 1]
        s = is_shake(t_arr[k - 1])
        if s:
            da = -(a - alpha_res) / tau_shake
        else:
            da = (alpha_max - a) / tau_grow
        alpha[k] = max(0.0, min(1.0, a + dt * da))
    return alpha


# ── Build splines once ─────────────────────────────────────────────────────
SPLINES = {}
T_GRIDS = {}
ALPHA_RANGES = {}
for T in (650, 700):
    ag, tg, Jg = load_grid(T)
    SPLINES[T] = RectBivariateSpline(ag, tg, Jg, kx=3, ky=3)
    T_GRIDS[T] = tg
    ALPHA_RANGES[T] = (ag[0], ag[-1])


def J_predicted(T, alpha_t):
    a_lo, a_hi = ALPHA_RANGES[T]
    a_clip = np.clip(alpha_t, a_lo, a_hi)
    return SPLINES[T](a_clip, T_GRIDS[T], grid=False)


# ── Loss function (650 + 700 combined) ─────────────────────────────────────
SHAKE_ON_650_HOURS = 3.91   # from the previous per-T fit; physically the
                            # time at which the operator started shaking
SHAKE_OFF_650_HOURS = 6.00


def predict_all(theta):
    amax_inf, E_max, tg0, E_tau, alpha_res = theta

    # Temperature points
    T_K_600, T_K_650, T_K_700 = 873.15, 923.15, 973.15

    amax_600 = alpha_max_T(T_K_600, amax_inf, E_max)
    amax_650 = alpha_max_T(T_K_650, amax_inf, E_max)
    amax_700 = alpha_max_T(T_K_700, amax_inf, E_max)

    tg_650 = tau_grow_T(T_K_650, tg0, E_tau)
    tg_700 = tau_grow_T(T_K_700, tg0, E_tau)

    # 650 °C
    t_650, _ = load_exp(650)
    a0_650 = min(amax_600, 1.0)    # inherited from 600 °C steady (capped at 1)
    intervals_650 = [(SHAKE_ON_650_HOURS * 3600, SHAKE_OFF_650_HOURS * 3600)]
    alpha_650 = alpha_history(t_650, a0_650, amax_650, alpha_res,
                              tg_650, 300.0, intervals_650)
    J_pred_650 = J_predicted(650, alpha_650)

    # 700 °C  (initial shake clears bubble to α_res, then it regrows)
    t_700, _ = load_exp(700)
    a0_700 = alpha_res
    intervals_700 = [(0.0, 0.08 * 3600)]
    alpha_700 = alpha_history(t_700, a0_700, amax_700, alpha_res,
                              tg_700, 300.0, intervals_700)
    J_pred_700 = J_predicted(700, alpha_700)

    return dict(
        amax_600=amax_600, amax_650=amax_650, amax_700=amax_700,
        tg_650=tg_650, tg_700=tg_700,
        alpha_650=alpha_650, alpha_700=alpha_700,
        J_pred_650=J_pred_650, J_pred_700=J_pred_700,
    )


def loss(theta):
    amax_inf, E_max, tg0, E_tau, alpha_res = theta
    # Physically reasonable bounds
    if not (1e-2 <= amax_inf <= 1e10): return 1e6
    if not (0.0 <= E_max  <= 3.0):     return 1e6
    if not (1e-20 <= tg0   <= 1.0):    return 1e6
    if not (0.0 <= E_tau  <= 8.0):     return 1e6
    if not (0.0 <= alpha_res <= 0.5):  return 1e6

    P = predict_all(theta)
    if P["amax_650"] >= 1.0 or P["amax_700"] >= 1.0:
        return 1e6
    if alpha_res > P["amax_650"]:
        return 1e6   # residual can't exceed equilibrium

    _, J_650 = load_exp(650)
    _, J_700 = load_exp(700)
    sc_650 = np.max(np.abs(J_650))
    sc_700 = np.max(np.abs(J_700))
    err_650 = np.sum((P["J_pred_650"] - J_650) ** 2) / (sc_650 ** 2) / len(J_650)
    err_700 = np.sum((P["J_pred_700"] - J_700) ** 2) / (sc_700 ** 2) / len(J_700)
    return float(err_650 + err_700)


# ── Run optimisation ───────────────────────────────────────────────────────
def main():
    # Initial guess motivated by the previous per-T fits:
    # α_max(650) = 0.30, α_max(700) = 0.46
    # → E_max ≈ kT_650 × T_700/(T_700−T_650) × ln(0.46/0.30) ≈ 0.66 eV
    # Take pre-factor consistent with α_max(650)=0.30:
    # α_max,∞ = 0.30 × exp(0.66 / (k_B × 923.15)) ≈ 1830
    # Similarly τ_grow(650)=117 min, τ_grow(700)=4 min
    # → E_τ ≈ 5.2 eV (very large — likely indicates additional physics).
    # We use 1.5 eV as a more physical starting point and let the optimiser
    # find what's needed.
    # α_max(650)=0.30 with E_max=0.66 eV gives pre-factor ~1200.
    # τ_grow(650)=117 min with E_τ=1.5 eV gives pre-factor ~5e-5.
    x0 = np.array([
        1200.0,      # α_max,∞
        0.66,        # E_max [eV]
        5e-5,        # τ_g,0  [s]
        1.5,         # E_τ   [eV]
        0.26,        # α_res
    ])

    print("=" * 70)
    print("Arrhenius-constrained fit: SINGLE parameter set for 650 °C + 700 °C")
    print("=" * 70)
    print(f"Initial: α_max,∞={x0[0]:.3g}, E_max={x0[1]:.3f} eV, "
          f"τ_g,0={x0[2]:.3g}, E_τ={x0[3]:.3f} eV, α_res={x0[4]:.3f}")
    print(f"loss(x0) = {loss(x0):.4e}")

    opt = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-7, maxiter=3000,
                                disp=False, adaptive=True))
    print(f"\nfinal loss = {opt.fun:.4e}")
    P = predict_all(opt.x)
    amax_inf, E_max, tg0, E_tau, alpha_res = opt.x
    print(f"\nFitted parameters (shared across 650 °C and 700 °C):")
    print(f"  α_max,∞ = {amax_inf:.3g}")
    print(f"  E_max   = {E_max:.4f} eV")
    print(f"  τ_g,0   = {tg0:.3g} s")
    print(f"  E_τ     = {E_tau:.4f} eV")
    print(f"  α_res   = {alpha_res:.4f}")
    print(f"\nDerived per-temperature values:")
    for label, key in [("α_max(600 °C)", "amax_600"),
                       ("α_max(650 °C)", "amax_650"),
                       ("α_max(700 °C)", "amax_700"),
                       ("τ_grow(650 °C) [min]", None),
                       ("τ_grow(700 °C) [min]", None)]:
        if key:
            print(f"  {label:>25s} = {P[key]:.4f}")
        elif "650" in label:
            print(f"  {label:>25s} = {P['tg_650']/60:.1f}")
        else:
            print(f"  {label:>25s} = {P['tg_700']/60:.1f}")
    print(f"  α₀(650 °C) = α_max(600 °C) = {P['amax_600']:.4f}")
    print(f"  α₀(700 °C) = α_res         = {alpha_res:.4f}")

    # ── Save results ───────────────────────────────────────────────────
    params_out = {
        "alpha_max_inf": amax_inf,
        "E_max_eV": E_max,
        "tau_grow_0_s": tg0,
        "E_tau_eV": E_tau,
        "alpha_res": alpha_res,
        "alpha_max_600": P["amax_600"],
        "alpha_max_650": P["amax_650"],
        "alpha_max_700": P["amax_700"],
        "tau_grow_650_s": P["tg_650"],
        "tau_grow_700_s": P["tg_700"],
        "alpha0_650": P["amax_600"],
        "alpha0_700": alpha_res,
        "shake_on_650_h": SHAKE_ON_650_HOURS,
        "shake_off_650_h": SHAKE_OFF_650_HOURS,
        "tau_shake_s": 300.0,
        "loss": opt.fun,
    }
    with (OUT_DIR / "params.txt").open("w") as f:
        f.write("# Arrhenius-constrained bubble model: shared params for 650 °C and 700 °C\n")
        for k, v in params_out.items():
            f.write(f"{k} = {v:.6e}\n")
    print(f"\nsaved {OUT_DIR / 'params.txt'}")

    # ── Separate 650 / 700 plots ───────────────────────────────────
    t650, J650 = load_exp(650)
    t700, J700 = load_exp(700)

    # 650 °C
    fig, (axJ, axa) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    axJ.plot(t650/3600, J650*1e6, "o", ms=3.0, color="0.35",
             label="experiment")
    axJ.plot(t650/3600, P["J_pred_650"]*1e6, "-", color="#d62728", lw=1.8,
             label="Arrhenius-constrained model")
    axJ.axvspan(SHAKE_ON_650_HOURS, SHAKE_OFF_650_HOURS, color="orange",
                alpha=0.15, label="shaking")
    axJ.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    axJ.set_title(f"650 °C   "
                  f"$\\alpha_{{max}}={P['amax_650']:.3f}$, "
                  f"$\\tau_{{grow}}={P['tg_650']/60:.0f}$ min  "
                  f"(derived from shared Arrhenius parameters)")
    axJ.legend(fontsize=10, loc="lower right"); axJ.grid(alpha=0.3)

    axa.plot(t650/3600, P["alpha_650"], color="#2ca02c", lw=1.8,
             label=r"$\alpha(t)$")
    axa.axhline(P["amax_650"], ls=":", color="0.5", lw=1,
                label=fr"$\alpha_{{max}}(650) = {P['amax_650']:.3f}$")
    axa.axhline(alpha_res, ls="--", color="0.5", lw=1,
                label=fr"$\alpha_{{res}} = {alpha_res:.3f}$")
    axa.axhline(P["amax_600"], ls="-.", color="0.5", lw=1,
                label=fr"$\alpha_0 = \alpha_{{max}}(600) = {P['amax_600']:.3f}$")
    axa.axvspan(SHAKE_ON_650_HOURS, SHAKE_OFF_650_HOURS, color="orange",
                alpha=0.15)
    axa.set_ylim(0, max(P["alpha_650"]) * 1.3 + 0.05)
    axa.set_xlabel("Time [h]"); axa.set_ylabel(r"Bubble coverage $\alpha(t)$")
    axa.legend(fontsize=9, loc="lower right"); axa.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "650C.png", dpi=180); plt.close(fig)
    print(f"saved {OUT_DIR / '650C.png'}")

    # 700 °C
    fig, (axJ, axa) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    axJ.plot(t700/3600, J700*1e6, "o", ms=3.0, color="0.35",
             label="experiment")
    axJ.plot(t700/3600, P["J_pred_700"]*1e6, "-", color="#d62728", lw=1.8,
             label="Arrhenius-constrained model")
    axJ.axvspan(0, 0.08, color="orange", alpha=0.15, label="initial shake")
    axJ.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    axJ.set_title(f"700 °C   "
                  f"$\\alpha_{{max}}={P['amax_700']:.3f}$, "
                  f"$\\tau_{{grow}}={P['tg_700']/60:.1f}$ min  "
                  f"(derived from shared Arrhenius parameters)")
    axJ.legend(fontsize=10, loc="lower right"); axJ.grid(alpha=0.3)

    axa.plot(t700/3600, P["alpha_700"], color="#2ca02c", lw=1.8,
             label=r"$\alpha(t)$")
    axa.axhline(P["amax_700"], ls=":", color="0.5", lw=1,
                label=fr"$\alpha_{{max}}(700) = {P['amax_700']:.3f}$")
    axa.axhline(alpha_res, ls="--", color="0.5", lw=1,
                label=fr"$\alpha_{{res}} = {alpha_res:.3f}$")
    axa.axvspan(0, 0.08, color="orange", alpha=0.15)
    axa.set_ylim(0, max(P["alpha_700"]) * 1.3 + 0.05)
    axa.set_xlabel("Time [h]"); axa.set_ylabel(r"Bubble coverage $\alpha(t)$")
    axa.legend(fontsize=9, loc="lower right"); axa.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "700C.png", dpi=180); plt.close(fig)
    print(f"saved {OUT_DIR / '700C.png'}")

    # ── Arrhenius plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    T_K = np.array([873.15, 923.15, 973.15])      # 600 / 650 / 700 C
    invT = 1000.0 / T_K
    amax_pts = np.array([P["amax_600"], P["amax_650"], P["amax_700"]])
    tg_pts   = np.array([np.nan,         P["tg_650"], P["tg_700"]])

    ax = axes[0]
    T_fine = np.linspace(870, 980, 300)
    amax_fine = [amax_inf * math.exp(-E_max / (KB_EV * T)) for T in T_fine]
    ax.semilogy(1000 / T_fine, amax_fine, "-", color="#d62728", lw=1.6,
                label=fr"$\alpha_{{max}} = {amax_inf:.2g}\cdot\exp(-{E_max:.3f}/k_B T)$")
    ax.semilogy(invT, amax_pts, "o", ms=8, mfc="white", mec="#d62728",
                mew=1.8, label="model predictions")
    for tk, am in zip(T_K, amax_pts):
        ax.annotate(f"{int(tk-273.15)} °C", xy=(1000/tk, am),
                    xytext=(4, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"$1000/T$  [1/K]")
    ax.set_ylabel(r"$\alpha_{max}$")
    ax.set_title(f"Equilibrium coverage   $E_{{max}}$ = {E_max:.3f} eV")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")

    ax = axes[1]
    tg_fine = [tg0 * math.exp(E_tau / (KB_EV * T)) for T in T_fine]
    ax.semilogy(1000 / T_fine, np.array(tg_fine) / 60, "-", color="#d62728",
                lw=1.6,
                label=fr"$\tau_{{grow}} = {tg0:.2g}\cdot\exp(+{E_tau:.3f}/k_B T)$")
    valid = ~np.isnan(tg_pts)
    ax.semilogy(invT[valid], tg_pts[valid] / 60, "o", ms=8, mfc="white",
                mec="#d62728", mew=1.8, label="model predictions")
    for tk, t in zip(T_K[valid], tg_pts[valid]):
        ax.annotate(f"{int(tk-273.15)} °C", xy=(1000/tk, t/60),
                    xytext=(4, -10), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"$1000/T$  [1/K]")
    ax.set_ylabel(r"$\tau_{grow}$  [min]")
    ax.set_title(fr"Growth time   $E_{{\tau}}$ = {E_tau:.3f} eV")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "arrhenius.png", dpi=180)
    plt.close(fig)
    print(f"saved {OUT_DIR / 'arrhenius.png'}")


if __name__ == "__main__":
    main()
