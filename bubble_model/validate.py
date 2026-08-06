"""Forward closure validation of the inversion-first workflow.

Two forward runs through the transient 2D grid J_2D(t; alpha):

(1) CLOSURE: alpha(t) = model-free alpha_eff(t) from the inversion
    (held at its first valid value during the masked bulk-transient window,
    and at 0 before/during the initial shake at 700 C).
    Tests that the quasi-steady inversion + steady-state geometric map are
    consistent with the full transient model.

(2) REDUCED-ORDER: alpha(t) from the two-branch ODE
        no shake : d(alpha)/dt = (alpha_eq - alpha)/tau_g
        shaking  : d(alpha)/dt = -(alpha - alpha_res)/tau_s
    with ALL parameters READ from the inversion (no optimisation):
        650 C: alpha_eq = 0.355, alpha_res = 0.26, tau_s = 59 min,
               shake = 3-6 h (lab record);
               alpha0 = 0.48, tau_g = 1.0 h (only constrain the masked
               early window; values consistent with the 1-3 h band)
        700 C: alpha_eq = 0.45, tau_g = 10 min (inversion upper bound
               ~20 min), initial shake 0-0.08 h, alpha0 = alpha_res = 0

Outputs: results_kint_inversion/forward_{650,700}C.png + RMSE table.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
GRID_DIR = HERE / "results_2d_alpha_grid"
KINT_DIR = HERE / "results_kint_inversion"
OUT_DIR = KINT_DIR

SHAKE_S = {650: [(3.0 * 3600, 6.0 * 3600)], 700: [(0.0, 0.08 * 3600)]}

ODE_PARAMS = {
    650: dict(alpha0=0.48, alpha_eq=0.355, alpha_res=0.26,
              tau_g=3600.0, tau_s=59 * 60.0),
    700: dict(alpha0=0.0, alpha_eq=0.45, alpha_res=0.0,
              tau_g=600.0, tau_s=59 * 60.0),
}


def load_exp(T):
    df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


def J_forward(T, alpha_t):
    z = np.load(GRID_DIR / f"grid_{T}C.npz")
    alphas_g, t_g, J_g = z["alphas"], z["t"], z["J"]
    spline = RectBivariateSpline(alphas_g, t_g, J_g, kx=3, ky=3)
    a = np.clip(alpha_t, alphas_g[0], alphas_g[-1])
    return spline(a, t_g, grid=False)


def alpha_ode(T, t):
    p = ODE_PARAMS[T]
    a = np.empty(len(t))
    a[0] = p["alpha0"]

    def shaking(tt):
        return any(lo <= tt < hi for lo, hi in SHAKE_S[T])

    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        prev = a[k - 1]
        if shaking(t[k - 1]):
            da = -(prev - p["alpha_res"]) / p["tau_s"]
        else:
            da = (p["alpha_eq"] - prev) / p["tau_g"]
        a[k] = min(1.0, max(0.0, prev + dt * da))
    return a


def alpha_closure(T, t):
    """Model-free alpha_eff(t), extended through the masked window."""
    df = pd.read_csv(KINT_DIR / f"alpha_eff_{T}C.csv")
    a = df["alpha_eff"].to_numpy().copy()
    valid = df["valid"].to_numpy().astype(bool)
    first = np.argmax(valid)
    a[:first] = a[first]                      # hold-back through masked window
    if T == 700:                              # before/during initial shake
        a[t <= 0.08 * 3600] = 0.0
    return a, valid


def rmse(x, y, sel=None):
    if sel is not None:
        x, y = x[sel], y[sel]
    return float(np.sqrt(np.mean((x - y) ** 2)))


def main():
    for T in (650, 700):
        t, J_exp = load_exp(T)
        a_cl, valid = alpha_closure(T, t)
        a_od = alpha_ode(T, t)
        J_cl = J_forward(T, a_cl)
        J_od = J_forward(T, a_od)

        r_cl_v = rmse(J_cl, J_exp, valid) * 1e6
        r_od_v = rmse(J_od, J_exp, valid) * 1e6
        r_od_all = rmse(J_od, J_exp) * 1e6
        Jbar = np.mean(J_exp[valid]) * 1e6
        print(f"=== {T} °C ===")
        print(f"  closure  J[alpha_eff(t)] : RMSE(valid) = {r_cl_v:.3f}"
              f"  ({100*r_cl_v/Jbar:.1f} % of mean flux)")
        print(f"  reduced-order ODE        : RMSE(valid) = {r_od_v:.3f}"
              f"  ({100*r_od_v/Jbar:.1f} %),  RMSE(all) = {r_od_all:.3f}"
              f"  [1e-6 mol/m2/s]")

        th = t / 3600.0
        fig, (axJ, axa) = plt.subplots(
            2, 1, figsize=(9, 6.8), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]})
        axJ.plot(th, J_exp * 1e6, "o", ms=2.6, color="0.4",
                 label="experiment")
        axJ.plot(th, J_cl * 1e6, "-", color="#2ca02c", lw=1.8,
                 label=r"closure: $J[\alpha_{\rm eff}(t)]$")
        axJ.plot(th, J_od * 1e6, "--", color="#d62728", lw=1.8,
                 label="reduced-order ODE (no fit)")
        first = np.argmax(valid)
        axJ.axvspan(th[0], th[first], color="0.85", alpha=0.5,
                    label="bulk transient (masked)")
        for lo, hi in SHAKE_S[T]:
            axJ.axvspan(lo / 3600, hi / 3600, color="orange", alpha=0.15)
        axJ.set_ylabel(r"Flux [$10^{-6}$ mol H$_2$/m$^2$/s]")
        axJ.set_title(f"{T} °C — forward closure validation")
        axJ.legend(fontsize=9, loc="lower right")
        axJ.grid(alpha=0.3)

        axa.plot(th, a_cl, "-", color="#2ca02c", lw=1.6,
                 label=r"$\alpha_{\rm eff}(t)$ (held in masked window)")
        axa.plot(th, a_od, "--", color="#d62728", lw=1.6,
                 label="reduced-order ODE")
        for lo, hi in SHAKE_S[T]:
            axa.axvspan(lo / 3600, hi / 3600, color="orange", alpha=0.15)
        axa.set_xlabel("Time [h]")
        axa.set_ylabel(r"$\alpha$")
        axa.set_ylim(0, 0.7)
        axa.legend(fontsize=9, loc="best")
        axa.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"forward_{T}C.png", dpi=180)
        plt.close(fig)
        print(f"  saved {OUT_DIR / f'forward_{T}C.png'}")


if __name__ == "__main__":
    main()
