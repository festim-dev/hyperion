"""Final all-temperatures comparison plot for the 2D axisymmetric model:
- 500/550/600 C: 2D model with α = 0  (fitted FLiBe baseline)
- 650/700 C   : 2D model with time-dependent α(t) (fitted ODE parameters)
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

DATA_DIR = Path(__file__).with_name("data")
BASELINE_FILE = Path(__file__).with_name("results_baseline_2d") / "fitted_baseline.txt"
ALPHA_FILE = Path(__file__).with_name("results_dynamic_alpha") / "params.txt"
GRID_DIR = Path(__file__).with_name("results_2d_alpha_grid")
OUT_DIR = Path(__file__).with_name("results_final_2d")
OUT_DIR.mkdir(exist_ok=True)
WORKER = Path(__file__).with_name("_run_single_case.py")
ENV_PYTHON = sys.executable


def read_kv(p):
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([\w_]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def load_exp(T):
    df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


def run_2d_no_bubble(T_C, base):
    t_exp, _ = load_exp(T_C)
    payload = dict(T_C=float(T_C), alpha=0.0,
                   Phi_F0_atoms=base["Phi_F0_atoms"],
                   E_PhiF=base["E_PhiF"],
                   D_F0=base["D_F0"], E_DF=base["E_DF"],
                   mesh_size=6e-4)
    cp = subprocess.run([ENV_PYTHON, str(WORKER), json.dumps(payload)],
                        capture_output=True, text=True, timeout=180)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr[-500:])
    j = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")][-1]
    r = json.loads(j)
    return np.asarray(r["t_exp"]), np.asarray(r["J_model"])


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


def J_dynamic(T, alpha_t):
    z = np.load(GRID_DIR / f"grid_{T}C.npz")
    alphas_g, t_g, J_g = z["alphas"], z["t"], z["J"]
    spline = RectBivariateSpline(alphas_g, t_g, J_g, kx=3, ky=3)
    a_clip = np.clip(alpha_t, alphas_g[0], alphas_g[-1])
    return spline(a_clip, t_g, grid=False)


def main():
    base = read_kv(BASELINE_FILE)
    alphas = read_kv(ALPHA_FILE)

    print("FLiBe baseline (fit on 500/550/600 °C, no bubble):")
    for k in ("Phi_F0_atoms", "E_PhiF", "D_F0", "E_DF"):
        print(f"  {k:>14s} = {base[k]:.4e}")
    print("\nDynamic α parameters:")
    for k in ("alpha0_650", "alpha_max_650", "alpha_res_650", "tau_grow_650",
              "shake_on_650_h", "alpha_max_700", "tau_grow_700"):
        print(f"  {k:>16s} = {alphas[k]:.4f}")

    # ── (1) Overview plot: 500/550/600 (no-bubble) all in one row ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for axp, T in zip(axes, [500, 550, 600]):
        t_exp, J_exp = load_exp(T)
        t_m, J_m = run_2d_no_bubble(T, base)
        axp.plot(t_exp / 3600, J_exp * 1e6, "o", ms=2.4, color="0.4",
                 label="experiment")
        axp.plot(t_m / 3600, J_m * 1e6, "-", color="#d62728", lw=1.6,
                 label="2D model, α=0")
        axp.set_title(f"{T} °C   (no bubble)")
        axp.set_xlabel("Time [h]")
        axp.grid(alpha=0.3)
        axp.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    fig.suptitle("2D axisymmetric model — no bubble (500/550/600 °C)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "no_bubble_500_550_600.png", dpi=180)
    plt.close(fig)
    print(f"saved {OUT_DIR / 'no_bubble_500_550_600.png'}")

    # ── (2) 650 °C: J(t) and α(t) on separate panels ──
    t650, J650 = load_exp(650)
    intervals_650 = [(alphas["shake_on_650_h"] * 3600, 6.0 * 3600)]
    a_t_650 = alpha_history(t650, alphas["alpha0_650"],
                            alphas["alpha_max_650"], alphas["alpha_res_650"],
                            alphas["tau_grow_650"], 300.0, intervals_650)
    J_m_650 = J_dynamic(650, a_t_650)

    fig, (axJ, axa) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    axJ.plot(t650 / 3600, J650 * 1e6, "o", ms=3.0, color="0.35",
             label="experiment")
    axJ.plot(t650 / 3600, J_m_650 * 1e6, "-", color="#d62728", lw=1.8,
             label="2D model with α(t)")
    axJ.axvspan(alphas["shake_on_650_h"], 6.0, color="orange", alpha=0.15,
                label="shaking")
    axJ.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    axJ.set_title("650 °C")
    axJ.legend(fontsize=10, loc="lower right")
    axJ.grid(alpha=0.3)

    axa.plot(t650 / 3600, a_t_650, color="#2ca02c", lw=1.8)
    axa.axhline(alphas["alpha_max_650"], color="0.5", ls=":", lw=1,
                label=fr"$\alpha_{{max}}$ = {alphas['alpha_max_650']:.3f}")
    axa.axhline(alphas["alpha_res_650"], color="0.5", ls="--", lw=1,
                label=fr"$\alpha_{{res}}$ = {alphas['alpha_res_650']:.3f}")
    axa.axvspan(alphas["shake_on_650_h"], 6.0, color="orange", alpha=0.15)
    axa.set_ylim(0, max(a_t_650) * 1.3 + 0.05)
    axa.set_xlabel("Time [h]")
    axa.set_ylabel(r"Bubble coverage  $\alpha(t)$")
    axa.legend(fontsize=9, loc="upper right")
    axa.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "650C.png", dpi=180)
    plt.close(fig)
    print(f"saved {OUT_DIR / '650C.png'}")

    # ── (3) 700 °C: J(t) and α(t) on separate panels ──
    t700, J700 = load_exp(700)
    a_t_700 = alpha_history(t700, 0.0, alphas["alpha_max_700"], 0.0,
                            alphas["tau_grow_700"], 300.0,
                            [(0.0, 0.08 * 3600.0)])
    J_m_700 = J_dynamic(700, a_t_700)

    fig, (axJ, axa) = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True,
                                    gridspec_kw={"height_ratios": [2.2, 1]})
    axJ.plot(t700 / 3600, J700 * 1e6, "o", ms=3.0, color="0.35",
             label="experiment")
    axJ.plot(t700 / 3600, J_m_700 * 1e6, "-", color="#d62728", lw=1.8,
             label="2D model with α(t)")
    axJ.axvspan(0.0, 0.08, color="orange", alpha=0.15, label="initial shake")
    axJ.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    axJ.set_title("700 °C")
    axJ.legend(fontsize=10, loc="lower right")
    axJ.grid(alpha=0.3)

    axa.plot(t700 / 3600, a_t_700, color="#2ca02c", lw=1.8)
    axa.axhline(alphas["alpha_max_700"], color="0.5", ls=":", lw=1,
                label=fr"$\alpha_{{max}}$ = {alphas['alpha_max_700']:.3f}")
    axa.axvspan(0.0, 0.08, color="orange", alpha=0.15)
    axa.set_ylim(0, max(a_t_700) * 1.3 + 0.05)
    axa.set_xlabel("Time [h]")
    axa.set_ylabel(r"Bubble coverage  $\alpha(t)$")
    axa.legend(fontsize=9, loc="lower right")
    axa.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "700C.png", dpi=180)
    plt.close(fig)
    print(f"saved {OUT_DIR / '700C.png'}")


if __name__ == "__main__":
    main()
