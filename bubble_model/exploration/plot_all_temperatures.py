"""Final summary plot: all 5 temperatures on a single figure.

- 500/550/600 °C: pure-diffusion (no bubble) model with baseline-fit FLiBe params
- 650/700 °C: same FLiBe baseline + bubble model with the fitted bubble params
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import bubble_model_v2 as bm

DATA_DIR = Path(__file__).with_name("data")
BASELINE_FILE = Path(__file__).with_name("results_baseline") / "fitted_baseline.txt"
VAL_FILE = Path(__file__).with_name("results_validation") / "validation_summary.txt"
OUT_DIR = Path(__file__).with_name("results_validation")

P_UP = 1.32e5


def read_kv(path: Path) -> dict:
    out = {}
    pat = re.compile(r"^\s*([A-Za-z_0-9]+)\s*=\s*([-+0-9.eE]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_validation_params(path: Path):
    """Pull out the alpha0/alpha_max/alpha_residual/tau_grow per temperature."""
    text = path.read_text()
    out = {}
    for T in (650, 700):
        line_match = re.search(rf"{T} C:.*", text)
        if not line_match:
            raise ValueError(f"could not find {T} C line")
        line = line_match.group(0)
        m = re.search(
            r"alpha0=([-0-9.eE]+),\s*alpha_max=([-0-9.eE]+),\s*"
            r"alpha_residual=([-0-9.eE]+),\s*tau_grow=([-0-9.eE]+) s",
            line,
        )
        if not m:
            raise ValueError(f"could not parse {T} C: {line}")
        shake_on_m = re.search(r"shake_on=([-0-9.eE]+)", line)
        out[T] = dict(
            alpha0=float(m.group(1)),
            alpha_max=float(m.group(2)),
            alpha_residual=float(m.group(3)),
            tau_grow=float(m.group(4)),
            shake_on=float(shake_on_m.group(1)) if shake_on_m else None,
        )
    return out


base = read_kv(BASELINE_FILE)
val = parse_validation_params(VAL_FILE)


def make_mat() -> bm.Materials:
    m = bm.Materials()
    m.Phi_F0 = base["Phi_F0"]
    m.E_PhiF = base["E_PhiF"]
    m.D_F0 = base["D_F0"]
    m.E_DF = base["E_DF"]
    return m


def load(T_C: int):
    df = pd.read_csv(DATA_DIR / f"T{T_C}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


def run_no_bubble(T_C, t_end):
    bub = bm.BubbleParams(alpha0=0.0, alpha_max=0.0, tau_grow=1e30, tau_shake=1e30)
    return bm.simulate(T_C=T_C, P_up=P_UP, t_end_s=t_end, mat=make_mat(),
                       bub=bub, shake=lambda t: 0.0, n_x=80, dt_target=20.0)


def run_bubble(T_C, t_end, shake, params):
    bub = bm.BubbleParams(
        alpha0=params["alpha0"], alpha_max=params["alpha_max"],
        tau_grow=params["tau_grow"], tau_shake=300.0,
        alpha_residual=params["alpha_residual"],
    )
    return bm.simulate(T_C=T_C, P_up=P_UP, t_end_s=t_end, mat=make_mat(),
                       bub=bub, shake=shake, n_x=120, dt_target=10.0)


# ---------------------------------------------------------------------------
# Build figure -- 5 panels (2x3, last empty), comparing model vs data
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0), sharey=True)
axes = axes.flatten()
panel_T = [500, 550, 600, 650, 700]

for ax, T in zip(axes, panel_T):
    t, J = load(T)
    if T in (500, 550, 600):
        res = run_no_bubble(T, t[-1] + 60)
        sub = "no bubble"
    else:
        if T == 650:
            shake = bm.make_shake_schedule([(val[T]["shake_on"], 6.0)])
            sub = f"shake {val[T]['shake_on']:.1f}–6.0 h"
        else:
            shake = bm.SHAKE_700
            sub = "initial shake 0–0.08 h"
        res = run_bubble(T, t[-1] + 60, shake, val[T])

    ax.plot(t / 3600, J * 1e6, "o", ms=2.6, color="#1f77b4", label="experiment")
    ax.plot(res.t / 3600, res.flux * 1e6, "-", color="#d62728", lw=1.6, label="model")
    ax.axhline(res.J_ss_clean * 1e6, color="0.5", ls="--", lw=1,
               label=rf"$J_{{ss}}^{{clean}}$ = {res.J_ss_clean*1e6:.2f}")
    if T == 650:
        ax.axvspan(val[T]["shake_on"], 6.0, color="orange", alpha=0.12)
    if T == 700:
        ax.axvspan(0.0, 0.08, color="orange", alpha=0.12)

    ax.set_title(f"{T} °C    ({sub})")
    ax.set_xlabel("Time [h]")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.5)

axes[0].set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
axes[3].set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")

# leave panel 5 (index 5) for a parameter summary
ax_text = axes[5]
ax_text.axis("off")
txt = (
    "FLiBe baseline (fit to 500/550/600 °C, no bubble):\n"
    f"  Φ$_F$ = {base['Phi_F0']:.2e} · exp(−{base['E_PhiF']:.3f}/k$_B$T)\n"
    f"  D$_F$ = {base['D_F0']:.2e} · exp(−{base['E_DF']:.3f}/k$_B$T)\n"
    "\n"
    "Bubble parameters (fit at 650 / 700 °C):\n"
    f"  650 °C:  α$_0$={val[650]['alpha0']:.2f},  α$_{{max}}$={val[650]['alpha_max']:.2f},  "
    f"α$_{{res}}$={val[650]['alpha_residual']:.2f},\n"
    f"           τ$_g$={val[650]['tau_grow']:.0f} s,  shake on at {val[650]['shake_on']:.1f} h\n"
    f"  700 °C:  α$_0$={val[700]['alpha0']:.2f},  α$_{{max}}$={val[700]['alpha_max']:.2f},  "
    f"α$_{{res}}$={val[700]['alpha_residual']:.2f},\n"
    f"           τ$_g$={val[700]['tau_grow']:.0f} s,  initial shake 0–0.08 h\n"
    "\n"
    "Ni properties (literature, fixed):\n"
    f"  Φ$_{{Ni}}$ = 2.5e-8 · exp(−0.547/k$_B$T)  mol H$_2$/m/s/Pa$^{{0.5}}$\n"
    "\n"
    "Model: 1-D FLiBe diffusion (implicit BE) + Ni-Sievert downstream BC,\n"
    "downstream area scaled by (1−α(t)). Bubble coverage α evolves as\n"
    " dα/dt = (α$_{max}$−α)/τ$_g$   when not shaking,\n"
    " dα/dt = −(α−α$_{res}$)/τ$_s$  when shaking  (τ$_s$ = 300 s)."
)
ax_text.text(0.0, 1.0, txt, fontsize=8.5, family="monospace",
             va="top", ha="left", transform=ax_text.transAxes)

fig.suptitle("FLiBe / Ni permeation cell — unified diffusion + bubble model")
fig.tight_layout()
out_path = OUT_DIR / "all_temperatures.png"
fig.savefig(out_path, dpi=180)
print("saved", out_path)
