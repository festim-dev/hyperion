"""Parameter studies for the bubble model.

Each panel sweeps one parameter while holding the others at the 650 C
validation-fit values, so you can see how each knob shapes the curve.
The 650 C experimental data is overlaid for reference.
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
OUT_DIR = Path(__file__).with_name("results_param_study")
OUT_DIR.mkdir(exist_ok=True)

P_UP = 1.32e5
T_C_REF = 650.0
T_END = 7.5 * 3600  # seconds
SHAKE_ON = 3.5      # hours (from validation fit)


def read_kv(path: Path) -> dict:
    out = {}
    pat = re.compile(r"^\s*([A-Za-z_0-9]+)\s*=\s*([-+0-9.eE]+)")
    for line in path.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


base = read_kv(BASELINE_FILE)


def make_mat() -> bm.Materials:
    m = bm.Materials()
    m.Phi_F0 = base["Phi_F0"]
    m.E_PhiF = base["E_PhiF"]
    m.D_F0 = base["D_F0"]
    m.E_DF = base["E_DF"]
    return m


# Reference (best-fit at 650 C from validate_650_700.py)
REF = dict(
    alpha0=0.95,
    alpha_max=0.423,
    alpha_residual=0.371,
    tau_grow=2678.0,
    tau_shake=300.0,
)


def run(alpha0, alpha_max, alpha_residual, tau_grow, tau_shake,
        shake_on=SHAKE_ON, shake_off=6.0, t_end=T_END, n_x=60):
    shake = bm.make_shake_schedule([(shake_on, shake_off)])
    bub = bm.BubbleParams(
        alpha0=alpha0, alpha_max=alpha_max, alpha_residual=alpha_residual,
        tau_grow=tau_grow, tau_shake=tau_shake,
    )
    return bm.simulate(T_C=T_C_REF, P_up=P_UP, t_end_s=t_end,
                       mat=make_mat(), bub=bub, shake=shake,
                       n_x=n_x, dt_target=15.0)


def sweep_panel(ax, sweep_name, values, value_formatter, ref_label, **fixed_overrides):
    """Plot the model for each value of one parameter; overlay 650 C data."""
    df = pd.read_csv(DATA_DIR / f"T{T_C_REF:.0f}C.csv")
    ax.plot(df["time_h"], df["flux_mol_m2_s"] * 1e6, "o", ms=2.0,
            color="0.4", label="exp (650 °C)", zorder=1)

    cmap = matplotlib.colormaps["viridis"]
    for i, v in enumerate(values):
        params = dict(REF)
        params.update(fixed_overrides)
        params[sweep_name] = v
        res = run(**params)
        c = cmap(i / max(1, len(values) - 1))
        lbl = f"{sweep_name} = {value_formatter(v)}"
        if abs(v - REF[sweep_name]) < 1e-9:
            lbl += "  ★ ref"
        ax.plot(res.t / 3600, res.flux * 1e6, "-", color=c, lw=1.4, label=lbl)

    ax.axvspan(SHAKE_ON, 6.0, color="orange", alpha=0.08)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    ax.set_title(f"sweep: {sweep_name}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")


# ---------------------------------------------------------------------------
# Build figure of 4 sweeps
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True, sharey=True)

sweep_panel(
    axes[0, 0], "alpha_max",
    [0.20, 0.35, 0.423, 0.55, 0.70],
    value_formatter=lambda v: f"{v:.2f}",
    ref_label="0.42",
)

sweep_panel(
    axes[0, 1], "alpha_residual",
    [0.00, 0.20, 0.371, 0.42],
    value_formatter=lambda v: f"{v:.2f}",
    ref_label="0.37",
)

sweep_panel(
    axes[1, 0], "tau_grow",
    [500, 1500, 2678, 8000, 30000],
    value_formatter=lambda v: f"{v:.0f} s",
    ref_label="2678 s",
)

sweep_panel(
    axes[1, 1], "alpha0",
    [0.00, 0.30, 0.60, 0.95],
    value_formatter=lambda v: f"{v:.2f}",
    ref_label="0.95",
)

fig.suptitle(
    "Bubble-model parameter study  (650 °C; other params fixed at fit values)",
    fontsize=12,
)
fig.tight_layout()
out_path = OUT_DIR / "parameter_sweeps.png"
fig.savefig(out_path, dpi=180)
print("saved", out_path)


# ---------------------------------------------------------------------------
# Bonus: tau_shake sweep and "no-bubble vs bubble" comparison
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)

# tau_shake
df = pd.read_csv(DATA_DIR / f"T{T_C_REF:.0f}C.csv")
ax[0].plot(df["time_h"], df["flux_mol_m2_s"] * 1e6, "o", ms=2.0, color="0.4",
           label="exp (650 °C)")
cmap = matplotlib.colormaps["plasma"]
ts_values = [60, 300, 1500, 5000]
for i, ts in enumerate(ts_values):
    params = dict(REF); params["tau_shake"] = ts
    res = run(**params)
    c = cmap(i / max(1, len(ts_values) - 1))
    ax[0].plot(res.t / 3600, res.flux * 1e6, "-", color=c, lw=1.4,
               label=f"τ_shake = {ts} s")
ax[0].axvspan(SHAKE_ON, 6.0, color="orange", alpha=0.08)
ax[0].set_xlabel("Time [h]"); ax[0].set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
ax[0].set_title("sweep: τ_shake  (shake response speed)")
ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8, loc="lower right")

# No bubble vs full bubble model
ax[1].plot(df["time_h"], df["flux_mol_m2_s"] * 1e6, "o", ms=2.0, color="0.4",
           label="exp (650 °C)")
res_nobub = run(alpha0=0.0, alpha_max=0.0, alpha_residual=0.0,
                tau_grow=1e30, tau_shake=1e30)
ax[1].plot(res_nobub.t / 3600, res_nobub.flux * 1e6, "-", color="#2ca02c", lw=1.6,
           label="no bubble")
res_ref = run(**REF)
ax[1].plot(res_ref.t / 3600, res_ref.flux * 1e6, "-", color="#d62728", lw=1.6,
           label="full bubble model (ref)")
ax[1].axhline(res_nobub.J_ss_clean * 1e6, color="0.5", ls="--", lw=1,
              label=f"J_ss_clean = {res_nobub.J_ss_clean*1e6:.2f}")
ax[1].axvspan(SHAKE_ON, 6.0, color="orange", alpha=0.08)
ax[1].set_xlabel("Time [h]")
ax[1].set_title("no bubble vs bubble model")
ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8, loc="lower right")

fig2.tight_layout()
out_path2 = OUT_DIR / "tau_shake_and_nobubble.png"
fig2.savefig(out_path2, dpi=180)
print("saved", out_path2)


# ---------------------------------------------------------------------------
# Print a small text table summarising sensitivities
# ---------------------------------------------------------------------------
print("\nSensitivity of pre-shake plateau (J at t=2 h) and post-shake jump")
print("ΔJ = J(after shake stabilises ~5 h) − J(pre-shake ~2 h):")
def at(res, t_h):
    return float(np.interp(t_h * 3600, res.t, res.flux))
for name, vals, fmt in [
    ("alpha_max", [0.20, 0.35, 0.423, 0.55, 0.70], "{:.2f}"),
    ("alpha_residual", [0.00, 0.20, 0.371, 0.42], "{:.2f}"),
    ("tau_grow [s]", [500, 1500, 2678, 8000, 30000], "{:.0f}"),
    ("alpha0", [0.00, 0.30, 0.60, 0.95], "{:.2f}"),
]:
    key = name.split()[0]
    print(f"  --- {name} ---")
    for v in vals:
        p = dict(REF); p[key] = v
        r = run(**p)
        J_pre = at(r, 2.0)
        J_post = at(r, 5.0)
        print(f"    {key:>17} = {fmt.format(v):>8s}  "
              f"J(2h)={J_pre*1e6:.3f}  J(5h)={J_post*1e6:.3f}  "
              f"ΔJ={1e6*(J_post-J_pre):+.3f}  (×1e-6 mol/m²/s)")
