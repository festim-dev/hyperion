"""Fit the FLiBe permeability and diffusivity to the 500/550/600 C data
using the 2D axisymmetric FESTIM model with NO bubble (alpha=0).

Four parameters fit (Arrhenius for both Phi_F and D_F):
    log10(Phi_F0_atoms), E_PhiF, log10(D_F0), E_DF
Ni permeability held fixed at the user's independent dry-run fit.

Each loss evaluation runs each temperature in a fresh subprocess so that
FESTIM/MPI/PETSc don't leak communicators across iterations.
"""
from __future__ import annotations
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

DATA_DIR = Path(__file__).with_name("data")
OUT_DIR = Path(__file__).with_name("results_baseline_2d")
OUT_DIR.mkdir(exist_ok=True)
WORKER = Path(__file__).with_name("worker.py")
ENV_PYTHON = sys.executable

TS_FIT = [500, 550, 600]


def load(T):
    df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


data = {T: load(T) for T in TS_FIT}


def run_subprocess(T_C, alpha, Phi_F0_atoms, E_PhiF, D_F0, E_DF, mesh_size=6e-4):
    payload = dict(T_C=float(T_C), alpha=float(alpha),
                   Phi_F0_atoms=float(Phi_F0_atoms),
                   E_PhiF=float(E_PhiF),
                   D_F0=float(D_F0), E_DF=float(E_DF),
                   mesh_size=float(mesh_size))
    cmd = [ENV_PYTHON, str(WORKER), json.dumps(payload)]
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if cp.returncode != 0:
        raise RuntimeError(f"worker failed: {cp.stderr[-2000:]}")
    # Worker prints possibly some FESTIM/MPI banner; only the last line is JSON.
    out_lines = [ln for ln in cp.stdout.splitlines() if ln.strip().startswith("{")]
    if not out_lines:
        raise RuntimeError(f"no JSON output. stdout:\n{cp.stdout[-1000:]}")
    return json.loads(out_lines[-1])


_eval_counter = [0]
_eval_start = [time.time()]


def loss(x):
    log_phi, E_phi, log_D, E_d = x
    if not (12.0 <= log_phi <= 16.0): return 1e6
    if not (0.05 <= E_phi <= 1.0): return 1e6
    if not (-9.0 <= log_D <= -5.0): return 1e6
    if not (0.05 <= E_d <= 1.0): return 1e6
    Phi_F0 = 10.0 ** log_phi
    D_F0 = 10.0 ** log_D
    err = 0.0
    for T in TS_FIT:
        t_exp, J_exp = data[T]
        try:
            r = run_subprocess(T, 0.0, Phi_F0, E_phi, D_F0, E_d)
        except Exception as e:
            print(f"  [WARN] T={T} failed: {e}", flush=True)
            return 1e6
        Jm = np.asarray(r["J_model"])
        w = np.where(t_exp / 3600 > 0.3, 1.0, 0.2)
        scale = max(np.max(np.abs(J_exp)), 1e-12)
        err += float(np.sum(w * (Jm - J_exp) ** 2) / (scale * scale) / len(t_exp))
    _eval_counter[0] += 1
    if _eval_counter[0] % 5 == 0 or _eval_counter[0] == 1:
        elapsed = (time.time() - _eval_start[0]) / 60
        print(f"  eval #{_eval_counter[0]:>3d}  loss={err:.4e}  "
              f"x=(logPhi={log_phi:.3f}, E_phi={E_phi:.3f}, "
              f"logD={log_D:.3f}, E_d={E_d:.3f})  "
              f"[{elapsed:.1f} min]", flush=True)
    return err


print("=" * 70)
print("Fitting 2D FLiBe permeability/diffusivity to 500/550/600 C")
print("=" * 70)

x0 = np.array([
    math.log10(4.16e13),  # log10(Phi_F0_atoms)  -- htm/Calderoni start
    0.466,                # E_PhiF [eV]
    math.log10(2.5e-7),   # log10(D_F0)
    0.24,                 # E_DF [eV]
])
print(f"x0 = log_phi={x0[0]:.3f}, E_phi={x0[1]:.3f}, "
      f"log_D={x0[2]:.3f}, E_d={x0[3]:.3f}")
t_init = time.time()
print(f"loss(x0) = {loss(x0):.4e}  ({time.time()-t_init:.1f} s for one eval)")

print("\nRunning Nelder-Mead ...")
t0 = time.time()
opt = minimize(loss, x0, method="Nelder-Mead",
               options=dict(xatol=2e-3, fatol=1e-5, maxiter=150,
                            disp=True, adaptive=True))
xf = opt.x
print(f"\nfit time: {(time.time()-t0)/60:.1f} min")
print(f"final loss: {opt.fun:.4e}")
print(f"  log10(Phi_F0_atoms) = {xf[0]:.4f}  ->  Phi_F0_atoms = {10**xf[0]:.3e}")
print(f"  E_PhiF              = {xf[1]:.4f} eV")
print(f"  log10(D_F0)         = {xf[2]:.4f}  ->  D_F0 = {10**xf[2]:.3e} m^2/s")
print(f"  E_DF                = {xf[3]:.4f} eV")

Phi_F0_fit = 10.0 ** xf[0]
E_PhiF_fit = xf[1]
D_F0_fit = 10.0 ** xf[2]
E_DF_fit = xf[3]

# Final plots
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
for axp, T in zip(axes, TS_FIT):
    t_exp, J_exp = data[T]
    r = run_subprocess(T, 0.0, Phi_F0_fit, E_PhiF_fit, D_F0_fit, E_DF_fit)
    axp.plot(t_exp / 3600, J_exp * 1e6, "o", ms=2.6, color="0.4",
             label="experiment")
    axp.plot(t_exp / 3600, np.asarray(r["J_model"]) * 1e6, "-",
             color="#d62728", lw=1.6, label="2D axisym fit")
    axp.set_title(f"{T} °C")
    axp.set_xlabel("Time [h]")
    axp.grid(alpha=0.3)
    axp.legend(fontsize=8, loc="lower right")
axes[0].set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
fig.suptitle("2D FLiBe baseline fit  (α=0; 500/550/600 °C)")
fig.tight_layout()
fig.savefig(OUT_DIR / "baseline_fit.png", dpi=180)
print(f"\nsaved {OUT_DIR / 'baseline_fit.png'}")

with (OUT_DIR / "fitted_baseline.txt").open("w") as f:
    f.write("# Fitted FLiBe parameters from 2D axisymmetric model on 500/550/600 C\n")
    f.write(f"Phi_F0_atoms = {Phi_F0_fit:.6e}\n")
    f.write(f"E_PhiF       = {E_PhiF_fit:.6f}\n")
    f.write(f"D_F0         = {D_F0_fit:.6e}\n")
    f.write(f"E_DF         = {E_DF_fit:.6f}\n")
    f.write(f"loss         = {opt.fun:.6e}\n")
print(f"saved {OUT_DIR / 'fitted_baseline.txt'}")
