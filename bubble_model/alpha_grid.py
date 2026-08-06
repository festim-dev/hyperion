"""Pre-compute the 2D-axisymmetric FESTIM transient at a grid of fixed α
values for 650 and 700 °C, using the fitted FLiBe baseline.

These pre-computed curves are then combined with a time-dependent α(t) to
produce a dynamic-α model — see fit_alpha.py.

Output: results_2d_alpha_grid/grid_{T}C.npz with arrays
    alphas : (N_alpha,)
    t      : (N_t,)             (same time grid as experiment for this T)
    J      : (N_alpha, N_t)     mol H2 / m^2 / s
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).with_name("data")
BASE_FILE = Path(__file__).with_name("results_baseline_2d") / "fitted_baseline.txt"
OUT_DIR = Path(__file__).with_name("results_2d_alpha_grid")
OUT_DIR.mkdir(exist_ok=True)
WORKER = Path(__file__).with_name("worker.py")
ENV_PYTHON = sys.executable


def read_kv(p):
    import re
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([\w_]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def run_one(T_C, alpha, base):
    payload = dict(T_C=float(T_C), alpha=float(alpha),
                   Phi_F0_atoms=base["Phi_F0_atoms"],
                   E_PhiF=base["E_PhiF"],
                   D_F0=base["D_F0"], E_DF=base["E_DF"],
                   mesh_size=6e-4)
    cp = subprocess.run([ENV_PYTHON, str(WORKER), json.dumps(payload)],
                        capture_output=True, text=True, timeout=180)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr[-500:])
    j = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")][-1]
    return json.loads(j)


# coarse + fine α grid covering the range we expect to need.
ALPHA_GRIDS = {
    600: np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]),
    650: np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]),
    700: np.array([0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60]),
}


def main():
    base = read_kv(BASE_FILE)
    print("baseline:", base)

    for T in (600, 650, 700):
        df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
        t_exp = df["time_s"].to_numpy()
        alphas = ALPHA_GRIDS[T]
        J_grid = np.zeros((len(alphas), len(t_exp)))
        print(f"\n=== T = {T} °C  ({len(alphas)} α values × {len(t_exp)} t points) ===")
        for i, a in enumerate(alphas):
            r = run_one(T, a, base)
            J_grid[i] = np.asarray(r["J_model"])
            print(f"  α = {a:.3f}  ->  J_ss = {J_grid[i, -1]*1e6:.3f}  "
                  f"J_max = {J_grid[i].max()*1e6:.3f}", flush=True)
        out = OUT_DIR / f"grid_{T}C.npz"
        np.savez(out, alphas=alphas, t=t_exp, J=J_grid)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
