"""Worker script: run the 2D axisymmetric model for ONE (T, alpha,
Phi_F0, E_PhiF, D_F0, E_DF) and write the interpolated flux time series
on the experimental grid to stdout as a tiny JSON.

Used by fit_baseline.py and related fitters to keep each FESTIM run in a
fresh process (cleaning up MPI communicators).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import transport as ax

DATA_DIR = Path(__file__).with_name("data")


def main():
    payload = json.loads(sys.argv[1])
    T_C = float(payload["T_C"])
    alpha = float(payload.get("alpha", 0.0))
    ax.PHI_F0_ATOMS = float(payload["Phi_F0_atoms"])
    ax.E_PHIF = float(payload["E_PhiF"])
    ax.D_FLIBE_0 = float(payload["D_F0"])
    ax.E_D_FLIBE = float(payload["E_DF"])
    mesh_size = float(payload.get("mesh_size", 6e-4))

    df = pd.read_csv(DATA_DIR / f"T{int(T_C)}C.csv")
    t_exp = df["time_s"].to_numpy()
    t_end = float(t_exp[-1] + 60)

    res = ax.run_axisym(
        T_C=T_C, P_up=1.32e5, alpha=alpha,
        t_end_s=t_end, dt_value=60.0, mesh_size=mesh_size,
    )
    J_on_exp = np.interp(t_exp, res["t"], res["J_total_per_A"]).tolist()
    print(json.dumps({"t_exp": t_exp.tolist(), "J_model": J_on_exp,
                      "alpha": res["alpha"]}))


if __name__ == "__main__":
    main()
