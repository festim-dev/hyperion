"""Model-free inversion of the effective interfacial transfer coefficient
k_int(t) from the measured permeation flux.

Physical picture
================
Treat the FLiBe/Ni interface as a finite-rate transfer step in series with
the bulk transport:

    J = k_int * (c_int - c_eq)          [c in mol H2 / m^3,  k_int in m/s]

With the downstream Ni acting as a strong sink (c_eq ~ 0), the series
resistance decomposition is

    R_tot(t)  = c_top / J_meas(t)
    R_bulk(t) = c_top / J_0(t)          (bubble-free 2D FESTIM model,
                                         includes the real geometry AND the
                                         bulk filling transient)
    R_int(t)  = R_tot(t) - R_bulk(t)
    k_int(t)  = 1 / R_int(t)

where  c_top = K_H(T) * P_up / (2 N_A)  is the Henry concentration imposed
at the FLiBe free surface (mol H2 / m^3) and
K_H = Phi_F / D_F  from the fitted FLiBe baseline.

NO bubble-coverage assumption, NO kinetics assumption: each time point is
inverted independently.  Validity: quasi-steady redistribution of the bulk
profile (good once the bulk transient is mostly complete; the early-time
window is masked accordingly).

Outputs (results_kint_inversion/):
    kint_{T}C.csv   : t_s, J_meas, J_0, R_int_s_per_m, k_int_m_per_s
    kint_{T}C.png   : 3-panel figure (J, R_int/R_bulk, k_int)
    kint_summary.png: k_int(t) at 650/700 C + bulk conductances
    J0_{T}C.npz     : cached bubble-free reference for 500/550 C
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

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
BASE_FILE = HERE / "results_baseline_2d" / "fitted_baseline.txt"
GRID_DIR = HERE / "results_2d_alpha_grid"
OUT_DIR = HERE / "results_kint_inversion"
OUT_DIR.mkdir(exist_ok=True)
WORKER = HERE / "worker.py"
ENV_PYTHON = sys.executable

KB_EV = 8.617333262e-5
N_A = 6.02214076e23
P_UP = 1.32e5  # Pa, same as in worker.py

# Shaking intervals (hours) — annotation only, NOT used in the inversion.
# 650 C: actual logged shaking period 3-6 h (user lab record).
SHAKE_H = {650: [(3.0, 6.0)], 700: [(0.0, 0.08)]}

TEMPS = [500, 550, 600, 650, 700]


def read_kv(p):
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([\w_]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def c_top_molH2(T_C, base):
    """Henry concentration at the FLiBe free surface, mol H2 / m^3."""
    T_K = T_C + 273.15
    K_H_atoms = (base["Phi_F0_atoms"] / base["D_F0"]) * np.exp(
        -(base["E_PhiF"] - base["E_DF"]) / (KB_EV * T_K))
    return K_H_atoms * P_UP / (2.0 * N_A)


def load_exp(T):
    df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


def get_J0(T, base):
    """Bubble-free 2D model flux on the experimental time grid."""
    grid_file = GRID_DIR / f"grid_{T}C.npz"
    if grid_file.exists():
        z = np.load(grid_file)
        i0 = int(np.argmin(np.abs(z["alphas"])))
        assert abs(z["alphas"][i0]) < 1e-9
        return z["t"], z["J"][i0]
    cache = OUT_DIR / f"J0_{T}C.npz"
    if cache.exists():
        z = np.load(cache)
        return z["t"], z["J"]
    print(f"  running bubble-free 2D model for {T} C ...", flush=True)
    payload = dict(T_C=float(T), alpha=0.0,
                   Phi_F0_atoms=base["Phi_F0_atoms"], E_PhiF=base["E_PhiF"],
                   D_F0=base["D_F0"], E_DF=base["E_DF"], mesh_size=6e-4)
    cp = subprocess.run([ENV_PYTHON, str(WORKER), json.dumps(payload)],
                        capture_output=True, text=True, timeout=600)
    if cp.returncode != 0:
        raise RuntimeError(cp.stderr[-800:])
    j = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")][-1]
    r = json.loads(j)
    t, J = np.asarray(r["t_exp"]), np.asarray(r["J_model"])
    np.savez(cache, t=t, J=J)
    return t, J


def smooth(y, win=7):
    """Centered moving median (robust to single-point spikes)."""
    y = np.asarray(y, float)
    n = len(y)
    out = np.empty(n)
    h = win // 2
    for i in range(n):
        out[i] = np.median(y[max(0, i - h):min(n, i + h + 1)])
    return out


def invert_one(T, base):
    t, J_meas = load_exp(T)
    t0, J0 = get_J0(T, base)
    if len(t0) != len(t) or np.max(np.abs(t0 - t)) > 1.0:
        J0 = np.interp(t, t0, J0)

    c_top = c_top_molH2(T, base)
    J_s = smooth(J_meas)

    with np.errstate(divide="ignore"):
        R_bulk = c_top / J0
        R_tot = c_top / J_s
    R_int = R_tot - R_bulk
    R_bulk_ss = c_top / J0[-1]
    k_bulk = 1.0 / R_bulk_ss

    # validity mask: bulk transient mostly complete
    valid = J0 > 0.8 * J0[-1]
    semi = J0 > 0.5 * J0[-1]

    with np.errstate(divide="ignore"):
        k_int = np.where(R_int > 0.02 * R_bulk_ss, 1.0 / R_int, np.nan)

    return dict(t=t, J_meas=J_meas, J_smooth=J_s, J0=J0, c_top=c_top,
                R_int=R_int, R_bulk_ss=R_bulk_ss, k_bulk=k_bulk,
                k_int=k_int, valid=valid, semi=semi)


def plot_one(T, r):
    th = r["t"] / 3600.0
    fig, (axJ, axR, axK) = plt.subplots(
        3, 1, figsize=(9, 9.5), sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1, 1]})

    axJ.plot(th, r["J_meas"] * 1e6, "o", ms=2.2, color="0.45",
             label="experiment")
    axJ.plot(th, r["J0"] * 1e6, "-", color="#d62728", lw=1.6,
             label="bubble-free 2D model $J_0(t)$")
    axJ.set_ylabel(r"Flux [$10^{-6}$ mol H$_2$/m$^2$/s]")
    axJ.set_title(f"{T} °C — model-free inversion of interfacial resistance")
    axJ.legend(fontsize=9, loc="lower right")

    ratio = r["R_int"] / r["R_bulk_ss"]
    axR.plot(th[r["semi"]], ratio[r["semi"]], ".", ms=2.5, color="#9ecae1")
    axR.plot(th[r["valid"]], ratio[r["valid"]], ".", ms=2.5, color="#1f77b4")
    axR.axhline(0.0, color="0.3", lw=0.8)
    axR.set_ylabel(r"$R_{\rm int}/R_{\rm bulk}^{ss}$")
    ymax = np.nanpercentile(ratio[r["valid"]], 99) if r["valid"].any() else 1
    axR.set_ylim(min(-0.3, -0.05 * ymax), max(1.0, 1.15 * ymax))

    axK.plot(th[r["valid"]], r["k_int"][r["valid"]], ".", ms=2.5,
             color="#2ca02c", label=r"$k_{\rm int}(t)$")
    axK.axhline(r["k_bulk"], color="#d62728", ls="--", lw=1.2,
                label=fr"bulk conductance $k_{{\rm bulk}}$ = "
                      fr"{r['k_bulk']:.2e} m/s")
    axK.set_yscale("log")
    axK.set_ylabel(r"$k_{\rm int}$ [m/s]")
    axK.set_xlabel("Time [h]")
    axK.legend(fontsize=9, loc="best")

    for ax in (axJ, axR, axK):
        ax.grid(alpha=0.3)
        for a, b in SHAKE_H.get(T, []):
            ax.axvspan(a, b, color="orange", alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"kint_{T}C.png", dpi=180)
    plt.close(fig)


def window_median(t_h, y, a, b, mask):
    sel = (t_h >= a) & (t_h < b) & mask & np.isfinite(y)
    return np.median(y[sel]) if sel.any() else np.nan


def main():
    base = read_kv(BASE_FILE)
    print("FLiBe baseline:", {k: f"{v:.4g}" for k, v in base.items()})
    results = {}
    for T in TEMPS:
        print(f"\n=== {T} °C ===", flush=True)
        r = invert_one(T, base)
        results[T] = r
        plot_one(T, r)
        pd.DataFrame({
            "t_s": r["t"], "J_meas": r["J_meas"], "J0_model": r["J0"],
            "R_int_s_per_m": r["R_int"], "k_int_m_per_s": r["k_int"],
            "valid": r["valid"].astype(int),
        }).to_csv(OUT_DIR / f"kint_{T}C.csv", index=False)

        th = r["t"] / 3600.0
        ratio = r["R_int"] / r["R_bulk_ss"]
        v = r["valid"]
        print(f"  c_top   = {r['c_top']:.3f} mol H2/m^3")
        print(f"  k_bulk  = {r['k_bulk']:.3e} m/s   "
              f"(R_bulk_ss = {r['R_bulk_ss']:.3e} s/m)")
        if v.any():
            print(f"  median R_int/R_bulk over valid window "
                  f"= {np.median(ratio[v]):+.3f}")
        if T == 650:
            for lab, (a, b) in [("pre-shake ", (1.5, 3.0)),
                                ("shaking   ", (3.0, 6.0)),
                                ("post-shake", (6.0, th[-1] + 1))]:
                km = window_median(th, r["k_int"], a, b, v)
                rm = window_median(th, ratio, a, b, v)
                print(f"  {lab}: median k_int = {km:.3e} m/s,  "
                      f"R_int/R_bulk = {rm:+.2f}")
        if T == 700:
            for lab, (a, b) in [("0.2-1 h   ", (0.2, 1.0)),
                                ("1-3 h     ", (1.0, 3.0)),
                                ("late      ", (3.0, th[-1] + 1))]:
                km = window_median(th, r["k_int"], a, b, v)
                rm = window_median(th, ratio, a, b, v)
                print(f"  {lab}: median k_int = {km:.3e} m/s,  "
                      f"R_int/R_bulk = {rm:+.2f}")

    # ── summary: k_int(t) at 650/700 + bulk conductances ──
    fig, ax = plt.subplots(figsize=(9, 5.2))
    colors = {650: "#1f77b4", 700: "#d62728"}
    for T in (650, 700):
        r = results[T]
        th = r["t"] / 3600.0
        v = r["valid"]
        ax.plot(th[v], r["k_int"][v], ".", ms=3, color=colors[T],
                label=fr"$k_{{\rm int}}(t)$, {T} °C")
        ax.axhline(r["k_bulk"], color=colors[T], ls="--", lw=1.0, alpha=0.7,
                   label=fr"$k_{{\rm bulk}}$({T} °C)")
    for a, b in SHAKE_H[650]:
        ax.axvspan(a, b, color="orange", alpha=0.12)
    ax.set_yscale("log")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"$k_{\rm int}$ [m/s]")
    ax.set_title("Inverted interfacial transfer coefficient (no kinetics assumed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "kint_summary.png", dpi=180)
    plt.close(fig)
    print(f"\nsaved figures + CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
