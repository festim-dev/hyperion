"""Map the inverted interfacial resistance R_int(t) to an EQUIVALENT bubble
coverage alpha_eff(t), using only the steady-state 2D FESTIM grid J_ss(alpha).

No kinetics assumed: the 2D grid provides a purely geometric conversion
    R_2D(alpha) = c_top * (1/J_ss(alpha) - 1/J_ss(0))
which is inverted pointwise:
    alpha_eff(t) = R_2D^{-1}( R_int(t) )

The result is compared with the alpha(t) ODE previously FITTED to the flux
(results_dynamic_alpha/params.txt).  Agreement would mean the pure
area-blocking picture is sufficient; systematic disagreement would point to
additional interface kinetics (partial blocking / eta).

Noise floor: the 500-600 C control inversions scatter within
|R_int| <= 0.16 R_bulk; this is propagated through the map as a band.

Inputs : results_kint_inversion/kint_{T}C.csv   (from invert_kint.py)
         results_2d_alpha_grid/grid_{T}C.npz
Outputs: results_kint_inversion/alpha_eff_{T}C.png, alpha_eff_{T}C.csv,
         alpha_eff_summary.png
"""
from __future__ import annotations
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
BASE_FILE = HERE / "results_baseline_2d" / "fitted_baseline.txt"
ALPHA_FILE = HERE / "results_dynamic_alpha" / "params.txt"
GRID_DIR = HERE / "results_2d_alpha_grid"
KINT_DIR = HERE / "results_kint_inversion"
OUT_DIR = KINT_DIR

KB_EV = 8.617333262e-5
N_A = 6.02214076e23
P_UP = 1.32e5

NOISE_R = 0.16          # |R_int|/R_bulk noise floor from 500-600 C controls
# 650 C: actual logged shaking period 3-6 h (user lab record).
SHAKE_H = {650: [(3.0, 6.0)], 700: [(0.0, 0.08)]}
ALPHA_VISUAL = 0.26     # coverage from visual bubble observation (r_b ~ 20 mm)


def read_kv(p):
    out = {}
    for line in p.read_text().splitlines():
        m = re.match(r"^\s*([\w_]+)\s*=\s*([-+0-9.eE]+)", line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def c_top_molH2(T_C, base):
    T_K = T_C + 273.15
    K_H = (base["Phi_F0_atoms"] / base["D_F0"]) * np.exp(
        -(base["E_PhiF"] - base["E_DF"]) / (KB_EV * T_K))
    return K_H * P_UP / (2.0 * N_A)


def alpha_map(T, base):
    """Geometric map R_2D(alpha) from the steady-state 2D grid."""
    z = np.load(GRID_DIR / f"grid_{T}C.npz")
    alphas, J = z["alphas"], z["J"]
    J_ss = J[:, -1]                      # late-time flux is steady (checked)
    c_top = c_top_molH2(T, base)
    R = c_top * (1.0 / J_ss - 1.0 / J_ss[0])
    assert np.all(np.diff(R) > 0), "R_2D(alpha) must be monotonic"
    return alphas, R, c_top, J_ss


def alpha_history(t_arr, alpha0, alpha_max, alpha_res, tau_grow,
                  tau_shake, shake_intervals_s):
    t_arr = np.asarray(t_arr, dtype=float)
    alpha = np.empty(len(t_arr))
    alpha[0] = alpha0

    def is_shake(t):
        return any(a <= t < b for a, b in shake_intervals_s)

    for k in range(1, len(t_arr)):
        dt = t_arr[k] - t_arr[k - 1]
        a = alpha[k - 1]
        if is_shake(t_arr[k - 1]):
            da = -(a - alpha_res) / tau_shake
        else:
            da = (alpha_max - a) / tau_grow
        alpha[k] = max(0.0, min(1.0, a + dt * da))
    return alpha


def ode_alpha(T, t, prm):
    if T == 650:
        iv = [(prm["shake_on_650_h"] * 3600, 6.0 * 3600)]
        return alpha_history(t, prm["alpha0_650"], prm["alpha_max_650"],
                             prm["alpha_res_650"], prm["tau_grow_650"],
                             300.0, iv)
    if T == 700:
        return alpha_history(t, 0.0, prm["alpha_max_700"], 0.0,
                             prm["tau_grow_700"], 300.0,
                             [(0.0, 0.08 * 3600.0)])
    return None


def main():
    base = read_kv(BASE_FILE)
    prm = read_kv(ALPHA_FILE)
    results = {}

    for T in (600, 650, 700):
        df = pd.read_csv(KINT_DIR / f"kint_{T}C.csv")
        t = df["t_s"].to_numpy()
        R_int = df["R_int_s_per_m"].to_numpy()
        valid = df["valid"].to_numpy().astype(bool)

        alphas_g, R_g, c_top, J_ss = alpha_map(T, base)
        R_bulk_ss = c_top / J_ss[0]

        def to_alpha(R):
            return np.interp(np.clip(R, 0.0, R_g[-1]), R_g, alphas_g)

        a_eff = to_alpha(R_int)
        a_lo = to_alpha(R_int - NOISE_R * R_bulk_ss)
        a_hi = to_alpha(R_int + NOISE_R * R_bulk_ss)
        saturated = R_int > R_g[-1]

        a_ode = ode_alpha(T, t, prm)
        results[T] = dict(t=t, a_eff=a_eff, a_lo=a_lo, a_hi=a_hi,
                          valid=valid, a_ode=a_ode, saturated=saturated)

        pd.DataFrame({
            "t_s": t, "alpha_eff": a_eff, "alpha_lo": a_lo, "alpha_hi": a_hi,
            "valid": valid.astype(int),
        }).to_csv(OUT_DIR / f"alpha_eff_{T}C.csv", index=False)

        # per-T figure
        th = t / 3600.0
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.fill_between(th[valid], a_lo[valid], a_hi[valid],
                        color="#2ca02c", alpha=0.18,
                        label="noise floor (low-T control)")
        ax.plot(th[valid], a_eff[valid], ".", ms=3.2, color="#2ca02c",
                label=r"$\alpha_{\rm eff}(t)$ from inversion (no kinetics)")
        if a_ode is not None:
            ax.plot(th, a_ode, "-", color="#d62728", lw=1.8,
                    label=r"previously fitted ODE $\alpha(t)$")
        ax.axhline(ALPHA_VISUAL, color="0.4", ls=":", lw=1.2,
                   label=fr"visual observation $\alpha \approx$ {ALPHA_VISUAL}")
        for a, b in SHAKE_H.get(T, []):
            ax.axvspan(a, b, color="orange", alpha=0.15)
        ax.set_xlabel("Time [h]")
        ax.set_ylabel(r"Bubble coverage  $\alpha$")
        ax.set_ylim(0, 0.75)
        ax.set_title(f"{T} °C — equivalent coverage from geometric map")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"alpha_eff_{T}C.png", dpi=180)
        plt.close(fig)

        v = valid
        print(f"=== {T} °C ===")
        print(f"  R_2D(alpha) grid: alpha in [0, {alphas_g[-1]:.2f}], "
              f"R up to {R_g[-1]/R_bulk_ss:.2f} R_bulk")
        if v.any():
            print(f"  median alpha_eff (valid) = {np.median(a_eff[v]):.3f}")
            print(f"  saturated points: {int(saturated[v].sum())}")
        if T == 650:
            for lab, (a, b) in [("pre-shake ", (1.5, 3.0)),
                                ("shaking   ", (3.0, 6.0)),
                                ("post-shake", (6.0, 99))]:
                s = (th >= a) & (th < b) & v
                if s.any():
                    print(f"  {lab}: alpha_eff = {np.median(a_eff[s]):.3f} "
                          f"[{np.median(a_lo[s]):.3f}, "
                          f"{np.median(a_hi[s]):.3f}]")
            # descriptive time constant of coverage removal during shaking:
            # alpha_eff = a_inf + (a_0 - a_inf) exp(-(t - t0)/tau)
            from scipy.optimize import curve_fit
            t0 = 3.0
            s = (th >= t0) & (th < 6.0) & v
            if s.sum() > 10:
                def expo(t, a_inf, a_0, tau_h):
                    return a_inf + (a_0 - a_inf) * np.exp(-(t - t0) / tau_h)
                p, _ = curve_fit(expo, th[s], a_eff[s],
                                 p0=[0.26, 0.36, 1.0],
                                 bounds=([0, 0, 0.05], [1, 1, 5]))
                print(f"  shake-removal exponential: alpha_inf = {p[0]:.3f}, "
                      f"alpha(3h) = {p[1]:.3f}, tau = {p[2]*60:.0f} min")
        if T == 700:
            for lab, (a, b) in [("1-3 h", (1.0, 3.0)), ("late ", (3.0, 99))]:
                s = (th >= a) & (th < b) & v
                if s.any():
                    print(f"  {lab}: alpha_eff = {np.median(a_eff[s]):.3f} "
                          f"[{np.median(a_lo[s]):.3f}, "
                          f"{np.median(a_hi[s]):.3f}]")

    # summary figure: 650 + 700
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, T in zip(axes, (650, 700)):
        r = results[T]
        th = r["t"] / 3600.0
        v = r["valid"]
        ax.fill_between(th[v], r["a_lo"][v], r["a_hi"][v],
                        color="#2ca02c", alpha=0.18)
        ax.plot(th[v], r["a_eff"][v], ".", ms=3.2, color="#2ca02c",
                label=r"$\alpha_{\rm eff}(t)$, model-free")
        ax.axhline(ALPHA_VISUAL, color="0.4", ls=":", lw=1.2,
                   label=fr"visual observation $\alpha\approx{ALPHA_VISUAL}$")
        for a, b in SHAKE_H.get(T, []):
            ax.axvspan(a, b, color="orange", alpha=0.15)
        ax.set_xlabel("Time [h]")
        ax.set_title(f"{T} °C")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")
    axes[0].set_ylabel(r"Bubble coverage  $\alpha$")
    axes[0].set_ylim(0, 0.75)
    fig.suptitle("Equivalent bubble coverage from the geometric map")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_eff_summary.png", dpi=180)
    plt.close(fig)
    print(f"\nsaved figures + CSVs in {OUT_DIR}")


if __name__ == "__main__":
    main()
