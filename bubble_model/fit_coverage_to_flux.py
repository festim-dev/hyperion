"""Fit a time-dependent bubble coverage α(t) for 650 / 700 °C, using the
pre-computed 2D-axisymmetric FESTIM flux grid (precompute_flux_grid.py).

Coverage model
--------------
α(t) ∈ [0, 1] is the fraction of FLiBe/Ni interface area blocked by a
bubble, evolving as

    dα/dt =  (α_max − α) / τ_grow      when NOT shaking
    dα/dt = −(α − α_res) / τ_shake     when shaking

  α_max  : steady-state coverage in still operation (growth/removal
           equilibrium). Larger T -> more H supply -> larger α_max.
  α_res  : residual coverage shaking cannot remove (gas trapped in
           Ni-surface roughness) — why shaking only partially recovers
           the no-bubble flux.
  τ_grow : bubble growth timescale, set by the H2 supply rate from FLiBe.
  τ_shake: shake-driven removal timescale, fixed at 300 s.
  α(t=0) : for 650 C, inherited from the prior 600 C run (α(0) ≈ α_max);
           for 700 C, the initial shake clears the bubble (α(0) ≈ 0).

The 2D FESTIM J(t; α=fixed) is pre-computed on a grid of α values; for a
slowly-varying α(t) the flux is approximated by interpolation,
J(t) ≈ J_2D(t; α(t)), valid once τ_diff(2D) << τ_grow/shake (here
τ_diff ≈ 30 min vs τ_grow ≈ 1 h, so the approximation is within ~10%).

Output: results_dynamic_alpha/{fit_650.png, fit_700.png, params.txt}
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

DATA_DIR = Path(__file__).with_name("data")
GRID_DIR = Path(__file__).with_name("results_2d_alpha_grid")
OUT_DIR = Path(__file__).with_name("results_dynamic_alpha")
OUT_DIR.mkdir(exist_ok=True)


def load_grid(T):
    z = np.load(GRID_DIR / f"grid_{T}C.npz")
    return z["alphas"], z["t"], z["J"]   # (Nα,), (Nt,), (Nα, Nt)


def load_exp(T):
    df = pd.read_csv(DATA_DIR / f"T{T}C.csv")
    return df["time_s"].to_numpy(), df["flux_mol_m2_s"].to_numpy()


def alpha_history(t_arr, alpha0, alpha_max, alpha_res, tau_grow,
                  tau_shake, shake_intervals_s):
    """Integrate the α ODE on the supplied (sorted) time grid."""
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


def J_model_from_alpha(alpha_t, alphas_grid, t_grid, J_grid):
    """For each (t_i, α_i) return J_2D interpolated bilinearly from the
    grid J_grid[α, t]."""
    # alpha_t is length N_t (same t_grid).  We evaluate J_grid(α, t) at
    # the diagonal (α_i, t_i).
    # Build a spline once for efficiency.
    spline = RectBivariateSpline(alphas_grid, t_grid, J_grid,
                                  kx=min(3, len(alphas_grid) - 1),
                                  ky=min(3, len(t_grid) - 1))
    # diag eval
    return spline(alpha_t, t_grid, grid=False)


# ─────────────────────────────────────────────────────────────────────────────
# Fit α(t) params to the 650 °C and 700 °C curves
# ─────────────────────────────────────────────────────────────────────────────
def fit_650():
    t_exp, J_exp = load_exp(650)
    alphas_g, t_g, J_g = load_grid(650)
    assert np.allclose(t_g, t_exp), "grid t must match experiment t"

    spline = RectBivariateSpline(alphas_g, t_g, J_g,
                                  kx=3, ky=3)

    def J_at(alpha_t):
        # alpha_t evaluated at t_exp (same grid).  Clip α to grid range.
        a = np.clip(alpha_t, alphas_g[0], alphas_g[-1])
        return spline(a, t_g, grid=False)

    # Shake schedule: user described "shake after first plateau, stop at 6 h".
    # Treat the start of shaking as another fit parameter.
    def loss(x):
        alpha0, alpha_max, alpha_res, tau_grow, t_shake_on = x
        if not (0.0 <= alpha0 <= 0.6): return 1e6
        if not (0.0 <= alpha_max <= 0.6): return 1e6
        if not (0.0 <= alpha_res <= alpha_max + 0.01): return 1e6
        if not (200 <= tau_grow <= 3e4): return 1e6
        if not (1.5 <= t_shake_on <= 4.5): return 1e6
        intervals = [(t_shake_on * 3600.0, 6.0 * 3600.0)]
        alpha_t = alpha_history(t_exp, alpha0, alpha_max, alpha_res,
                                tau_grow, 300.0, intervals)
        Jm = J_at(alpha_t)
        scale = max(np.max(np.abs(J_exp)), 1e-12)
        return float(np.sum((Jm - J_exp) ** 2) / (scale * scale) / len(t_exp))

    x0 = [0.30, 0.36, 0.26, 2500.0, 3.5]
    opt = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-7, maxiter=2000,
                                disp=False, adaptive=True))
    return opt.x, opt.fun


def fit_700():
    t_exp, J_exp = load_exp(700)
    alphas_g, t_g, J_g = load_grid(700)
    assert np.allclose(t_g, t_exp), "grid t must match experiment t"

    spline = RectBivariateSpline(alphas_g, t_g, J_g, kx=3, ky=3)

    def J_at(alpha_t):
        a = np.clip(alpha_t, alphas_g[0], alphas_g[-1])
        return spline(a, t_g, grid=False)

    # 700 C: initial shake clears bubble, then no shake.  Fit: α_max, τ_grow.
    # α(0) and α_res fixed at 0 (initial shake cleaned everything).
    def loss(x):
        alpha_max, tau_grow = x
        if not (0.0 <= alpha_max <= 0.7): return 1e6
        if not (200 <= tau_grow <= 3e4): return 1e6
        alpha_t = alpha_history(t_exp, 0.0, alpha_max, 0.0,
                                tau_grow, 300.0, [(0.0, 0.08 * 3600.0)])
        Jm = J_at(alpha_t)
        scale = max(np.max(np.abs(J_exp)), 1e-12)
        return float(np.sum((Jm - J_exp) ** 2) / (scale * scale) / len(t_exp))

    x0 = [0.44, 2500.0]
    opt = minimize(loss, x0, method="Nelder-Mead",
                   options=dict(xatol=1e-3, fatol=1e-7, maxiter=2000,
                                disp=False, adaptive=True))
    return opt.x, opt.fun


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("Fitting dynamic α(t) — 650 °C")
    print("=" * 70)
    x650, l650 = fit_650()
    a0, amax, ares, tg, tson = x650
    print(f"  loss={l650:.4e}")
    print(f"  α(0)      = {a0:.3f}")
    print(f"  α_max     = {amax:.3f}")
    print(f"  α_res     = {ares:.3f}")
    print(f"  τ_grow    = {tg:.0f} s  (≈ {tg/60:.1f} min)")
    print(f"  shake on  = {tson:.2f} h  -> off at 6.00 h")

    print("\n" + "=" * 70)
    print("Fitting dynamic α(t) — 700 °C")
    print("=" * 70)
    x700, l700 = fit_700()
    amax_700, tg_700 = x700
    print(f"  loss={l700:.4e}")
    print(f"  α(0)      = 0.000  (cleaned by initial shake)")
    print(f"  α_max     = {amax_700:.3f}")
    print(f"  α_res     = 0.000")
    print(f"  τ_grow    = {tg_700:.0f} s  (≈ {tg_700/60:.1f} min)")
    print(f"  shake     = 0 - 0.08 h only")

    # ── reconstruct curves for plotting ──
    t_exp_650, J_exp_650 = load_exp(650)
    alphas_g, t_g, J_g = load_grid(650)
    spline_650 = RectBivariateSpline(alphas_g, t_g, J_g, kx=3, ky=3)
    intervals_650 = [(tson * 3600, 6.0 * 3600)]
    alpha_t_650 = alpha_history(t_exp_650, a0, amax, ares, tg, 300.0, intervals_650)
    J_model_650 = spline_650(np.clip(alpha_t_650, alphas_g[0], alphas_g[-1]),
                              t_g, grid=False)

    t_exp_700, J_exp_700 = load_exp(700)
    alphas_g7, t_g7, J_g7 = load_grid(700)
    spline_700 = RectBivariateSpline(alphas_g7, t_g7, J_g7, kx=3, ky=3)
    alpha_t_700 = alpha_history(t_exp_700, 0.0, amax_700, 0.0, tg_700,
                                  300.0, [(0.0, 0.08 * 3600.0)])
    J_model_700 = spline_700(np.clip(alpha_t_700, alphas_g7[0], alphas_g7[-1]),
                              t_g7, grid=False)

    # ── plot: J(t) and α(t) for both temperatures ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")

    ax = axes[0, 0]
    ax.plot(t_exp_650 / 3600, J_exp_650 * 1e6, "o", ms=2.6, color="0.4",
            label="experiment")
    ax.plot(t_exp_650 / 3600, J_model_650 * 1e6, "-", color="#d62728",
            lw=1.7, label="2D model with α(t)")
    ax.axvspan(tson, 6.0, color="orange", alpha=0.12, label="shaking")
    ax.set_ylabel(r"Flux  [$10^{-6}$ mol H$_2$/m$^2$/s]")
    ax.set_title(f"650 °C   "
                 f"α_max={amax:.2f}, α_res={ares:.2f}, "
                 f"τ_grow={tg/60:.0f} min")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_exp_650 / 3600, alpha_t_650, color="#2ca02c", lw=1.6)
    ax.axvspan(tson, 6.0, color="orange", alpha=0.12)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"$\alpha(t)$  (coverage)")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_exp_700 / 3600, J_exp_700 * 1e6, "o", ms=2.6, color="0.4",
            label="experiment")
    ax.plot(t_exp_700 / 3600, J_model_700 * 1e6, "-", color="#d62728",
            lw=1.7, label="2D model with α(t)")
    ax.axvspan(0.0, 0.08, color="orange", alpha=0.12, label="initial shake")
    ax.set_title(f"700 °C   "
                 f"α_max={amax_700:.2f}, τ_grow={tg_700/60:.0f} min")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t_exp_700 / 3600, alpha_t_700, color="#2ca02c", lw=1.6)
    ax.axvspan(0.0, 0.08, color="orange", alpha=0.12)
    ax.set_xlabel("Time [h]")
    ax.set_ylim(-0.02, 1.0)
    ax.grid(alpha=0.3)

    fig.suptitle("2D axisymmetric model with time-dependent α(t)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dynamic_alpha_fit.png", dpi=180)
    print(f"\nsaved {OUT_DIR / 'dynamic_alpha_fit.png'}")

    # ── save params ──
    with (OUT_DIR / "params.txt").open("w") as f:
        f.write("# Fitted α(t) ODE parameters (2D axisymmetric model)\n")
        f.write("# τ_shake fixed at 300 s.\n\n")
        f.write("# 650 °C\n")
        f.write(f"alpha0_650     = {a0:.4f}\n")
        f.write(f"alpha_max_650  = {amax:.4f}\n")
        f.write(f"alpha_res_650  = {ares:.4f}\n")
        f.write(f"tau_grow_650   = {tg:.1f}\n")
        f.write(f"shake_on_650_h = {tson:.3f}\n")
        f.write(f"loss_650       = {l650:.4e}\n\n")
        f.write("# 700 °C  (α(0) and α_res both 0 by initial-shake assumption)\n")
        f.write(f"alpha_max_700  = {amax_700:.4f}\n")
        f.write(f"tau_grow_700   = {tg_700:.1f}\n")
        f.write(f"loss_700       = {l700:.4e}\n")
    print(f"saved {OUT_DIR / 'params.txt'}")


if __name__ == "__main__":
    main()
