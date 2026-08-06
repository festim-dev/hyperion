"""Quantitative error of the quasi-steady inversion, calibrated on
synthetic data from an EXACT transient 1-D model.

The 2-D grid cannot generate truly dynamic synthetic data (alpha is static
per mesh), so we use a 1-D analogue with identical physics structure and
matched timescales: a FLiBe slab (thickness L, diffusivity D at 650 C) with
the Henry concentration c_top at the top and a finite interfacial transfer
coefficient k(t) at the bottom (Ni treated as a perfect sink, justified by
its ~2e4 larger conductance):

    dC/dt = D d2C/dx2,   C(0,t) = c_top,   -D dC/dx|_L = k(t) C(L,t) = J(t)

Crank-Nicolson finite differences give the EXACT flux J(t) for any
prescribed k(t).  The quasi-steady inversion of the paper,

    k_rec(t) = 1 / ( c_top/J(t) - c_top/J0(t) ),

(J0 = run with C(L)=0, i.e. k = infinity) is then applied to the synthetic
flux, and k_rec is compared against the known truth.

Experiments:
  (E1) constant k          -> recovery error after the bulk transient
  (E2) exponential k_a->k_b transitions at t0 = 3 h with time constants
       tau_true in {0 (step), 5, 15, 30, 60, 120} min
       -> max recovery error and APPARENT time constant tau_app vs tau_true
  (E3) the paper's case: which tau_true reproduces the observed
       tau_app = 59 min at 650 C?

Output: results_kint_inversion/qs_error_1d.png + printed table.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

OUT_DIR = Path(__file__).parent / "results_kint_inversion"

# 650 C FLiBe parameters (fitted baseline)
D = 2.33e-9          # m^2/s
L = 5.30e-3          # m
C_TOP = 7.814        # mol H2 / m^3
K_A = 5.2e-7         # m/s  (pre-shake k_int at 650 C)
K_B = 8.4e-7         # m/s  (post-shake)

NX = 201
DT = 5.0             # s
T_END = 8 * 3600.0
T0 = 3 * 3600.0      # transition start


def solve(k_of_t):
    """Crank-Nicolson on [0, L]; Robin BC J = k(t) C(L); returns t, J."""
    x = np.linspace(0, L, NX)
    dx = x[1] - x[0]
    r = D * DT / (2 * dx * dx)
    n_steps = int(T_END / DT)
    C = np.zeros(NX)
    C[0] = C_TOP

    # interior matrix (tridiagonal), Robin row updated each step
    t_out, J_out = [], []
    lower = np.full(NX, -r)
    diag = np.full(NX, 1 + 2 * r)
    upper = np.full(NX, -r)
    # Dirichlet at x=0
    diag[0], upper[0] = 1.0, 0.0

    from scipy.linalg import solve_banded
    for n in range(1, n_steps + 1):
        t = n * DT
        k = k_of_t(t)
        # ghost-node Robin at x=L:  -D (C_g - C_{N-2})/(2dx) = k C_{N-1}
        # => C_g = C_{N-2} - 2 dx k / D * C_{N-1}
        beta = 2 * dx * k / D
        rhs = C.copy()
        rhs[1:-1] = C[1:-1] + r * (C[2:] - 2 * C[1:-1] + C[:-2])
        rhs[0] = C_TOP
        # last row: CN with ghost node on both sides
        rhs[-1] = C[-1] + r * (2 * C[-2] - (2 + beta) * C[-1])
        ab = np.zeros((3, NX))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        ab[1, -1] = 1 + r * (2 + beta)
        ab[2, -2] = -2 * r
        C = solve_banded((1, 1), ab, rhs)
        if n % 36 == 0:                      # one sample per 3 min
            t_out.append(t)
            J_out.append(k * C[-1])
    return np.asarray(t_out), np.asarray(J_out)


def k_const(k):
    return lambda t: k


def k_exp(k_a, k_b, t0, tau):
    if tau <= 0:
        return lambda t: k_a if t < t0 else k_b
    return lambda t: k_a if t < t0 else k_b + (k_a - k_b) * np.exp(-(t - t0) / tau)


def invert(J, J0):
    with np.errstate(divide="ignore"):
        R = C_TOP / J - C_TOP / J0
    return np.where(R > 1e3, 1.0 / R, np.nan)


def main():
    print("solving J0 (k = infinity reference) ...")
    t, J0 = solve(k_const(1.0))            # k=1 m/s is effectively infinite
    valid = J0 > 0.8 * J0[-1]
    R_bulk = C_TOP / J0[-1]
    print(f"  1-D R_bulk = {R_bulk:.3e} s/m (2-D value 2.96e6); "
          f"valid window opens at {t[np.argmax(valid)]/3600:.2f} h")

    # (E1) constant k
    print("\n(E1) constant k: relative recovery error in valid window")
    for k in (K_A, K_B, 3.0e-7):
        _, J = solve(k_const(k))
        k_rec = invert(J, J0)
        err = np.nanmedian(np.abs(k_rec[valid] - k) / k)
        print(f"  k = {k:.2e}: median |err| = {100*err:.2f} %")

    # (E2) transitions
    taus_min = [0, 5, 15, 30, 60, 120]
    results = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(taus_min)))
    print("\n(E2) k_a -> k_b transitions at t0 = 3 h")
    print(f"{'tau_true':>9} {'max |err| after t0':>19} {'t63_app':>9} "
          f"{'tau_fit':>9} {'overshoot':>10}")
    target = K_B + (K_A - K_B) / np.e
    for c, tau_m in zip(cmap, taus_min):
        kf = k_exp(K_A, K_B, T0, tau_m * 60.0)
        _, J = solve(kf)
        k_rec = invert(J, J0)
        k_true = np.array([kf(tt) for tt in t])
        sel = (t > T0) & valid
        err_max = np.nanmax(np.abs(k_rec[sel] - k_true[sel]) / k_true[sel])

        # metric 1: 63 %-crossing time of the recovered curve
        tt_h, kk = t[sel] / 3600, k_rec[sel]
        t63 = (tt_h[np.argmax(kk >= target)] - T0 / 3600) * 60
        # metric 2: free 3-parameter exponential fit (same procedure as on
        # the real data) — can absorb a mild overshoot in the endpoints
        def expo(tt, k_inf, k_0, tau_h):
            return k_inf + (k_0 - k_inf) * np.exp(-(tt - T0 / 3600) / tau_h)
        try:
            p, _ = curve_fit(expo, tt_h, kk * 1e7,   # fit in 1e-7 m/s units
                             p0=[K_B * 1e7, K_A * 1e7, 1.0],
                             bounds=([0, 0, 0.02], [100, 100, 10]))
            tau_fit = p[2] * 60
        except Exception:
            tau_fit = np.nan
        overshoot = (np.nanmax(kk) - K_B) / K_B
        results.append((tau_m, err_max, t63, tau_fit, overshoot))
        print(f"{tau_m:>6.0f} min {100*err_max:>17.1f} % {t63:>7.0f} min "
              f"{tau_fit:>7.0f} min {100*overshoot:>8.1f} %")
        ax1.plot(tt_h, kk * 1e7, ".", ms=2.5, color=c,
                 label=fr"$\tau_{{\rm true}}$ = {tau_m:.0f} min")
        ax1.plot(t / 3600, k_true * 1e7, "-", lw=0.8, color=c, alpha=0.5)

    ax1.set_xlim(2.5, 8)
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel(r"$k_{\rm int}$ [$10^{-7}$ m/s]")
    ax1.set_title("true (lines) vs quasi-steady recovery (dots)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    taus = np.array([r[0] for r in results])
    t63s = np.array([r[2] for r in results])
    errs = np.array([r[1] for r in results]) * 100
    ax2.plot(taus, t63s, "o-", color="#1f77b4",
             label=r"apparent 63 %-crossing time")
    ax2.plot([0, 120], [0, 120], "--", color="0.6", lw=1,
             label="no-distortion line")
    ax2.axhline(59, color="#d62728", ls=":", lw=1.4,
                label=r"observed value at 650 °C (59 min)")
    ax2.set_xlabel(r"true interface time constant $\tau_{\rm true}$ [min]")
    ax2.set_ylabel(r"apparent time constant [min]")
    axe = ax2.twinx()
    axe.plot(taus, errs, "s--", color="#2ca02c", alpha=0.7, ms=5)
    axe.set_ylabel(r"max $|k_{\rm rec}-k_{\rm true}|/k_{\rm true}$ [%]",
                   color="#2ca02c")
    axe.tick_params(axis="y", colors="#2ca02c")
    ax2.set_title("distortion of the quasi-steady inversion")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8, loc="upper left")

    # (E3) which tau_true reproduces the observed apparent 59 min?
    tau_true_59 = np.interp(59.0, t63s, taus)
    step_overshoot = results[0][4]
    print(f"\n(E3) apparent 59 min corresponds to tau_true ≈ "
          f"{tau_true_59:.0f} min")
    print(f"     a step (tau_true = 0) would overshoot by "
          f"{100*step_overshoot:.0f} % — absent in the data")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "qs_error_1d.png", dpi=180)
    print(f"saved {OUT_DIR / 'qs_error_1d.png'}")


if __name__ == "__main__":
    main()
