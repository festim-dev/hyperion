"""Lumped 1D bubble-coverage model for the FLiBe/Ni permeation cell.

Physical picture
----------------
- A FLiBe layer (Henry's law) sits on top of a Ni membrane (Sievert's law).
- H2 dissolves into FLiBe at the upstream face, diffuses through FLiBe, crosses
  to Ni at the FLiBe/Ni interface, and desorbs at the downstream face.
- Bubbles nucleate at the FLiBe/Ni interface (H supersaturates there before
  being absorbed by Ni). They cover a fraction `alpha` of the interface and
  block permeation in those areas.
- Shaking dislodges bubbles -> alpha -> 0.

Reduced model
-------------
The FLiBe layer is resolved in 1D (finite volume).  Ni is collapsed into a
local impedance at the FLiBe/Ni interface (Ni diffusion is fast: tau ~ L_Ni^2 /
D_Ni << experiment).

State:
  C(x,t)  : H concentration in FLiBe, x in [0, L_F]
  alpha(t): bubble coverage at x = L_F

Eqs:
  dC/dt = D_F d^2C/dx^2                                  (FLiBe diffusion)
  C(0,t)   = K_H(T) * P_up                               (upstream, Henry)
  -D_F dC/dx|_{L_F} = (1-alpha) * Phi_Ni * sqrt(P_int)   (downstream, Sievert through Ni)
       with P_int = max(C(L_F)/K_H, 0)
  dalpha/dt = k_g(T) * (alpha_max - alpha) * (1-s(t)) - k_s * alpha * s(t)

Measured downstream flux is the Sievert flux above (== uncovered fraction * Ni
permeation).

The parameters K_H(T), D_F(T), Phi_Ni(T) come from htm-style Arrhenius fits;
bubble parameters (alpha_max, k_g, k_s) are fit to the 650/700 C curves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

KB_EV = 8.617333262e-5  # eV/K
R_GAS = 8.314462618     # J/mol/K
N_A = 6.02214076e23     # 1/mol


# ---------------------------------------------------------------------------
# Material properties (Arrhenius forms).  Values from bubble_1d.py + htm.
#
# All fluxes / concentrations in this model are in mol H2.
# The htm-style pre-exponentials supplied in bubble_1d.py are in *atoms H* per
# m / s / Pa (single H, not H2).  We convert with the factor 1/(2 * N_A).
# ---------------------------------------------------------------------------
_ATOMS_TO_MOLH2 = 1.0 / (2.0 * N_A)


@dataclass
class Materials:
    # FLiBe diffusivity (Henry layer)
    D_F0: float = 2.5e-7       # m^2/s pre-exp
    E_DF: float = 0.24         # eV

    # FLiBe permeability:  Phi_F = D_F * K_H  (mol H2/m/s/Pa)
    # bubble_1d.py used pre_exp=4.158e13 atoms/m/s/Pa  ->  mol H2/m/s/Pa:
    Phi_F0: float = 41587400565660.95 * _ATOMS_TO_MOLH2  # ~3.45e-11
    E_PhiF: float = 0.4655730255084721

    # Ni permeability (Sievert)  Phi_Ni  (mol H2/m/s/Pa^0.5)
    # Calibrated so that, in series with the FLiBe layer above, the clean
    # (bubble-free) steady-state at 650 C reaches ~2.0e-6 mol H2/m^2/s and at
    # 700 C reaches ~2.7e-6 mol H2/m^2/s -- consistent with the data once the
    # bubble layer is removed by shaking.  E_PhiNi taken near Reiter (1985).
    Phi_Ni0: float = 2.5e-8    # mol H2/m/s/Pa^0.5
    E_PhiNi: float = 0.547     # eV

    L_Ni: float = 0.002032     # m
    # FLiBe thickness at T (linear interp around the measured values)
    L_F_by_T: dict = field(default_factory=lambda: {
        500.0: 0.005139858,
        550.0: 0.005194021,
        600.0: 0.005249337,
        650.0: 0.005305845,
        700.0: 0.005363582,
    })

    def D_F(self, T_K: float) -> float:
        return self.D_F0 * math.exp(-self.E_DF / (KB_EV * T_K))

    def Phi_F(self, T_K: float) -> float:
        return self.Phi_F0 * math.exp(-self.E_PhiF / (KB_EV * T_K))

    def K_H(self, T_K: float) -> float:
        """Henry solubility K_H = Phi_F / D_F  [mol/m^3/Pa]."""
        return self.Phi_F(T_K) / self.D_F(T_K)

    def Phi_Ni(self, T_K: float) -> float:
        return self.Phi_Ni0 * math.exp(-self.E_PhiNi / (KB_EV * T_K))

    def L_F(self, T_C: float) -> float:
        Ts = np.array(sorted(self.L_F_by_T.keys()))
        Ls = np.array([self.L_F_by_T[t] for t in Ts])
        return float(np.interp(T_C, Ts, Ls))


# ---------------------------------------------------------------------------
# Bubble dynamics
# ---------------------------------------------------------------------------
@dataclass
class BubbleParams:
    alpha_max: float = 0.85    # max coverage when bubbles fully developed
    tau_grow: float = 6000.0   # s, e-folding time for bubble growth (no shake)
    tau_shake: float = 300.0   # s, e-folding time for bubble removal under shake
    alpha0: float = 0.0
    # Residual coverage that shaking *cannot* remove (e.g. bubbles trapped in
    # surface roughness).  Shaking drives alpha -> alpha_residual.
    alpha_residual: float = 0.0


def make_shake_schedule(intervals_h: Sequence[tuple[float, float]]) -> Callable[[float], float]:
    """Return s(t_seconds) -> 1 if t falls in any (t_on, t_off) hour-interval."""
    iv = [(a * 3600.0, b * 3600.0) for a, b in intervals_h]

    def s(t: float) -> float:
        for a, b in iv:
            if a <= t < b:
                return 1.0
        return 0.0
    return s


# ---------------------------------------------------------------------------
# Solver: explicit FV in space + explicit Euler in time (small dt, robust).
# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    t: np.ndarray            # s
    flux: np.ndarray         # mol/m^2/s  (downstream, effective)
    alpha: np.ndarray
    C_interface: np.ndarray  # H conc in FLiBe at the FLiBe/Ni interface
    J_ss_clean: float        # steady-state flux if no bubble (alpha=0)


def simulate(
    T_C: float,
    P_up: float,
    t_end_s: float,
    mat: Materials,
    bub: BubbleParams,
    shake: Callable[[float], float],
    n_x: int = 40,
    n_t: int | None = None,
    dt_target: float = 30.0,
):
    """Implicit (Backward-Euler) diffusion solver + explicit ODE on alpha.

    The downstream BC is non-linear (sqrt), so we linearise about C[-1] each
    step (one Newton step), which is plenty for the modest sensitivity here.
    """
    T_K = T_C + 273.15
    L_F = mat.L_F(T_C)
    D_F = mat.D_F(T_K)
    K_H = mat.K_H(T_K)
    Phi_Ni = mat.Phi_Ni(T_K)
    g_Ni = Phi_Ni / mat.L_Ni  # Ni permeance, mol/m^2/s/Pa^0.5

    dx = L_F / n_x
    if n_t is None:
        n_t = max(int(math.ceil(t_end_s / dt_target)), 50)
    dt = t_end_s / n_t

    N = n_x + 1
    C = np.zeros(N)
    C_up = K_H * P_up
    C[0] = C_up

    # output every step (already coarse, n_t ~ 1000)
    ts, fluxes, alphas, C_ints = [], [], [], []

    alpha = bub.alpha0
    r = D_F * dt / (dx * dx)

    # Pre-build tridiagonal coefficients for interior nodes (constant in time).
    # Backward Euler for diffusion: -r C_{i-1} + (1+2r) C_i - r C_{i+1} = C_i^old
    # Nodes 0 and N-1 are handled with boundary equations:
    #   row 0: C_0 = C_up
    #   row N-1: see below (Newton-linearised sqrt BC)
    main = np.full(N, 1.0 + 2.0 * r)
    lower = np.full(N - 1, -r)
    upper = np.full(N - 1, -r)

    # Override boundary rows
    main[0] = 1.0
    upper[0] = 0.0

    # For the downstream Neumann/Sievert BC at i=N-1:
    #   -D_F (C_N - C_{N-1})/dx = (1-alpha) g_Ni sqrt(C_N/K_H)
    #   => C_N - C_{N-1} + (dx/D_F)(1-alpha) g_Ni sqrt(C_N/K_H) = 0
    # Linearise sqrt(C_N) around C* (previous value):
    #     sqrt(C_N) ≈ sqrt(C*) + (C_N - C*)/(2 sqrt(C*))
    # giving
    #     C_N (1 + b/(2 sqrt(C*))) - C_{N-1} = -b * sqrt(C*)/2
    # where b = (dx/D_F)(1-alpha) g_Ni / sqrt(K_H)

    inv_sqrt_KH = 1.0 / math.sqrt(K_H)

    from numpy.linalg import solve as _solve  # not used, we use thomas
    def thomas(a, b, c, d):
        # Solve tri-diagonal system: a (sub), b (diag), c (super), d (rhs)
        n = len(b)
        cc = np.empty(n - 1)
        dd = np.empty(n)
        cc[0] = c[0] / b[0]
        dd[0] = d[0] / b[0]
        for i in range(1, n):
            m = b[i] - a[i - 1] * (cc[i - 1] if i - 1 < n - 1 else 0.0)
            if i < n - 1:
                cc[i] = c[i] / m
            dd[i] = (d[i] - a[i - 1] * dd[i - 1]) / m
        x = np.empty(n)
        x[-1] = dd[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dd[i] - cc[i] * x[i + 1]
        return x

    out_every = max(1, n_t // 1000)

    for k in range(n_t + 1):
        # diagnostics
        P_int = max(C[-1] / K_H, 0.0)
        J_eff = (1.0 - alpha) * g_Ni * math.sqrt(P_int)

        if k % out_every == 0 or k == n_t:
            ts.append(k * dt)
            fluxes.append(J_eff)
            alphas.append(alpha)
            C_ints.append(C[-1])

        if k == n_t:
            break

        # ---- build rhs ----
        rhs = C.copy()
        rhs[0] = C_up

        # Linearise sqrt BC about current C[-1]
        C_star = max(C[-1], 1e-30)
        b_coef = (dx / D_F) * (1.0 - alpha) * g_Ni * inv_sqrt_KH
        main_last = 1.0 + 0.5 * b_coef / math.sqrt(C_star)
        rhs_last = -0.5 * b_coef * math.sqrt(C_star)
        # row N-1: -C_{N-2} + main_last * C_{N-1} = rhs_last
        # so lower[N-2] = -1, main[N-1] = main_last
        main_arr = main.copy()
        lower_arr = lower.copy()
        upper_arr = upper.copy()
        main_arr[-1] = main_last
        lower_arr[-1] = -1.0  # coefficient at row N-1, column N-2
        # upper_arr[N-2] stays = -r: that is the C_{N-2} -> C_{N-1} coupling in
        # row N-2 (standard interior diffusion row), which we must NOT wipe.
        rhs[-1] = rhs_last

        C = thomas(lower_arr, main_arr, upper_arr, rhs)
        if C[-1] < 0.0:
            C[-1] = 0.0

        # bubble coverage update (explicit Euler with sub-stepping if needed)
        s = shake(k * dt)
        d_alpha = (
            (1.0 - s) * (bub.alpha_max - alpha) / bub.tau_grow
            - s * (alpha - bub.alpha_residual) / bub.tau_shake
        )
        alpha = max(0.0, min(1.0, alpha + dt * d_alpha))

    # steady-state clean flux (alpha=0) — closed-form for series Henry+Sievert:
    # J_ss = g_Ni * sqrt(P_int_ss), and J_ss = Phi_F/L_F * (P_up - P_int_ss).
    # Let phi_F = Phi_F/L_F, define x = sqrt(P_int_ss):
    #   phi_F*(P_up - x^2) = g_Ni*x  -> phi_F*x^2 + g_Ni*x - phi_F*P_up = 0
    phi_F = mat.Phi_F(T_K) / L_F
    a, b, c = phi_F, g_Ni, -phi_F * P_up
    disc = b * b - 4 * a * c
    x_ss = (-b + math.sqrt(disc)) / (2 * a)
    J_ss_clean = g_Ni * x_ss

    return SimResult(
        t=np.asarray(ts),
        flux=np.asarray(fluxes),
        alpha=np.asarray(alphas),
        C_interface=np.asarray(C_ints),
        J_ss_clean=J_ss_clean,
    )


# ---------------------------------------------------------------------------
# Pre-baked shake schedules for the two experiments
# ---------------------------------------------------------------------------
# 650 C: re-start at t=0; shaking starts after the first plateau (~2.7 h),
# stops at t=6 h.
SHAKE_650 = make_shake_schedule([(2.7, 6.0)])

# 700 C: brief shake at the very beginning to remove leftover 650 C bubbles
# (clears alpha to ~0 over the first ~5 min), no shaking afterwards.
SHAKE_700 = make_shake_schedule([(0.0, 0.08)])  # 0..5 min


if __name__ == "__main__":
    # quick sanity
    mat = Materials()
    bub = BubbleParams()
    res = simulate(T_C=650.0, P_up=1.32e5, t_end_s=7.5 * 3600,
                   mat=mat, bub=bub, shake=SHAKE_650)
    print(f"650C  J_ss_clean = {res.J_ss_clean:.3e}  final J = {res.flux[-1]:.3e}")
    res = simulate(T_C=700.0, P_up=1.32e5, t_end_s=6.2 * 3600,
                   mat=mat, bub=bub, shake=SHAKE_700)
    print(f"700C  J_ss_clean = {res.J_ss_clean:.3e}  final J = {res.flux[-1]:.3e}")
