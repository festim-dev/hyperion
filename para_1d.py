"""
1D Ni-FLiBe permeation model for sim vs exp comparison.

FLiBe permeability is loaded from results/fitted_params.csv (para_swap_pure.py output).
Ni permeability is loaded from results/dry_run_phi_arrhenius_fits.txt (dry_run_fitting.py output).
All experimental data and diffusivities are imported from exp_data.py.

Outputs (saved to results/):
    all_results_1d.csv          -- J_sim vs J_exp per case/run/T
    permeabilities_used_1d.csv  -- FLiBe Arrhenius parameters used per case/run
    profiles/                   -- concentration profiles per run #commented out for now, but can be re-enabled if needed
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional

import festim as F
import h_transport_materials as htm
import numpy as np
from mpi4py import MPI

from exp_data import (
    D_flibe,
    D_nickel,
    normal_infinite,
    normal_transparent,
    normal_flux_err,
    swap_infinite,
    swap_transparent,
    swap_flux_err,
    load_ni_permeability,
    load_flibe_permeability,
)

# ── Constants & globals ───────────────────────────────────────────────────────

kB_eV = 8.617333262e-5  # Boltzmann constant [eV/K]

OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

_RANK0 = MPI.COMM_WORLD.rank == 0

# FLiBe thickness [m] per temperature [C], from experimental setup
L_FLIBE_BY_TEMP_C: Dict[float, float] = {
    500.0: 0.005139858,
    550.0: 0.005194021,
    600.0: 0.005249337,
    650.0: 0.005305845,
    700.0: 0.005363582,
}


def flibe_thickness_from_T_C(T_C: float) -> float:
    """Linearly interpolate FLiBe thickness [m] at T_C [C]."""
    temps = np.array(sorted(L_FLIBE_BY_TEMP_C.keys()))
    Ls = np.array([L_FLIBE_BY_TEMP_C[T] for T in temps])
    return float(np.interp(T_C, temps, Ls))


# ── Ni solubility from permeability ──────────────────────────────────────────


def _ni_solubility_for_bc(bc_type: str) -> htm.Solubility:
    """
    Load Ni permeability from dry_run_phi_arrhenius_fits.txt and derive
    solubility via S(T) = phi(T) / D(T), using the shared D_nickel from exp_data.

    bc_type: 'particle_flux_zero' (ideal coating) or 'sieverts' (uncoated).
    """
    ni_perm = load_ni_permeability()
    prm = ni_perm[bc_type]
    phi_0 = prm["phi_0"] * 6.02214076e23  # mol -> particles
    E_phi_eV = prm["E_phi_kJmol"] / 96.485332123  # kJ/mol -> eV
    perm = htm.Permeability(pre_exp=phi_0, act_energy=E_phi_eV, law="sievert")
    return htm.Solubility(
        S_0=perm.pre_exp / D_nickel.pre_exp,
        E_S=perm.act_energy - D_nickel.act_energy,
        law="sievert",
    )


# ── Cylindrical flux wrapper ──────────────────────────────────────────────────


class CylindricalFlux1D(F.SurfaceFlux):
    """
    Wraps FESTIM SurfaceFlux for a 1D model. The parent computes flux per unit
    area [H/m²/s]; this class multiplies by the actual cylinder cross-section
    area (pi * R²) to give total flux [H/s].
    """

    def __init__(self, field, surface, radius: float, filename=None):
        super().__init__(field=field, surface=surface, filename=filename)
        self.radius = radius

    def compute(self, u, ds, entity_maps=None):
        super().compute(u, ds, entity_maps)
        area = np.pi * self.radius**2
        self.value *= area
        if self.data:
            self.data[-1] = self.value
        else:
            self.data.append(self.value)


# ── Materials ─────────────────────────────────────────────────────────────────


def _make_materials_1d(
    K_S_nickel: htm.Solubility, permeability_flibe: htm.Permeability
):
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )
    mat_ni = F.Material(
        D_0=D_nickel.pre_exp.magnitude,
        E_D=D_nickel.act_energy.magnitude,
        K_S_0=K_S_nickel.pre_exp.magnitude,
        E_K_S=K_S_nickel.act_energy.magnitude,
        solubility_law="sievert",
    )
    mat_flibe = F.Material(
        D_0=D_flibe.pre_exp.magnitude,
        E_D=D_flibe.act_energy.magnitude,
        K_S_0=K_S_liquid.pre_exp.magnitude,
        E_K_S=K_S_liquid.act_energy.magnitude,
        solubility_law="henry",
    )
    return mat_ni, mat_flibe, K_S_liquid


# ── 1D model builder ─────────────────────────────────────────────────────────


def _make_model_1d(
    T_K: float,
    P_up: float,
    P_down: float,
    L_Ni: float,
    L_flibe: float,
    radius: float,
    K_S_nickel: htm.Solubility,
    permeability_flibe: htm.Permeability,
    penalty_term: float = 1e20,
):
    """
    1D geometry: [0, L_Ni] = Ni, [L_Ni, L_Ni + L_flibe] = FLiBe.
    Left BC: Sieverts on Ni; right BC: Henry on FLiBe surface.
    """
    mat_ni, mat_flibe, K_S_liquid = _make_materials_1d(K_S_nickel, permeability_flibe)

    x0, x1, x2 = 0.0, L_Ni, L_Ni + L_flibe
    vol_ni = F.VolumeSubdomain1D(id=1, borders=[x0, x1], material=mat_ni)
    vol_flibe = F.VolumeSubdomain1D(id=2, borders=[x1, x2], material=mat_flibe)
    left_surf = F.SurfaceSubdomain1D(id=1, x=x0)
    right_surf = F.SurfaceSubdomain1D(id=2, x=x2)

    model = F.HydrogenTransportProblemDiscontinuous()
    model.subdomains = [vol_ni, vol_flibe, left_surf, right_surf]
    model.surface_to_volume = {left_surf: vol_ni, right_surf: vol_flibe}
    model.interfaces = [
        F.Interface(id=3, subdomains=[vol_ni, vol_flibe], penalty_term=penalty_term)
    ]

    H = F.Species("H", subdomains=model.volume_subdomains)
    model.species = [H]
    model.temperature = T_K

    vertices = np.concatenate(
        [
            np.linspace(x0, x1, 60),
            np.linspace(x1, x2, 60)[1:],
        ]
    )
    model.mesh = F.Mesh1D(vertices)

    model.boundary_conditions = [
        F.SievertsBC(
            subdomain=left_surf,
            species=H,
            pressure=P_up,
            S_0=K_S_nickel.pre_exp.magnitude,
            E_S=K_S_nickel.act_energy.magnitude,
        ),
        F.HenrysBC(
            subdomain=right_surf,
            species=H,
            pressure=P_down,
            H_0=K_S_liquid.pre_exp.magnitude,
            E_H=K_S_liquid.act_energy.magnitude,
        ),
    ]

    flux_in = CylindricalFlux1D(field=H, surface=left_surf, radius=radius)
    flux_out = CylindricalFlux1D(field=H, surface=right_surf, radius=radius)
    prof_ni = F.Profile1DExport(field=H, subdomain=vol_ni)
    prof_flibe = F.Profile1DExport(field=H, subdomain=vol_flibe)
    model.exports = [flux_in, flux_out, prof_ni, prof_flibe]
    model.settings = F.Settings(transient=False, atol=1e10, rtol=1e-12)

    return model, flux_in, flux_out, prof_ni, prof_flibe


# ── Run one simulation ────────────────────────────────────────────────────────


def _run_once_1d(
    case_name: str,
    run_name: str,
    T_C: float,
    P_up: float,
    P_down: float,
    K_S_nickel: htm.Solubility,
    permeability_flibe: htm.Permeability,
    L_Ni: float,
    radius: float,
) -> tuple[float, float, float, float]:
    """Run one steady-state 1D solve. Returns (J_in, J_out, T_K, L_flibe)."""
    T_K = T_C + 273.15
    L_flibe = flibe_thickness_from_T_C(T_C)
    # stem = f"{case_name}_{run_name}_T{int(T_C)}C"

    model, flux_in, flux_out, prof_ni, prof_flibe = _make_model_1d(
        T_K=T_K,
        P_up=P_up,
        P_down=P_down,
        L_Ni=L_Ni,
        L_flibe=L_flibe,
        radius=radius,
        K_S_nickel=K_S_nickel,
        permeability_flibe=permeability_flibe,
    )
    model.initialise()
    model.run()

    J_in = float(flux_in.data[-1])
    J_out = float(flux_out.data[-1])

    # prof_dir = OUTDIR / "profiles"
    # prof_dir.mkdir(parents=True, exist_ok=True)
    # if prof_ni.data:
    #     np.savetxt(
    #         prof_dir / f"{stem}_ni.csv", np.array(prof_ni.data[-1]), delimiter=","
    #     )
    # if prof_flibe.data:
    #     np.savetxt(
    #         prof_dir / f"{stem}_flibe.csv", np.array(prof_flibe.data[-1]), delimiter=","
    #     )

    return J_in, J_out, T_K, L_flibe


# ── Batch runner ──────────────────────────────────────────────────────────────


def run_all_cases_1d(
    cases: Dict,
    T2K: Dict[float, float],
    flibe_perm_by_case_run: Dict,
    K_S_nickel_by_case: Dict[str, htm.Solubility],
    L_Ni: float = 0.002032,
    radius: float = 3.07 * 0.0254 / 2.0,
) -> List[dict]:
    """
    Run one 1D simulation per (case, temperature, run).

    flibe_perm_by_case_run : {(case_name, run_name): htm.Permeability}
        Loaded from results/fitted_params.csv via load_flibe_permeability().
    K_S_nickel_by_case : {case_name: htm.Solubility}
        Per-case Ni solubility derived from dry-run Arrhenius fits.
    """
    all_results: List[dict] = []

    for case_name, cfg in cases.items():
        K_S_nickel = K_S_nickel_by_case[case_name]
        out_mode = cfg.get("out_mode", "particle_flux_zero")

        for Tc, entry in cfg["table"].items():
            for run_name, cond in entry.get("runs", {}).items():
                perm_flibe = flibe_perm_by_case_run.get((case_name, run_name))
                if perm_flibe is None:
                    if _RANK0:
                        print(f"[skip] no permeability for ({case_name}, {run_name})")
                    continue

                J_in, J_out, T_K, L_flibe = _run_once_1d(
                    case_name=case_name,
                    run_name=run_name,
                    T_C=float(Tc),
                    P_up=float(cond["P_up"]),
                    P_down=float(cond["P_down"]),
                    K_S_nickel=K_S_nickel,
                    permeability_flibe=perm_flibe,
                    L_Ni=L_Ni,
                    radius=radius,
                )

                all_results.append(
                    dict(
                        case=case_name,
                        run=run_name,
                        T_C=float(Tc),
                        T_K=float(T_K),
                        P_up=float(cond["P_up"]),
                        P_down=float(cond["P_down"]),
                        P_gb=float(cond["P_gb"]) if "P_gb" in cond else None,
                        phi0=float(perm_flibe.pre_exp.magnitude),
                        E=float(perm_flibe.act_energy.magnitude),
                        law=perm_flibe.law,
                        J_in=J_in,
                        J_out=J_out,
                        J_sim=J_out,
                        J_exp=float(cond.get("J_exp", np.nan)),
                        ni_out_mode=out_mode,
                        K_S_0_Ni=float(K_S_nickel.pre_exp.magnitude),
                        E_S_Ni=float(K_S_nickel.act_energy.magnitude),
                    )
                )

                if _RANK0:
                    print(
                        f"[1D] {case_name:>18s} | T={float(Tc):5.1f}C | {run_name:>5s} | "
                        f"L_flibe={L_flibe:.5e} m | "
                        f"J_in={J_in:.3e} H/s | J_out={J_out:.3e} H/s"
                    )

    return all_results


# ── Experimental error lookup ─────────────────────────────────────────────────


def get_exp_error(case_name: str, temp: float, run_name: str) -> Optional[float]:
    """Return 1-sigma flux uncertainty (k=2 value divided by 2)."""
    all_err = {**normal_flux_err, **swap_flux_err}
    case = all_err.get(case_name)
    if case is None:
        return None
    raw = case.get(float(temp), {}).get("runs", {}).get(run_name)
    try:
        val = float(raw) / 2.0
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) and val > 0.0 else None


# ── CSV exports ───────────────────────────────────────────────────────────────


def save_results_1d(all_results: List[dict]) -> None:
    path = OUTDIR / "all_results_1d.csv"
    fields = [
        "case",
        "run",
        "T_C",
        "T_K",
        "P_up",
        "P_down",
        "P_gb",
        "phi0",
        "E",
        "law",
        "J_in",
        "J_out",
        "J_sim",
        "J_exp",
        "J_exp_err_1sigma",
        "ni_out_mode",
        "K_S_0_Ni",
        "E_S_Ni",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in all_results:
            j_err = get_exp_error(r["case"], r["T_C"], r["run"])
            w.writerow(
                {
                    **{k: r.get(k, "") for k in fields},
                    "P_gb": "" if r.get("P_gb") is None else r["P_gb"],
                    "J_exp_err_1sigma": "" if j_err is None else j_err,
                }
            )
    if _RANK0:
        print(f"[saved] {path}")


def save_permeabilities_used(flibe_perm_by_case_run: Dict) -> None:
    path = OUTDIR / "permeabilities_used_1d.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "run", "phi0", "E_eV", "law"])
        w.writeheader()
        for (case_name, run_name), perm in flibe_perm_by_case_run.items():
            w.writerow(
                dict(
                    case=case_name,
                    run=run_name,
                    phi0=float(perm.pre_exp.magnitude),
                    E_eV=float(perm.act_energy.magnitude),
                    law=perm.law,
                )
            )
    if _RANK0:
        print(f"[saved] {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    T2K = {Tc: Tc + 273.15 for Tc in [500.0, 550.0, 600.0, 650.0, 700.0]}

    cases = {
        "normal_infinite": {"table": normal_infinite, "out_mode": "particle_flux_zero"},
        "normal_transparent": {"table": normal_transparent, "out_mode": "sieverts"},
        "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }

    # FLiBe permeability from inversion output (results/fitted_params.csv)
    flibe_perm_by_case_run = load_flibe_permeability()

    # Ni solubility per case from dry-run Arrhenius fits
    K_S_nickel_by_case = {
        case_name: _ni_solubility_for_bc(cfg["out_mode"])
        for case_name, cfg in cases.items()
    }

    all_results = run_all_cases_1d(
        cases=cases,
        T2K=T2K,
        flibe_perm_by_case_run=flibe_perm_by_case_run,
        K_S_nickel_by_case=K_S_nickel_by_case,
    )

    if _RANK0:
        print("\n===== 1D Sim vs Exp =====")
        for r in sorted(all_results, key=lambda x: (x["case"], x["T_C"], x["run"])):
            P_T = r["phi0"] * np.exp(-r["E"] / (kB_eV * r["T_K"]))
            print(
                f"{r['case']:>20s} | T={r['T_C']:5.1f}C | {r['run']:>5s} | "
                f"phi(T)={P_T:.3e} | J_sim={r['J_sim']:.3e} | J_exp={r['J_exp']:.3e}"
            )

    save_results_1d(all_results)
    save_permeabilities_used(flibe_perm_by_case_run)

    if _RANK0:
        print(f"\nAll outputs saved to: {OUTDIR.resolve()}")
