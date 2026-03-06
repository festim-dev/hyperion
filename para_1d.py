import festim as F
import numpy as np
import h_transport_materials as htm
from typing import Dict, List, Optional
from mpi4py import MPI
from pathlib import Path
import csv

# ------------------------------ Globals ------------------------------
_RANK0 = MPI.COMM_WORLD.rank == 0

# FLiBe thickness [m] as a function of temperature [°C] (from experimental setup)
L_FLIBE_BY_TEMP_C: Dict[float, float] = {
    500.0: 0.005139858,
    550.0: 0.005194021,
    600.0: 0.005249337,
    650.0: 0.005305845,
    700.0: 0.005363582,
}
kB_eV = 8.617333262e-5  # Boltzmann constant in eV/K


def flibe_thickness_from_T_C(T_C: float) -> float:
    """
    Return FLiBe thickness [m] at given temperature [°C].

    If T_C is exactly in the table, use the value.
    Otherwise, linearly interpolate between neighbouring points.
    """
    temps = np.array(sorted(L_FLIBE_BY_TEMP_C.keys()))
    Ls = np.array([L_FLIBE_BY_TEMP_C[T] for T in temps])
    return float(np.interp(T_C, temps, Ls))


# ------------------------------ 1D cylindrical flux ------------------------------
class CylindricalFlux1D(F.SurfaceFlux):
    """
    Use FESTIM SurfaceFlux on a 1D model to compute j [H/m²/s],
    then multiply by the real cylinder area A = pi * R² to obtain
    a total flux in [H/s].
    """

    def __init__(self, field, surface, radius: float, filename: str | None = None):
        super().__init__(field=field, surface=surface, filename=filename)
        self.radius = radius

    def compute(self, u, ds, entity_maps=None):
        # First let the parent class compute the surface flux (per m²)
        super().compute(u, ds, entity_maps)
        # Convert to a total flux [H/s]
        area = np.pi * self.radius**2
        self.value *= (
            area / 2.0
        )  # for the symmetric 1D, we only consider half for comparison with 2D symmetric
        if self.data:
            self.data[-1] = self.value
        else:
            self.data.append(self.value)


# ------------------------------ Materials & 1D model ------------------------------
def make_materials_1d(
    D_nickel: htm.Diffusivity,
    D_flibe: htm.Diffusivity,
    K_S_nickel: htm.Solubility,
    permeability_flibe: htm.Permeability,
):
    """
    Build FESTIM Material objects for Ni (sievert) and FLiBe (henry),
    """
    # Ni
    mat_ni = F.Material(
        D_0=D_nickel.pre_exp.magnitude,
        E_D=D_nickel.act_energy.magnitude,
        K_S_0=K_S_nickel.pre_exp.magnitude,
        E_K_S=K_S_nickel.act_energy.magnitude,
        solubility_law="sievert",
    )

    # FLiBe: reconstruct solubility from permeability and diffusivity
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )

    mat_flibe = F.Material(
        D_0=D_flibe.pre_exp.magnitude,
        E_D=D_flibe.act_energy.magnitude,
        K_S_0=K_S_liquid.pre_exp.magnitude,
        E_K_S=K_S_liquid.act_energy.magnitude,
        solubility_law="henry",
    )

    return mat_ni, mat_flibe, K_S_liquid


def make_ni_flibe_1d_model(
    T_K: float,
    P_up: float,
    P_down: float,
    L_Ni: float,
    L_flibe: float,
    radius: float,
    D_nickel: htm.Diffusivity,
    D_flibe: htm.Diffusivity,
    K_S_nickel: htm.Solubility,
    permeability_flibe: htm.Permeability,
    penalty_term: float = 1e20,
    stem: str = "",
):
    """
    1D geometry: [0, L_Ni] = Ni, [L_Ni, L_Ni+L_flibe] = FLiBe
    Left boundary: Sievert on Ni
    Right boundary: Henry on FLiBe surface

    Returns:
        model, flux_out, flux_in
    where flux_out / flux_in are CylindricalFlux1D objects (H/s).
    """

    # ---- materials ----
    mat_ni, mat_flibe, K_S_liquid = make_materials_1d(
        D_nickel=D_nickel,
        D_flibe=D_flibe,
        K_S_nickel=K_S_nickel,
        permeability_flibe=permeability_flibe,
    )
    K_S_0_Ni = K_S_nickel.pre_exp.magnitude
    E_S_Ni = K_S_nickel.act_energy.magnitude
    H_0_liq = K_S_liquid.pre_exp.magnitude
    E_H_liq = K_S_liquid.act_energy.magnitude

    # ---- 1D subdomains ----
    x0 = 0.0
    x1 = L_Ni
    x2 = L_Ni + L_flibe

    vol_ni = F.VolumeSubdomain1D(id=1, borders=[x0, x1], material=mat_ni)
    vol_flibe = F.VolumeSubdomain1D(id=2, borders=[x1, x2], material=mat_flibe)

    boundary_left = F.SurfaceSubdomain1D(id=1, x=x0)
    boundary_right = F.SurfaceSubdomain1D(id=2, x=x2)

    model = F.HydrogenTransportProblemDiscontinuous()
    model.subdomains = [vol_ni, vol_flibe, boundary_left, boundary_right]

    # Map boundaries to volumes
    model.surface_to_volume = {
        boundary_left: vol_ni,
        boundary_right: vol_flibe,
    }

    # Ni–FLiBe interface with penalty method
    model.interfaces = [
        F.Interface(id=3, subdomains=[vol_ni, vol_flibe], penalty_term=penalty_term)
    ]

    # ---- species ----
    H = F.Species("H", subdomains=model.volume_subdomains)
    model.species = [H]

    # ---- mesh ----
    vertices_ni = np.linspace(x0, x1, num=60)
    vertices_flibe = np.linspace(x1, x2, num=60)[1:]
    vertices = np.concatenate([vertices_ni, vertices_flibe])
    model.mesh = F.Mesh1D(vertices)

    # ---- temperature ----
    model.temperature = T_K

    # ---- boundary conditions ----
    left_bc = F.SievertsBC(
        subdomain=boundary_left,
        species=H,
        pressure=P_up,
        S_0=K_S_0_Ni,
        E_S=E_S_Ni,
    )

    right_bc = F.HenrysBC(
        subdomain=boundary_right,
        species=H,
        pressure=P_down,
        H_0=H_0_liq,
        E_H=E_H_liq,
    )

    model.boundary_conditions = [left_bc, right_bc]

    # ---- flux exports in H/s (cylindrical) ----
    flux_out = CylindricalFlux1D(field=H, surface=boundary_right, radius=radius)
    flux_in = CylindricalFlux1D(field=H, surface=boundary_left, radius=radius)

    prof_ni = F.Profile1DExport(
        field=H,
        subdomain=vol_ni,
    )
    prof_flibe = F.Profile1DExport(
        field=H,
        subdomain=vol_flibe,
    )

    model.exports = [flux_in, flux_out, prof_ni, prof_flibe]

    # ---- steady-state solve ----
    model.settings = F.Settings(transient=False, atol=1e10, rtol=1e-12)

    return model, flux_out, flux_in, prof_flibe, prof_ni


# ------------------------------ Permeability helpers ------------------------------
def permability_by_case_name(
    case_name: str,
    run_name: str = "",
) -> htm.Permeability:
    """
    Baseline permeability laws, identical to the 2D script.
    """
    if case_name.startswith("normal"):
        if case_name == "normal_infinite":
            return htm.Permeability(
                pre_exp=3169563873541.4814, act_energy=0.400882080795477, law="henry"
            )
        elif case_name == "normal_transparent":
            return htm.Permeability(
                pre_exp=316467738703083.4, act_energy=0.6746935001483895, law="henry"
            )
    elif case_name.startswith("swap"):
        if case_name == "swap_infinite":
            if run_name == "Run 1":
                return htm.Permeability(
                    pre_exp=131720234232.97202,
                    act_energy=0.14259800636177897,
                    law="henry",
                )
            elif run_name == "Run 2":
                return htm.Permeability(
                    pre_exp=363415748298.5359,
                    act_energy=0.19672903147640755,
                    law="henry",
                )
            else:
                return htm.Permeability(
                    pre_exp=918010284564.2983,
                    act_energy=0.36689018612723834,
                    law="henry",
                )
        elif case_name == "swap_transparent":
            if run_name == "Run 1":
                return htm.Permeability(
                    pre_exp=18812216660963.566,
                    act_energy=0.4309849518078271,
                    law="henry",
                )
            elif run_name == "Run 2":
                return htm.Permeability(
                    pre_exp=19858409294360.746,
                    act_energy=0.4235587220438897,
                    law="henry",
                )
            else:
                return htm.Permeability(
                    pre_exp=139688360493517.1,
                    act_energy=0.6133713869157279,
                    law="henry",
                )
    raise ValueError(f"Unknown case name for permeability: {case_name}")


PermeabilityByCaseRun = Dict[str, Dict[str, htm.Permeability]]
PermeabilityMap = Dict[str, Dict[float, Dict[str, htm.Permeability]]]


def get_permeability_for_run(
    case_name: str,
    T_C: float,
    run_name: str,
    per_case_run_perm: Optional[PermeabilityByCaseRun] = None,
    permeability_map: Optional[PermeabilityMap] = None,
) -> htm.Permeability:
    """
    1. temperature-specific permeability_map
    2. per_case_run_perm
    3. fallback permability_by_case_name
    """
    if permeability_map is not None:
        case_block = permeability_map.get(case_name)
        if case_block is not None:
            temp_block = case_block.get(float(T_C))
            if temp_block is not None:
                perm = temp_block.get(run_name)
                if perm is not None:
                    return perm

    if per_case_run_perm is not None:
        case_block = per_case_run_perm.get(case_name)
        if case_block is not None:
            perm = case_block.get(run_name)
            if perm is not None:
                return perm

    return permability_by_case_name(case_name, run_name)


# ------------------------------ Run 1D once  ------------------------------
def run_once_1d(
    case_name: str,
    T_C: float,
    P_up: float,
    P_down: float,
    D_flibe: htm.Diffusivity,
    D_nickel: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    K_S_nickel: htm.Solubility,
    L_Ni: float,
    radius: float,
    run_name: str,
):
    """
    Solve the 1D Ni-FLiBe problem once for given case and conditions.
    Returns (J_in, J_out) in [H/s].
    """
    T_K = T_C + 273.15
    L_flibe = flibe_thickness_from_T_C(T_C)

    stem = f"{case_name}_{run_name}_T{int(T_C)}C"
    model, flux_out, flux_in, prof_ni, prof_flibe = make_ni_flibe_1d_model(
        T_K=T_K,
        P_up=P_up,
        P_down=P_down,
        L_Ni=L_Ni,
        L_flibe=L_flibe,
        radius=radius,
        D_nickel=D_nickel,
        D_flibe=D_flibe,
        K_S_nickel=K_S_nickel,
        permeability_flibe=permeability_flibe,
        penalty_term=1e20,
        stem=stem,
    )

    model.initialise()
    model.run()

    J_out = float(flux_out.data[-1])
    J_in = float(flux_in.data[-1])

    if prof_ni.data:
        arr_ni = np.array(prof_ni.data[-1])
        out_dir = Path("exports/results_1d/profiles")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_dir / f"{stem}_ni.csv", arr_ni, delimiter=",")

    if prof_flibe.data:
        arr_fl = np.array(prof_flibe.data[-1])
        out_dir = Path("exports/results_1d/profiles")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_dir / f"{stem}_flibe.csv", arr_fl, delimiter=",")

    return (
        J_in,
        J_out,
        T_K,
        L_flibe,
    )


# ------------------------------ Batch runner -----------------------------
def run_all_cases_1d(
    cases: Dict,
    T2K: Dict[float, float],
    D_flibe: htm.Diffusivity,
    D_nickel: htm.Diffusivity,
    K_S_nickel: htm.Solubility,
    per_case_run_perm: Optional[PermeabilityByCaseRun] = None,
    permeability_map: Optional[PermeabilityMap] = None,
    L_Ni: float = 0.002032,
    radius: float = 3.07 * 0.0254 / 2.0,
) -> List[dict]:
    """
    Loop over all (case, T_C, run) and return a flat list of results
    J_sim and J_out are both the 1D outlet flux [H/s].
    """
    all_results: List[dict] = []

    for case_name, case_cfg in cases.items():
        table = case_cfg["table"]

        for Tc, entry in table.items():
            runs = entry.get("runs", {})
            for run_name, cond in runs.items():
                perm_flibe = get_permeability_for_run(
                    case_name=case_name,
                    T_C=float(Tc),
                    run_name=run_name,
                    per_case_run_perm=per_case_run_perm,
                    permeability_map=permeability_map,
                )

                J_in, J_out, T_K, L_flibe = run_once_1d(
                    case_name=case_name,
                    T_C=float(Tc),
                    P_up=float(cond["P_up"]),
                    P_down=float(cond["P_down"]),
                    D_flibe=D_flibe,
                    D_nickel=D_nickel,
                    permeability_flibe=perm_flibe,
                    K_S_nickel=K_S_nickel,
                    L_Ni=L_Ni,
                    radius=radius,
                    run_name=run_name,
                )

                all_results.append(
                    {
                        "case": case_name,
                        "T_C": float(Tc),
                        "T_K": float(T_K),
                        "run": run_name,
                        "P_up": float(cond["P_up"]),
                        "P_down": float(cond["P_down"]),
                        "P_gb": float(cond.get("P_gb", 0.0))
                        if "P_gb" in cond
                        else None,
                        "phi0": float(perm_flibe.pre_exp.magnitude),
                        "E": float(perm_flibe.act_energy.magnitude),
                        "law": perm_flibe.law,
                        "J_sim": float(J_out),
                        "J_in": float(J_in),
                        "J_out": float(J_out),
                        "J_exp": float(cond.get("J_exp", np.nan)),
                    }
                )

                if _RANK0:
                    print(
                        f"[1D] {case_name:>18s} | T={float(Tc):5.1f} °C | "
                        f"{run_name:>5s} | L_flibe={L_flibe:.5e} m | "
                        f"J_in={J_in:.3e} H/s | J_out={J_out:.3e} H/s"
                    )

    return all_results


# ------------------------------ Experimental error ------------------------------
exp_error_data = {
    "normal_infinite": {
        500.0: {"runs": {"Run 1": 2.72e13, "Run 2": 4.49e13}},
        550.0: {"runs": {"Run 1": 4.62e13, "Run 2": 5.50e13}},
        600.0: {"runs": {"Run 1": 5.04e13, "Run 2": 7.63e13}},
        650.0: {"runs": {"Run 1": 7.87e13, "Run 2": 6.96e13}},
        700.0: {"runs": {"Run 1": 9.65e13, "Run 2": 9.34e13}},
    },
    "normal_transparent": {
        500.0: {"runs": {"Run 1": 2.72e13, "Run 2": 4.49e13}},
        550.0: {"runs": {"Run 1": 4.62e13, "Run 2": 5.50e13}},
        600.0: {"runs": {"Run 1": 5.04e13, "Run 2": 7.63e13}},
        650.0: {"runs": {"Run 1": 7.87e13, "Run 2": 6.96e13}},
        700.0: {"runs": {"Run 1": 9.65e13, "Run 2": 9.34e13}},
    },
    "swap_infinite": {
        500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13, "Run 3": 4.90e13}},
        550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
        600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14, "Run 3": 1.01e14}},
        650.0: {"runs": {"Run 2": 2.26e14}},
        700.0: {"runs": {"Run 1": 1.99e14, "Run 2": 2.19e14, "Run 3": 1.52e14}},
    },
    "swap_transparent": {
        500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13, "Run 3": 4.90e13}},
        550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
        600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14, "Run 3": 1.01e14}},
        650.0: {"runs": {"Run 2": 2.26e14}},
        700.0: {"runs": {"Run 1": 1.99e14, "Run 2": 2.19e14, "Run 3": 1.52e14}},
    },
}


def get_exp_error(case_name: str, temp, run_name: str = "Run 1"):
    """
    get experimental 1-sigma error.
    """
    import numpy as np

    case = exp_error_data.get(case_name)
    if case is None:
        return None
    entry = case.get(float(temp))
    if entry is None:
        return None
    if isinstance(entry, (int, float)):
        val = float(entry)
        return val if np.isfinite(val) and val > 0.0 else None
    if isinstance(entry, dict):
        runs = entry.get("runs")
        if isinstance(runs, dict):
            val = runs.get(run_name)
        else:
            val = entry.get(run_name)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) and val > 0.0 else None
    return None


def save_results_1d(all_results: List[dict], filepath: Path) -> None:
    """
    Save all 1D results into a single CSV file.

    Each row corresponds to one (case, T_C, run).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "case",
        "run",
        "T_C",
        "T_K",
        "P_up",
        "P_down",
        "P_gb",
        "phi0",
        "E",
        "J_in",
        "J_out",
        "J_sim",
        "J_exp",
    ]

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_permeabilities_summary(
    per_case_run_perm: PermeabilityByCaseRun, filepath: Path
) -> None:
    """
    Save the permeability used for each (case, run) into a CSV file.
    This is independent of temperature, because per_case_run_perm
    is an Arrhenius law reused for all temperatures.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["case", "run", "phi0", "E", "law"]

    with filepath.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for case_name, runs in per_case_run_perm.items():
            for run_name, perm in runs.items():
                writer.writerow(
                    {
                        "case": case_name,
                        "run": run_name,
                        "phi0": float(perm.pre_exp.magnitude),
                        "E": float(perm.act_energy.magnitude),
                        "law": perm.law,
                    }
                )


# ------------------------------ Main------------------------------
if __name__ == "__main__":
    # ---- materials ----
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")
    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

    # Temperatures and conversion to Kelvin
    temps_C_all = [500.0, 550.0, 600.0, 650.0, 700.0]
    T2K = {Tc: Tc + 273.15 for Tc in temps_C_all}

    # ---- input tables for cases ----
    normal_infinite = {
        500.0: {
            "runs": {
                "Run 1": {"P_up": 1.11e5, "P_down": 4.55, "J_exp": 1.04e15},
                "Run 2": {"P_up": 1.05e5, "P_down": 4.84, "J_exp": 1.09e15},
            }
        },
        550.0: {
            "runs": {
                "Run 1": {"P_up": 1.10e5, "P_down": 7.10, "J_exp": 1.52e15},
                "Run 2": {"P_up": 1.05e5, "P_down": 7.80, "J_exp": 1.69e15},
            }
        },
        600.0: {
            "runs": {
                "Run 1": {"P_up": 1.05e5, "P_down": 9.36, "J_exp": 2.01e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 1.07e1, "J_exp": 2.38e15},
            }
        },
        650.0: {
            "runs": {
                "Run 1": {"P_up": 1.05e5, "P_down": 1.51e1, "J_exp": 3.30e15},
                "Run 2": {"P_up": 1.04e5, "P_down": 1.40e1, "J_exp": 3.00e15},
            }
        },
        700.0: {
            "runs": {
                "Run 1": {"P_up": 1.03e5, "P_down": 1.97e1, "J_exp": 4.34e15},
                "Run 2": {"P_up": 1.02e5, "P_down": 2.04e1, "J_exp": 4.38e15},
            }
        },
    }
    normal_transparent = {
        500.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.11e5,
                    "P_down": 4.55,
                    "P_gb": 5.0,
                    "J_exp": 1.04e15,
                },
                "Run 2": {
                    "P_up": 1.05e5,
                    "P_down": 4.84,
                    "P_gb": 3.0,
                    "J_exp": 1.09e15,
                },
            }
        },
        550.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.10e5,
                    "P_down": 7.10,
                    "P_gb": 5.0,
                    "J_exp": 1.52e15,
                },
                "Run 2": {
                    "P_up": 1.05e5,
                    "P_down": 7.08,
                    "P_gb": 5.0,
                    "J_exp": 1.69e15,
                },
            }
        },
        600.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.05e5,
                    "P_down": 9.36,
                    "P_gb": 5.0,
                    "J_exp": 2.01e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 1.07e1,
                    "P_gb": 5.0,
                    "J_exp": 2.38e15,
                },
            }
        },
        650.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.05e5,
                    "P_down": 1.51e1,
                    "P_gb": 5.0,
                    "J_exp": 3.30e15,
                },
                "Run 2": {
                    "P_up": 1.04e5,
                    "P_down": 1.40e1,
                    "P_gb": 5.0,
                    "J_exp": 3.00e15,
                },
            }
        },
        700.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.03e5,
                    "P_down": 1.97e1,
                    "P_gb": 7.0,
                    "J_exp": 4.34e15,
                },
                "Run 2": {
                    "P_up": 1.02e5,
                    "P_down": 2.04e1,
                    "P_gb": 7.0,
                    "J_exp": 4.38e15,
                },
            }
        },
    }
    swap_infinite = {
        500.0: {
            "runs": {
                "Run 1": {"P_up": 1.31e5, "P_down": 1.77e1, "J_exp": 3.89e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 1.99e1, "J_exp": 4.34e15},
                "Run 3": {"P_up": 1.31e5, "P_down": 8.66, "J_exp": 1.91e15},
            }
        },
        550.0: {
            "runs": {
                "Run 1": {"P_up": 1.31e5, "P_down": 3.21e1, "J_exp": 7.20e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 3.89e1, "J_exp": 8.58e15},
            }
        },
        600.0: {
            "runs": {
                "Run 1": {"P_up": 1.33e5, "P_down": 3.57e1, "J_exp": 7.64e15},
                "Run 2": {"P_up": 1.32e5, "P_down": 4.62e1, "J_exp": 1.01e16},
                "Run 3": {"P_up": 1.33e5, "P_down": 2.10e1, "J_exp": 4.50e15},
            }
        },
        650.0: {
            "runs": {"Run 2": {"P_up": 1.32e5, "P_down": 5.02e1, "J_exp": 1.10e16}}
        },
        700.0: {
            "runs": {
                "Run 1": {"P_up": 1.32e5, "P_down": 4.07e1, "J_exp": 9.04e15},
                "Run 2": {"P_up": 1.32e5, "P_down": 4.78e1, "J_exp": 1.04e16},
                "Run 3": {"P_up": 1.31e5, "P_down": 3.23e1, "J_exp": 7.12e15},
            }
        },
    }

    # swap_transparent uses Sieverts out BC (so it needs P_gb), but you want to force all to 1e-30
    swap_transparent = {
        500.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 1.77e1,
                    "P_gb": 1e-30,
                    "J_exp": 3.89e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 1.99e1,
                    "P_gb": 1e-30,
                    "J_exp": 4.34e15,
                },
                "Run 3": {
                    "P_up": 1.31e5,
                    "P_down": 8.66,
                    "P_gb": 1e-30,
                    "J_exp": 1.91e15,
                },
            }
        },
        550.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 3.21e1,
                    "P_gb": 1e-30,
                    "J_exp": 7.20e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 3.89e1,
                    "P_gb": 1e-30,
                    "J_exp": 8.58e15,
                },
            }
        },
        600.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.33e5,
                    "P_down": 3.57e1,
                    "P_gb": 1e-30,
                    "J_exp": 7.64e15,
                },
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 4.62e1,
                    "P_gb": 1e-30,
                    "J_exp": 1.01e16,
                },
                "Run 3": {
                    "P_up": 1.33e5,
                    "P_down": 2.10e1,
                    "P_gb": 1e-30,
                    "J_exp": 4.50e15,
                },
            }
        },
        650.0: {
            "runs": {
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 5.02e1,
                    "P_gb": 1e-30,
                    "J_exp": 1.10e16,
                },
            }
        },
        700.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.32e5,
                    "P_down": 4.07e1,
                    "P_gb": 1e-30,
                    "J_exp": 9.04e15,
                },
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 4.78e1,
                    "P_gb": 1e-30,
                    "J_exp": 1.04e16,
                },
                "Run 3": {
                    "P_up": 1.31e5,
                    "P_down": 3.230e1,
                    "P_gb": 1e-30,
                    "J_exp": 7.12e15,
                },
            }
        },
    }
    cases = {
        "normal_infinite": {"table": normal_infinite},
        "normal_transparent": {"table": normal_transparent},
        "swap_infinite": {"table": swap_infinite},
        "swap_transparent": {"table": swap_transparent},
    }

    # Arrhenius permeability per case+run
    per_case_run_perm: PermeabilityByCaseRun = {
        "normal_infinite": {
            "Run 1": htm.Permeability(
                pre_exp=3169563873541.4814, act_energy=0.400882080795477, law="henry"
            ),
            "Run 2": htm.Permeability(
                pre_exp=3169563873541.4814, act_energy=0.400882080795477, law="henry"
            ),
        },
        "normal_transparent": {
            "Run 1": htm.Permeability(
                pre_exp=316467738703083.4, act_energy=0.6746935001483895, law="henry"
            ),
            "Run 2": htm.Permeability(
                pre_exp=316467738703083.4, act_energy=0.6746935001483895, law="henry"
            ),
        },
        "swap_infinite": {
            "Run 1": htm.Permeability(
                pre_exp=131720234232.97202, act_energy=0.14259800636177897, law="henry"
            ),
            "Run 2": htm.Permeability(
                pre_exp=363415748298.5359, act_energy=0.19672903147640755, law="henry"
            ),
            "Run 3": htm.Permeability(
                pre_exp=918010284564.2983, act_energy=0.36689018612723834, law="henry"
            ),
        },
        "swap_transparent": {
            "Run 1": htm.Permeability(
                pre_exp=18812216660963.566, act_energy=0.4309849518078271, law="henry"
            ),
            "Run 2": htm.Permeability(
                pre_exp=19858409294360.746, act_energy=0.4235587220438897, law="henry"
            ),
            "Run 3": htm.Permeability(
                pre_exp=139688360493517.1, act_energy=0.6133713869157279, law="henry"
            ),
        },
    }

    permeability_map: PermeabilityMap = {}

    # Run all 1D cases and collect results
    all_results_1d = run_all_cases_1d(
        cases=cases,
        T2K=T2K,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        per_case_run_perm=per_case_run_perm,
        permeability_map=permeability_map,
    )

    if _RANK0:
        print("\n===== 1D Sim vs Exp (J_sim = J_out) =====")
        for r in all_results_1d:
            # permeability at this temperature (Henry-law constant) from Arrhenius
            T_K = r["T_K"]
            phi0 = r["phi0"]
            E = r["E"]
            P_T = phi0 * np.exp(-E / (kB_eV * T_K))

            print(
                f"{r['case']:>18s} | T={r['T_C']:5.1f} °C | {r['run']:>5s} | "
                f"phi0={phi0:.3e} | E={E:.3f} eV | law={r['law']} | "
                f"P(T)={P_T:.3e} | "
                f"J_sim={r['J_sim']:.3e} | J_exp={r['J_exp']:.3e}"
            )

    # Save all 1D results to a single CSV file
    out_root = Path("exports") / "results_1d"
    save_results_1d(all_results_1d, out_root / "all_results_1d.csv")

    # Save the permeability used for each (case, run)
    save_permeabilities_summary(
        per_case_run_perm, out_root / "permeabilities_used_1d.csv"
    )
