import festim as F
from mesh import generate_mesh, set_y_ft
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import h_transport_materials as htm
from typing import Literal, Tuple, Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import math
import matplotlib
import gc
from petsc4py import PETSc
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# ------------------------------ Globals ------------------------------
_DEFER_SHOW = True
kB_eV = 8.617333262e-5  # eV/K

# Mesh & model caches (per process)
_mesh_cache: Dict[str, tuple] = {}

# Rank guard for logs (avoid duplicated logs with MPI>1)
_RANK0 = MPI.COMM_WORLD.rank == 0


# ------------------------------ Type aliases ------------------------------
# Arrhenius-permeability per case+run (same law used for all T)
PermeabilityByCaseRun = Dict[str, Dict[str, htm.Permeability]]
# Optional temperature-specific override: case -> T_C -> run -> Permeability
PermeabilityMap = Dict[str, Dict[float, Dict[str, htm.Permeability]]]


# ------------------------------ Utilities ------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _arrhenius_str(Phi0: float, E_eV: float, unit="H·m⁻¹·s⁻¹·Pa⁻¹") -> str:
    return (
        r"$\varphi(T)$ = "
        f"{Phi0:.4g} · exp("
        + r"$-\dfrac{"
        + f"{E_eV:.4g}"
        + r"\ \mathrm{eV}}{k_B T}$"
        + f")  [{unit}]"
    )


def _ensure_and_save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if _RANK0:
        print(f"[saved] {out_path}")
    plt.close(fig)


def _phi_at_T(T_K: float, perm) -> float:
    try:
        return float(perm.value(T_K).magnitude)
    except Exception:
        return float(perm.pre_exp) * math.exp(
            -float(perm.act_energy) / (kB_eV * float(T_K))
        )


def _dispose_model(m):
    if m is None:
        return
    try:
        for e in getattr(m, "exports", []) or []:
            for attr in ("data", "field", "surface"):
                if hasattr(e, attr):
                    setattr(e, attr, None)
        m.exports = []
        m.interfaces = []
        m.subdomains = []
        m.boundary_conditions = []
        for attr in ("_forms", "_function_spaces", "_solvers", "_timers"):
            if hasattr(m, attr):
                setattr(m, attr, None)
        m.mesh = None
    except Exception:
        pass

    gc.collect()
    try:
        PETSc.garbage_cleanup()
    except Exception:
        pass
    try:
        MPI.COMM_WORLD.barrier()
    except Exception:
        pass


# ------------------------------ Mesh I/O ------------------------------
def _mesh_key_from_yft(y_ft: float) -> str:
    y5 = float(f"{float(y_ft):.5f}")
    return f"mesh_{y5:.5f}.msh"


def load_or_make_mesh(mesh_file: str, mesh_size: float, model_rank: int = 0):
    if not Path(mesh_file).exists():
        if _RANK0:
            print(f"[mesh] creating '{mesh_file}' (size={mesh_size:g})")
        generate_mesh(mesh_size=mesh_size, fname=mesh_file)

    if mesh_file in _mesh_cache:
        return _mesh_cache[mesh_file]

    _read = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, model_rank)
    mesh = _read.mesh
    cell_tags = _read.cell_tags
    facet_tags = _read.facet_tags
    _mesh_cache[mesh_file] = (mesh, cell_tags, facet_tags)
    return _mesh_cache[mesh_file]


# ------------------------------ Materials & Model ------------------------------
def make_materials(D_solid, D_liquid, K_solid, permeability_liquid):
    # solid
    D_0_solid = D_solid.pre_exp.magnitude
    E_D_solid = D_solid.act_energy.magnitude
    K_S_0_solid = K_solid.pre_exp.magnitude
    E_K_S_solid = K_solid.act_energy.magnitude
    # liquid
    D_0_liquid = D_liquid.pre_exp.magnitude
    E_D_liquid = D_liquid.act_energy.magnitude
    K_S_liquid = htm.Solubility(
        S_0=permeability_liquid.pre_exp / D_liquid.pre_exp,
        E_S=permeability_liquid.act_energy - D_liquid.act_energy,
        law=permeability_liquid.law,
    )
    K_S_0_liquid = K_S_liquid.pre_exp.magnitude
    E_K_S_liquid = K_S_liquid.act_energy.magnitude

    mat_solid = F.Material(
        D_0=D_0_solid,
        E_D=E_D_solid,
        K_S_0=K_S_0_solid,
        E_K_S=E_K_S_solid,
        solubility_law="sievert",
    )
    mat_liquid = F.Material(
        D_0=D_0_liquid,
        E_D=E_D_liquid,
        K_S_0=K_S_0_liquid,
        E_K_S=E_K_S_liquid,
        solubility_law="henry",
    )
    return mat_solid, mat_liquid


def make_model(
    D_flibe: htm.Diffusivity,
    D_nickel: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    K_S_nickel: htm.Solubility,
    temperature: float,
    P_up: float,
    case: Literal[
        "normal_infinite",
        "normal_transparent",
        "swap_infinite",
        "swap_transparent",
    ],
    mesh_size: float = 2e-4,
    penalty_term: float = 1e22,
    P_down: float = 5.0,
    out_bc: dict | None = None,
    y_ft: float | None = None,
) -> Tuple[F.HydrogenTransportProblemDiscontinuous, Dict[str, List | CylindricalFlux]]:
    if y_ft is None:
        raise ValueError("y_ft must be provided")

    y_ft_5 = float(f"{float(y_ft):.5f}")
    set_y_ft(y_ft_5)

    Path("exports/yft_record").mkdir(parents=True, exist_ok=True)
    if _RANK0:
        with open("exports/yft_record/y_ft_values.txt", "a") as f:
            f.write(f"{y_ft_5:.5f}\n")

    mesh_file = _mesh_key_from_yft(y_ft_5)
    model_rank = 0
    mesh, cell_tags, facet_tags = load_or_make_mesh(
        mesh_file, mesh_size, model_rank=model_rank
    )

    mat_solid, mat_liquid = make_materials(
        D_solid=D_nickel,
        D_liquid=D_flibe,
        K_solid=K_S_nickel,
        permeability_liquid=permeability_flibe,
    )

    # BC-side parameters (Henry on liquid, Sieverts on Ni)
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )
    H_0_liq = K_S_liquid.pre_exp.magnitude
    E_H_liq = K_S_liquid.act_energy.magnitude
    K_S_0_Ni = K_S_nickel.pre_exp.magnitude
    E_S_Ni = K_S_nickel.act_energy.magnitude

    # subdomains
    fluid_volume = F.VolumeSubdomain(id=1, material=mat_liquid)
    solid_volume = F.VolumeSubdomain(id=2, material=mat_solid)

    out_surf = F.SurfaceSubdomain(id=3)
    left_bc_liquid = F.SurfaceSubdomain(id=41)
    left_bc_top_Ni = F.SurfaceSubdomain(id=42)
    left_bc_middle_Ni = F.SurfaceSubdomain(id=43)
    left_bc_bottom_Ni = F.SurfaceSubdomain(id=44)
    top_cap_Ni = F.SurfaceSubdomain(id=5)
    top_sidewall_Ni = F.SurfaceSubdomain(id=6)
    bottom_sidewall_Ni = F.SurfaceSubdomain(id=7)
    liquid_surface = F.SurfaceSubdomain(id=8)
    mid_membrane_Ni = F.SurfaceSubdomain(id=9)
    bottom_cap_Ni = F.SurfaceSubdomain(id=10)
    liquid_solid_interface = F.SurfaceSubdomain(id=99)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags
    my_model.subdomains = [
        solid_volume,
        fluid_volume,
        out_surf,
        left_bc_liquid,
        left_bc_top_Ni,
        left_bc_middle_Ni,
        left_bc_bottom_Ni,
        top_cap_Ni,
        top_sidewall_Ni,
        bottom_sidewall_Ni,
        liquid_surface,
        mid_membrane_Ni,
        bottom_cap_Ni,
        liquid_solid_interface,
    ]

    my_model.method_interface = "penalty"
    interface = F.Interface(
        id=99, subdomains=[solid_volume, fluid_volume], penalty_term=penalty_term
    )
    my_model.interfaces = [interface]

    my_model.surface_to_volume = {
        out_surf: solid_volume,
        left_bc_liquid: fluid_volume,
        left_bc_top_Ni: solid_volume,
        left_bc_middle_Ni: solid_volume,
        left_bc_bottom_Ni: solid_volume,
        top_cap_Ni: solid_volume,
        top_sidewall_Ni: solid_volume,
        bottom_sidewall_Ni: solid_volume,
        liquid_surface: fluid_volume,
        mid_membrane_Ni: solid_volume,
        bottom_cap_Ni: solid_volume,
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = temperature

    # case logic
    if case in ("normal_infinite", "normal_transparent"):
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [top_cap_Ni, top_sidewall_Ni]
        ] + [
            F.HenrysBC(
                subdomain=liquid_surface,
                species=H,
                pressure=P_down,
                H_0=H_0_liq,
                E_H=E_H_liq,
            )
        ]
    elif case in ("swap_infinite", "swap_transparent"):
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [top_cap_Ni, top_sidewall_Ni]
        ] + [
            F.HenrysBC(
                subdomain=liquid_surface,
                species=H,
                pressure=P_up,
                H_0=H_0_liq,
                E_H=E_H_liq,
            )
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
        ]
    else:
        raise ValueError(f"Unknown case: {case}")

    # optional out-surface (glovebox)
    if out_bc is None:
        out_bc = {"type": "none"}
    out_bcs: List = []
    t = out_bc.get("type", "none").lower()
    if t == "sieverts":
        out_bcs = [
            F.SievertsBC(
                subdomain=out_surf,
                species=H,
                pressure=float(out_bc.get("pressure", 0.0)),
                S_0=K_S_0_Ni,
                E_S=E_S_Ni,
            )
        ]
    elif t == "particle_flux_zero":
        out_bcs = [F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)]

    my_model.boundary_conditions = upstream_bcs + downstream_bcs + out_bcs
    my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)

    # Flux monitors
    flux_out_top_cap_Ni = CylindricalFlux(field=H, surface=top_cap_Ni)
    flux_out_top_sidewall_Ni = CylindricalFlux(field=H, surface=top_sidewall_Ni)
    flux_out_liquid_surface = CylindricalFlux(field=H, surface=liquid_surface)
    flux_out_mid_membrane_Ni = CylindricalFlux(field=H, surface=mid_membrane_Ni)
    flux_out_bottom_sidewall_Ni = CylindricalFlux(field=H, surface=bottom_sidewall_Ni)
    flux_out_bottom_cap_Ni = CylindricalFlux(field=H, surface=bottom_cap_Ni)
    glovebox_flux = CylindricalFlux(field=H, surface=out_surf)

    my_model.exports = [
        flux_out_top_cap_Ni,
        flux_out_top_sidewall_Ni,
        flux_out_bottom_sidewall_Ni,
        flux_out_liquid_surface,
        flux_out_mid_membrane_Ni,
        flux_out_bottom_cap_Ni,
        glovebox_flux,
    ]

    flux_by_label = {
        "top_cap_Ni": flux_out_top_cap_Ni,
        "top_sidewall_Ni": flux_out_top_sidewall_Ni,
        "liquid_surface": flux_out_liquid_surface,
        "mid_membrane_Ni": flux_out_mid_membrane_Ni,
        "bottom_sidewall_Ni": flux_out_bottom_sidewall_Ni,
        "bottom_cap_Ni": flux_out_bottom_cap_Ni,
    }
    six_labels = [
        "top_cap_Ni",
        "top_sidewall_Ni",
        "liquid_surface",
        "mid_membrane_Ni",
        "bottom_sidewall_Ni",
        "bottom_cap_Ni",
    ]
    up_labels = (
        ["mid_membrane_Ni", "bottom_cap_Ni", "bottom_sidewall_Ni"]
        if case in ("normal_infinite", "normal_transparent")
        else ["top_cap_Ni", "top_sidewall_Ni", "liquid_surface"]
    )
    down_labels = (
        ["top_cap_Ni", "top_sidewall_Ni", "liquid_surface"]
        if case in ("normal_infinite", "normal_transparent")
        else ["mid_membrane_Ni", "bottom_cap_Ni", "bottom_sidewall_Ni"]
    )
    fluxes_dict = {
        "flux_by_label": flux_by_label,
        "six_labels": six_labels,
        "glovebox_flux": glovebox_flux,
        "up_labels": up_labels,
        "down_labels": down_labels,
    }
    return my_model, fluxes_dict


def _get_flux_value(flux_obj) -> float:
    try:
        data = getattr(flux_obj, "data", None)
        if data is not None:
            a = np.asarray(data, dtype=float)
            if a.size > 0:
                return float(a[-1])
        val = getattr(flux_obj, "value", None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 0.0


def run_once(
    case: str,
    T_K: float,
    P_up: float,
    P_down: float,
    D_flibe,
    D_nickel,
    permeability_flibe,
    K_S_nickel,
    out_bc: dict | None = None,
    y_ft: float | None = None,
):
    """
    Solve steady-state once and return aggregated fluxes.
    """
    my_model, fluxes_dict = make_model(
        temperature=T_K,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        permeability_flibe=permeability_flibe,
        K_S_nickel=K_S_nickel,
        P_up=P_up,
        case=case,
        P_down=P_down,
        out_bc=out_bc,
        y_ft=y_ft,
    )
    my_model.initialise()
    my_model.run()

    flux_objects = fluxes_dict["flux_by_label"]
    six_labels = fluxes_dict["six_labels"]
    vals_six = {label: _get_flux_value(flux_objects[label]) for label in six_labels}
    glovebox_val = float(_get_flux_value(fluxes_dict["glovebox_flux"]))
    up_labels = fluxes_dict["up_labels"]
    down_labels = fluxes_dict["down_labels"]
    total_up = float(np.sum([vals_six[label] for label in up_labels], dtype=float))
    total_down = float(np.sum([vals_six[label] for label in down_labels], dtype=float))

    display_names = {
        "top_cap_Ni": "Top cap (Ni)",
        "top_sidewall_Ni": "Upper Ni sidewall",
        "liquid_surface": "FLiBe surface",
        "mid_membrane_Ni": "Middle membrane",
        "bottom_cap_Ni": "Bottom Ni (top face)",
        "bottom_sidewall_Ni": "Lower Ni sidewall",
    }
    per_surface = {
        "labels": six_labels,
        "values": [vals_six[label] for label in six_labels],
        "up_labels": up_labels,
        "down_labels": down_labels,
        "up_names": [display_names[label] for label in up_labels],
        "down_names": [display_names[label] for label in down_labels],
        "up_vals": [vals_six[label] for label in up_labels],
        "down_vals": [vals_six[label] for label in down_labels],
    }

    _dispose_model(my_model)
    return dict(
        total_in=total_up,
        glovebox=glovebox_val,
        total_out=total_down,
        balance=total_up + glovebox_val + total_down,
        per_surface=per_surface,
    )


# ------------------------------ Permeability selection helpers ------------------------------
def permability_by_case_name(
    case_name: str,
    run_name: str = "",
) -> htm.Permeability:
    """
    Baseline permeability laws, used if no custom Arrhenius is provided.
    You can ignore this if you always provide per_case_run_perm.
    """
    if case_name.startswith("normal"):
        # same for infinite/transparent; can be adjusted
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
                    pre_exp=437941789378.1358,
                    act_energy=0.1621177114636059,
                    law="henry",
                )
            else:
                return htm.Permeability(
                    pre_exp=41587400565660.95,
                    act_energy=0.4655730255084721,
                    law="henry",
                )
        elif case_name == "swap_transparent":
            if run_name == "Run 1":
                return htm.Permeability(
                    pre_exp=1670438225847.1777,
                    act_energy=0.23510226611596444,
                    law="henry",
                )
            else:
                return htm.Permeability(
                    pre_exp=61674694703011.12,
                    act_energy=0.4724257569796252,
                    law="henry",
                )
    raise ValueError(f"Unknown case name for permeability: {case_name}")


def get_permeability_for_run(
    case_name: str,
    T_C: float,
    run_name: str,
    per_case_run_perm: Optional[PermeabilityByCaseRun] = None,
    permeability_map: Optional[PermeabilityMap] = None,
) -> htm.Permeability:
    """
    Decide which permeability to use for a given (case, T_C, run).

    Priority:
    1. If provided: permeability_map[case_name][T_C][run_name]
       (temperature-specific overrides)
    2. Else if provided: per_case_run_perm[case_name][run_name]
       (Arrhenius law reused for all temperatures)
    3. Otherwise: permability_by_case_name(case_name, run_name)
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


def run_all_cases_with_custom_permeabilities(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    D_nickel,
    K_S_nickel,
    per_case_run_perm: Optional[PermeabilityByCaseRun] = None,
    permeability_map: Optional[PermeabilityMap] = None,
) -> List[dict]:
    """
    Loop over all cases / temperatures / runs, using
      - temp-specific permeability if given (permeability_map), or
      - Arrhenius per case+run (per_case_run_perm), or
      - baseline permability_by_case_name otherwise.

    Returns flat list of dicts with sim vs exp flux.
    """
    all_results: List[dict] = []

    for case_name, case_cfg in cases.items():
        table = case_cfg["table"]
        out_mode = case_cfg.get("out_mode", "particle_flux_zero")

        for Tc, entry in table.items():
            Tk = T2K[Tc]
            y_raw = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])
            y_ft = float(f"{float(y_raw):.5f}")

            runs = entry.get("runs", {})
            for run_name, cond in runs.items():
                perm_flibe = get_permeability_for_run(
                    case_name=case_name,
                    T_C=float(Tc),
                    run_name=run_name,
                    per_case_run_perm=per_case_run_perm,
                    permeability_map=permeability_map,
                )

                if out_mode == "sieverts":
                    out_bc = {
                        "type": "sieverts",
                        "pressure": float(cond.get("P_gb", 0.0)),
                    }
                else:
                    out_bc = {"type": "particle_flux_zero"}

                res = run_once(
                    case=case_name,
                    T_K=Tk,
                    P_up=float(cond["P_up"]),
                    P_down=float(cond["P_down"]),
                    D_flibe=D_flibe,
                    D_nickel=D_nickel,
                    permeability_flibe=perm_flibe,
                    K_S_nickel=K_S_nickel,
                    out_bc=out_bc,
                    y_ft=y_ft,
                )

                all_results.append(
                    {
                        "case": case_name,
                        "T_C": float(Tc),
                        "T_K": float(Tk),
                        "run": run_name,
                        "P_up": float(cond["P_up"]),
                        "P_down": float(cond["P_down"]),
                        "P_gb": float(cond.get("P_gb", 0.0))
                        if "P_gb" in cond
                        else None,
                        "phi0": float(perm_flibe.pre_exp.magnitude),
                        "E": float(perm_flibe.act_energy.magnitude),
                        # choose total_out as J_sim; change to total_in if needed
                        "J_sim": float(res["total_out"]),
                        "J_in": float(res["total_in"]),
                        "J_gb": float(res["glovebox"]),
                        "balance": float(res["balance"]),
                        "J_exp": float(cond.get("J_exp", np.nan)),
                    }
                )

    return all_results


# ------------------------------ Experimental errors ------------------------------
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
        500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13}},
        550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
        600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14}},
        700.0: {"runs": {"Run 1": 1.99e14}},
    },
    "swap_transparent": {
        500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13}},
        550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
        600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14}},
        700.0: {"runs": {"Run 1": 1.99e14}},
    },
}


def get_exp_error(case_name: str, temp, run_name: str = "Run 1"):
    """
    Return experimental error (1-sigma) for given case / temperature / run,
    or None if not available.
    """
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


# ------------------------------ Plotting: J_sim vs J_exp + error bars ------------------------------
def plot_jsim_vs_jexp_with_errorbars(
    all_results: List[dict],
    outdir_root: Path = Path("exports") / "figs" / "permeability_comparison",
):
    """
    For each (case, run):
      - J_sim shown as bar chart
      - J_exp shown as points with experimental error bars

    Uses global get_exp_error(case_name, temp, run_name).
    Saves one figure per (case, run).
    """
    if not all_results:
        if _RANK0:
            print("No results to plot (all_results is empty).")
        return

    # group by (case, run)
    by_case_run: Dict[Tuple[str, str], List[dict]] = {}
    for row in all_results:
        key = (row["case"], row["run"])
        by_case_run.setdefault(key, []).append(row)

    for (case_name, run_name), rows in by_case_run.items():
        # sort by temperature
        rows.sort(key=lambda r: r["T_C"])
        T = np.array([r["T_C"] for r in rows], dtype=float)  # °C
        J_sim = np.array([r["J_sim"] for r in rows], dtype=float)
        J_exp = np.array([r["J_exp"] for r in rows], dtype=float)

        # experimental error from your table
        J_err = np.zeros_like(J_exp)
        for i, Tc in enumerate(T):
            err_val = get_exp_error(case_name, Tc, run_name)
            J_err[i] = 0.0 if err_val is None else float(err_val)

        x = np.arange(len(T), dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4))

        # J_sim as bars
        ax.bar(x, J_sim, width=0.6, label="J_sim", alpha=0.7)

        # J_exp as points with error bars
        ax.errorbar(
            x,
            J_exp,
            yerr=J_err,
            fmt="o",
            capsize=4,
            color="red",
            label="J_exp (exp ± error)",
        )

        ax.set_xticks(x)
        ax.set_xticklabels([f"{Tc:.0f}" for Tc in T])
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("Flux [H/s]")
        ax.set_title(f"{case_name} — {run_name}: J_sim vs J_exp")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(fontsize=8)

        fig.tight_layout()

        outdir = outdir_root / "jsim_vs_jexp" / case_name
        _ensure_and_save(
            fig,
            outdir / f"{case_name}_{run_name}_Jsim_vs_Jexp.png",
        )


# ------------------------------ Plotting: relative error ------------------------------
def plot_relative_error(
    all_results: List[dict],
    outdir_root: Path = Path("exports") / "figs" / "permeability_comparison",
):
    """
    For each (case, run), plot relative error:
        (J_sim - J_exp) / J_exp  vs T

    One figure per (case, run), separate from J_sim/J_exp plot.
    """
    if not all_results:
        if _RANK0:
            print("No results to plot (all_results is empty).")
        return

    by_case_run: Dict[Tuple[str, str], List[dict]] = {}
    for row in all_results:
        key = (row["case"], row["run"])
        by_case_run.setdefault(key, []).append(row)

    for (case_name, run_name), rows in by_case_run.items():
        rows.sort(key=lambda r: r["T_C"])
        T = np.array([r["T_C"] for r in rows], dtype=float)
        J_sim = np.array([r["J_sim"] for r in rows], dtype=float)
        J_exp = np.array([r["J_exp"] for r in rows], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(J_exp != 0.0, (J_sim - J_exp) / J_exp, np.nan)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(T, rel_err, marker="d", linestyle="-", label="relative error")

        ax.axhline(0.0, linewidth=0.8)
        ax.set_xlabel("Temperature [°C]")
        ax.set_ylabel("(J_sim - J_exp) / J_exp")
        ax.set_title(f"{case_name} — {run_name}: relative error")
        ax.legend(fontsize=8)

        fig.tight_layout()

        outdir = outdir_root / "relative_error" / case_name
        _ensure_and_save(
            fig,
            outdir / f"{case_name}_{run_name}_relative_error.png",
        )


# ------------------------------ Optional calibration helpers (unchanged style) ------------------------------
@dataclass
class CalibPoint:
    case: str
    T_C: float
    T_K: float
    run: str
    P_up: float
    P_down: float
    P_gb: Optional[float]
    y_ft: float
    J_exp: float


def _collect_points_for_cases(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    allowed_case_names: Optional[List[str]] = None,
) -> List[CalibPoint]:
    pts: List[CalibPoint] = []
    for case_name, cfg in cases.items():
        if allowed_case_names and case_name not in allowed_case_names:
            continue
        if not (case_name.startswith("normal") or case_name.startswith("swap")):
            continue

        table = cfg["table"]
        for Tc, row in table.items():
            Tk = T2K[Tc]
            y_raw = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])
            y5 = float(f"{float(y_raw):.5f}")
            runs = row.get("runs", {})
            for run_name, cond in runs.items():
                pts.append(
                    CalibPoint(
                        case=case_name,
                        T_C=float(Tc),
                        T_K=float(Tk),
                        run=str(run_name),
                        P_up=float(cond["P_up"]),
                        P_down=float(cond["P_down"]),
                        P_gb=float(cond.get("P_gb")) if "P_gb" in cond else None,
                        y_ft=y5,
                        J_exp=float(cond["J_exp"]),
                    )
                )
    if not pts:
        raise RuntimeError("No calibration points found.")
    pts.sort(key=lambda p: (p.case, p.T_C, p.run))
    return pts


def fig_saving(case_name: str, T_C: int, run_name: str) -> Path:
    return Path("exports") / "figs" / "calibration_A" / case_name / f"{T_C}C" / run_name


def save_breakdown(fig, outdir: Path, stem: str):
    _ensure_and_save(fig, outdir / f"{stem}.png")


def plot_case_breakdowns_with_exp(
    case_name: str,
    case_cfg: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    permeability_flibe_fitted,
    D_nickel,
    K_S_nickel,
    outdir_root: Path = Path("exports") / "figs" / "calibration_A",
):
    table = case_cfg.get("table", {})
    case_outdir = outdir_root / case_name
    _ensure_dir(case_outdir)

    for Tc in sorted(table.keys()):
        Tk = T2K[Tc]
        y_val = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])
        y_ft = float(f"{float(y_val):.5f}")

        runs = table[Tc].get("runs", {})
        for run_name, cond in runs.items():
            out_bc = (
                {"type": "sieverts", "pressure": float(cond.get("P_gb", 0.0))}
                if "P_gb" in cond
                else {"type": "particle_flux_zero"}
            )
            res = run_once(
                case=case_name,
                T_K=Tk,
                P_up=float(cond["P_up"]),
                P_down=float(cond["P_down"]),
                D_flibe=D_flibe,
                D_nickel=D_nickel,
                K_S_nickel=K_S_nickel,
                permeability_flibe=permeability_flibe_fitted,
                out_bc=out_bc,
                y_ft=y_ft,
            )

            per = res["per_surface"]
            six_labels = per["labels"]
            six_values = [float(v) for v in per["values"]]
            values = [
                float(cond["J_exp"]),
                float(res["total_in"]),
                float(res["total_out"]),
                float(res["glovebox"]),
            ] + six_values
            labels = ["Flux exp", "Flux in", "Flux out", "Glovebox"] + list(six_labels)

            fig, ax = plt.subplots(figsize=(13, 6))
            x = np.arange(len(labels), dtype=float)
            bars = ax.bar(x, np.abs(values))
            for bar, v in zip(bars, values):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    (h if h > 0 else 0.0) * 1.02 + (1e-30 if h == 0 else 0),
                    f"{v:.2e}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Flux [H/s]")
            ax.set_title(
                f"{case_name} — {run_name} — {int(Tc)} °C  (y_ft={float(y_ft):.5f} m)"
            )
            info = (
                f"Case: {case_name}\nRun: {run_name}\nT = {int(Tc)} °C (T_K={Tk:.2f})\n"
                f"y_ft = {float(y_ft):.5f} m\nP_up = {float(cond['P_up']):.2e} Pa\n"
                f"P_down = {float(cond['P_down']):.2e} Pa\n"
                + (
                    f"P_glovebox = {float(cond['P_gb']):.2e} Pa"
                    if "P_gb" in cond
                    else "P_glovebox = (closed)"
                )
            )
            ax.text(
                0.99,
                0.98,
                info,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0.5),
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.tight_layout()

            fd = fig_saving(case_name, int(Tc), run_name)
            save_breakdown(
                fig,
                fd,
                stem=f"{case_name}_{int(Tc)}C_{run_name}_breakdown_exp_vs_model",
            )


# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)

    # ---- materials ----
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")
    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

    # Temperatures and K conversion
    temps_C_all = [500.0, 550.0, 600.0, 650.0, 700.0]
    T2K = {Tc: Tc + 273.15 for Tc in temps_C_all}

    # temperature-dependent FLiBe thickness [m]
    Y_FT_BY_TEMP_C = {
        500.0: 0.02914,
        550.0: 0.02919,
        600.0: 0.02925,
        650.0: 0.02930,
        700.0: 0.02936,
    }

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
            }
        },
        700.0: {
            "runs": {"Run 1": {"P_up": 1.32e5, "P_down": 4.07e1, "J_exp": 9.04e15}}
        },
    }
    swap_transparent = {
        500.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 1.77e1,
                    "P_gb": 7.0,
                    "J_exp": 3.89e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 3.89e1,
                    "P_gb": 7.0,
                    "J_exp": 4.34e15,
                },
            }
        },
        550.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 3.21e1,
                    "P_gb": 1.0e1,
                    "J_exp": 7.20e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 3.89e1,
                    "P_gb": 1.0e1,
                    "J_exp": 8.58e15,
                },
            }
        },
        600.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.33e5,
                    "P_down": 3.57e1,
                    "P_gb": 1.2e1,
                    "J_exp": 7.64e15,
                },
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 4.62e1,
                    "P_gb": 1.2e1,
                    "J_exp": 1.01e16,
                },
            }
        },
        700.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.32e5,
                    "P_down": 4.07e1,
                    "P_gb": 2.2e1,
                    "J_exp": 9.04e15,
                }
            }
        },
    }

    cases = {
        "normal_infinite": {"table": normal_infinite, "out_mode": "particle_flux_zero"},
        "normal_transparent": {"table": normal_transparent, "out_mode": "sieverts"},
        "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }

    # ---------------------- ARRHENIUS INPUT PER CASE+RUN ----------------------
    # Fill this dict with the Arrhenius parameters you want to test.
    # Example below just mirrors the baseline; replace with your own values.
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
                pre_exp=437941789378.1358, act_energy=0.1621177114636059, law="henry"
            ),
            "Run 2": htm.Permeability(
                pre_exp=41587400565660.95, act_energy=0.4655730255084721, law="henry"
            ),
        },
        "swap_transparent": {
            "Run 1": htm.Permeability(
                pre_exp=1670438225847.1777, act_energy=0.23510226611596444, law="henry"
            ),
            "Run 2": htm.Permeability(
                pre_exp=61674694703011.12, act_energy=0.4724257569796252, law="henry"
            ),
        },
    }

    # Optional: temperature-specific overrides (usually not needed)
    permeability_map: PermeabilityMap = {}

    # Run all cases with these permeabilities
    all_results = run_all_cases_with_custom_permeabilities(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        per_case_run_perm=per_case_run_perm,
        permeability_map=permeability_map,
    )

    if _RANK0:
        print("\n===== Sim vs Exp (J_sim = total_out) =====")
        for r in all_results:
            print(
                f"{r['case']:>18s} | T={r['T_C']:5.1f} °C | {r['run']:>5s} | "
                f"phi0={r['phi0']:.3e} | E={r['E']:.3f} eV | "
                f"J_sim={r['J_sim']:.3e} | J_exp={r['J_exp']:.3e}"
            )

    # 1) J_sim bars + J_exp points with error bars
    plot_jsim_vs_jexp_with_errorbars(all_results)

    # 2) Separate relative error plots
    plot_relative_error(all_results)

    if (not _DEFER_SHOW) and ("agg" not in matplotlib.get_backend().lower()):
        plt.show()
