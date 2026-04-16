"""
Simulate downstream flux for all experimental cases and compare against measurements.
FLiBe permeability is loaded from results/fitted_params.csv (para_swap_pure.py output).
Ni permeability is loaded from results/dry_run_phi_arrhenius_fits.txt (dry_run_fitting.py output).
All experimental data and uncertainties are imported from exp_data.py.

Outputs (saved to results/):
    master_summary.csv       -- J_sim vs J_exp with relative error per case/run/T
    jsim_jexp.csv            -- long-format table for plotting
    percentage_metrics.csv   -- sidewall leakage and liquid/membrane split metrics
    surface_breakdown.csv    -- per-surface flux breakdown
"""

import csv
import gc
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
from dolfinx.io import gmsh as gmshio
from dolfinx.log import LogLevel, set_log_level
import festim as F
import h_transport_materials as htm
from mpi4py import MPI
from petsc4py import PETSc

from cylindrical_flux import CylindricalFlux
from mesh import generate_mesh, set_y_ft
from exp_data import (
    D_nickel,
    D_flibe,
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

OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

_RANK0 = MPI.COMM_WORLD.rank == 0
_mesh_cache: Dict[str, tuple] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mesh_key_from_yft(y_ft: float) -> str:
    return f"mesh_{float(y_ft):.5f}.msh"


def load_or_make_mesh(mesh_file: str, mesh_size: float, model_rank: int = 0):
    if not Path(mesh_file).exists():
        if _RANK0:
            print(f"[mesh] creating '{mesh_file}' (size={mesh_size:g})")
        generate_mesh(mesh_size=mesh_size, fname=mesh_file)
    if mesh_file in _mesh_cache:
        return _mesh_cache[mesh_file]
    _read = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, model_rank)
    _mesh_cache[mesh_file] = (_read.mesh, _read.cell_tags, _read.facet_tags)
    return _mesh_cache[mesh_file]


def _dispose_model(m) -> None:
    if m is None:
        return
    try:
        for e in getattr(m, "exports", []) or []:
            for attr in ("data", "field", "surface"):
                if hasattr(e, attr):
                    setattr(e, attr, None)
        for attr in ("exports", "interfaces", "subdomains", "boundary_conditions"):
            setattr(m, attr, [])
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


def _safe_rel_err(j_sim: float, j_exp: float) -> float:
    if not np.isfinite(j_exp) or j_exp == 0.0:
        return np.nan
    return (j_sim - j_exp) / j_exp


def _run_number(run_name: str) -> int:
    try:
        return int(str(run_name).replace("Run ", "").strip())
    except Exception:
        return 999


def _split_case_group(case_name: str) -> tuple[str, str]:
    for prefix in ("normal_", "swap_"):
        if case_name.startswith(prefix):
            return prefix.rstrip("_"), case_name[len(prefix) :]
    raise ValueError(f"Unrecognized case name: {case_name}")


def get_exp_error(case_name: str, temp: float, run_name: str) -> Optional[float]:
    """Return 1-sigma flux uncertainty (k=2 stored value divided by 2)."""
    all_err = {**normal_flux_err, **swap_flux_err}
    case = all_err.get(case_name)
    if case is None:
        return None
    entry = case.get(float(temp))
    if entry is None:
        return None
    raw = entry.get("runs", {}).get(run_name)
    try:
        val = float(raw) / 2.0
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) and val > 0.0 else None


# ── Materials ─────────────────────────────────────────────────────────────────


def _make_materials(D_flibe, K_S_nickel, permeability_flibe):
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )
    mat_solid = F.Material(
        D_0=D_nickel.pre_exp.magnitude,
        E_D=D_nickel.act_energy.magnitude,
        K_S_0=K_S_nickel.pre_exp.magnitude,
        E_K_S=K_S_nickel.act_energy.magnitude,
        solubility_law="sievert",
    )
    mat_liquid = F.Material(
        D_0=D_flibe.pre_exp.magnitude,
        E_D=D_flibe.act_energy.magnitude,
        K_S_0=K_S_liquid.pre_exp.magnitude,
        E_K_S=K_S_liquid.act_energy.magnitude,
        solubility_law="henry",
    )
    return mat_solid, mat_liquid


# ── Model builder ─────────────────────────────────────────────────────────────


def _make_model(
    case: Literal[
        "normal_infinite", "normal_transparent", "swap_infinite", "swap_transparent"
    ],
    D_flibe: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    K_S_nickel: htm.Solubility,
    temperature: float,
    P_up: float,
    P_down: float,
    out_bc: dict,
    y_ft: float,
    mesh_size: float = 2e-4,
    penalty_term: float = 1e22,
):
    y_ft_5 = float(f"{float(y_ft):.5f}")
    set_y_ft(y_ft_5)

    yft_log = OUTDIR / "yft_record" / "y_ft_values.txt"
    yft_log.parent.mkdir(parents=True, exist_ok=True)
    if _RANK0:
        with open(yft_log, "a") as f:
            f.write(f"{y_ft_5:.5f}\n")

    mesh, cell_tags, facet_tags = load_or_make_mesh(
        _mesh_key_from_yft(y_ft_5), mesh_size
    )
    mat_solid, mat_liquid = _make_materials(D_flibe, K_S_nickel, permeability_flibe)

    K_S_0_Ni = mat_solid.K_S_0
    E_S_Ni = mat_solid.E_K_S
    H_0_liq = mat_liquid.K_S_0
    E_H_liq = mat_liquid.E_K_S

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

    all_surface_subdomains = [
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

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags
    my_model.subdomains = [solid_volume, fluid_volume] + all_surface_subdomains
    my_model.method_interface = "penalty"
    my_model.interfaces = [
        F.Interface(
            id=99, subdomains=[solid_volume, fluid_volume], penalty_term=penalty_term
        )
    ]
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

    # Upstream/downstream surfaces differ between normal and swap configurations.
    if case in ("normal_infinite", "normal_transparent"):
        upstream_surfaces = [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
        downstream_surfaces = [top_cap_Ni, top_sidewall_Ni]
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in upstream_surfaces
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in downstream_surfaces
        ] + [
            F.HenrysBC(
                subdomain=liquid_surface,
                species=H,
                pressure=P_down,
                H_0=H_0_liq,
                E_H=E_H_liq,
            )
        ]
    else:  # swap
        upstream_surfaces = [top_cap_Ni, top_sidewall_Ni]
        downstream_surfaces = [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in upstream_surfaces
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
            for s in downstream_surfaces
        ]

    t = out_bc.get("type", "none").lower()
    if t == "sieverts":
        outer_bcs = [
            F.SievertsBC(
                subdomain=out_surf,
                species=H,
                pressure=float(out_bc.get("pressure", 0.0)),
                S_0=K_S_0_Ni,
                E_S=E_S_Ni,
            )
        ]
    elif t == "particle_flux_zero":
        outer_bcs = [F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)]
    else:
        outer_bcs = []

    my_model.boundary_conditions = upstream_bcs + downstream_bcs + outer_bcs
    my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)

    surface_labels = [
        "top_cap_Ni",
        "top_sidewall_Ni",
        "liquid_surface",
        "mid_membrane_Ni",
        "bottom_sidewall_Ni",
        "bottom_cap_Ni",
    ]
    surfaces = {
        "top_cap_Ni": top_cap_Ni,
        "top_sidewall_Ni": top_sidewall_Ni,
        "liquid_surface": liquid_surface,
        "mid_membrane_Ni": mid_membrane_Ni,
        "bottom_sidewall_Ni": bottom_sidewall_Ni,
        "bottom_cap_Ni": bottom_cap_Ni,
    }
    flux_objs = {
        label: CylindricalFlux(field=H, surface=surf)
        for label, surf in surfaces.items()
    }
    glovebox_flux = CylindricalFlux(field=H, surface=out_surf)
    my_model.exports = list(flux_objs.values()) + [glovebox_flux]

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

    return my_model, flux_objs, glovebox_flux, surface_labels, up_labels, down_labels


# ── Run one simulation ────────────────────────────────────────────────────────


def _run_once(
    case: str,
    T_K: float,
    P_up: float,
    P_down: float,
    D_flibe: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    K_S_nickel: htm.Solubility,
    out_bc: dict,
    y_ft: float,
) -> dict:
    my_model, flux_objs, glovebox_flux, surface_labels, up_labels, down_labels = (
        _make_model(
            case=case,
            D_flibe=D_flibe,
            permeability_flibe=permeability_flibe,
            K_S_nickel=K_S_nickel,
            temperature=T_K,
            P_up=P_up,
            P_down=P_down,
            out_bc=out_bc,
            y_ft=y_ft,
        )
    )
    my_model.initialise()
    my_model.run()

    vals = {label: _get_flux_value(flux_objs[label]) for label in surface_labels}
    total_up = float(sum(vals[label] for label in up_labels))
    total_down = float(sum(vals[label] for label in down_labels))
    glovebox = float(_get_flux_value(glovebox_flux))

    _dispose_model(my_model)
    return dict(
        total_in=total_up,
        total_out=total_down,
        glovebox=glovebox,
        surface_labels=surface_labels,
        surface_values=[vals[label] for label in surface_labels],
        up_labels=up_labels,
        down_labels=down_labels,
        up_vals=[vals[label] for label in up_labels],
        down_vals=[vals[label] for label in down_labels],
    )


# ── Run all cases ─────────────────────────────────────────────────────────────


def run_all_cases(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe: htm.Diffusivity,
    flibe_perm_by_case_run: Dict,
    K_S_nickel_by_case: Dict[str, htm.Solubility],
) -> List[dict]:
    """
    Run one steady-state simulation per (case, temperature, run) combination
    using the FLiBe permeability loaded from the inversion output.

    flibe_perm_by_case_run : {(case_name, run_name): htm.Permeability}
        Loaded from results/fitted_params.csv via load_flibe_permeability().
    K_S_nickel_by_case : {case_name: htm.Solubility}
        Ni solubility per case, derived from dry-run Arrhenius fits.
    """
    all_results: List[dict] = []

    for case_name, cfg in cases.items():
        out_mode = cfg.get("out_mode", "particle_flux_zero")
        K_S_nickel = K_S_nickel_by_case[case_name]

        for Tc, entry in cfg["table"].items():
            T_K = T2K[Tc]
            y_ft = float(
                f"{float(Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])):.5f}"
            )

            for run_name, cond in entry.get("runs", {}).items():
                perm_flibe = flibe_perm_by_case_run.get((case_name, run_name))
                if perm_flibe is None:
                    if _RANK0:
                        print(f"[skip] no permeability for ({case_name}, {run_name})")
                    continue

                if out_mode == "sieverts":
                    out_bc = {
                        "type": "sieverts",
                        "pressure": float(cond.get("P_gb", 0.0)),
                    }
                else:
                    out_bc = {"type": "particle_flux_zero"}

                res = _run_once(
                    case=case_name,
                    T_K=T_K,
                    P_up=float(cond["P_up"]),
                    P_down=float(cond["P_down"]),
                    D_flibe=D_flibe,
                    permeability_flibe=perm_flibe,
                    K_S_nickel=K_S_nickel,
                    out_bc=out_bc,
                    y_ft=y_ft,
                )

                vals = dict(zip(res["surface_labels"], res["surface_values"]))
                J_in = abs(float(res["total_in"]))
                J_out = abs(float(res["total_out"]))
                J_liquid = abs(vals.get("liquid_surface", 0.0))
                J_membrane = abs(vals.get("mid_membrane_Ni", 0.0))

                # Fraction of upstream flux that bypasses the primary path via sidewalls
                if J_in > 0.0:
                    pct_sidewall_leak = 100.0 * (
                        1 - J_membrane / J_in
                        if case_name.startswith("normal")
                        else 1 - J_liquid / J_in
                    )
                else:
                    pct_sidewall_leak = float("nan")

                # Fraction of downstream flux contributed by sidewalls
                if J_out > 0.0:
                    pct_sidewall_comp = 100.0 * (
                        1 - J_liquid / J_out
                        if case_name.startswith("normal")
                        else 1 - J_membrane / J_out
                    )
                    pct_liq_mem_diff = 100.0 * (J_liquid - J_membrane) / J_out
                else:
                    pct_sidewall_comp = float("nan")
                    pct_liq_mem_diff = float("nan")

                all_results.append(
                    dict(
                        case=case_name,
                        T_C=float(Tc),
                        T_K=float(T_K),
                        run=run_name,
                        P_up=float(cond["P_up"]),
                        P_down=float(cond["P_down"]),
                        P_gb=float(cond["P_gb"]) if "P_gb" in cond else None,
                        phi0=float(perm_flibe.pre_exp.magnitude),
                        E=float(perm_flibe.act_energy.magnitude),
                        J_sim=float(res["total_out"]),
                        J_in=float(res["total_in"]),
                        J_exp=float(cond.get("J_exp", np.nan)),
                        pct_sidewall_leak=pct_sidewall_leak,
                        pct_sidewall_comp=pct_sidewall_comp,
                        pct_liq_mem_diff=pct_liq_mem_diff,
                        surface_labels=res["surface_labels"],
                        surface_values=[float(v) for v in res["surface_values"]],
                        up_labels=res["up_labels"],
                        down_labels=res["down_labels"],
                        up_vals=[float(v) for v in res["up_vals"]],
                        down_vals=[float(v) for v in res["down_vals"]],
                    )
                )

    return all_results


# ── CSV exports ───────────────────────────────────────────────────────────────


def _sorted_results(all_results):
    return sorted(
        all_results, key=lambda r: (_run_number(r["run"]), r["case"], r["T_C"])
    )


def export_master_summary_csv(all_results: List[dict]) -> None:
    path = OUTDIR / "master_summary.csv"
    fields = [
        "run",
        "case",
        "type",
        "config",
        "T_C",
        "T_K",
        "P_up",
        "P_down",
        "P_gb",
        "phi0",
        "E_eV",
        "J_sim",
        "J_exp",
        "J_exp_err_1sigma",
        "relative_error",
        "J_in",
        "pct_sidewall_leak",
        "pct_sidewall_comp",
        "pct_liq_mem_diff",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in _sorted_results(all_results):
            typ, cfg = _split_case_group(r["case"])
            j_err = get_exp_error(r["case"], r["T_C"], r["run"])
            rel = _safe_rel_err(r["J_sim"], r["J_exp"])
            w.writerow(
                dict(
                    run=r["run"],
                    case=r["case"],
                    type=typ,
                    config=cfg,
                    T_C=r["T_C"],
                    T_K=r["T_K"],
                    P_up=r["P_up"],
                    P_down=r["P_down"],
                    P_gb="" if r["P_gb"] is None else r["P_gb"],
                    phi0=r["phi0"],
                    E_eV=r["E"],
                    J_sim=r["J_sim"],
                    J_exp=r["J_exp"],
                    J_exp_err_1sigma="" if j_err is None else j_err,
                    relative_error="" if not np.isfinite(rel) else rel,
                    J_in=r["J_in"],
                    pct_sidewall_leak=""
                    if not np.isfinite(r["pct_sidewall_leak"])
                    else r["pct_sidewall_leak"],
                    pct_sidewall_comp=""
                    if not np.isfinite(r["pct_sidewall_comp"])
                    else r["pct_sidewall_comp"],
                    pct_liq_mem_diff=""
                    if not np.isfinite(r["pct_liq_mem_diff"])
                    else r["pct_liq_mem_diff"],
                )
            )
    if _RANK0:
        print(f"[saved] {path}")


def export_jsim_jexp_csv(all_results: List[dict]) -> None:
    path = OUTDIR / "jsim_jexp.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "case", "type", "config", "T_C", "series", "value", "error"])
        for r in _sorted_results(all_results):
            typ, cfg = _split_case_group(r["case"])
            j_err = get_exp_error(r["case"], r["T_C"], r["run"])
            w.writerow(
                [r["run"], r["case"], typ, cfg, r["T_C"], "J_sim", r["J_sim"], ""]
            )
            w.writerow(
                [
                    r["run"],
                    r["case"],
                    typ,
                    cfg,
                    r["T_C"],
                    "J_exp",
                    r["J_exp"],
                    "" if j_err is None else j_err,
                ]
            )
    if _RANK0:
        print(f"[saved] {path}")


def export_percentage_metrics_csv(all_results: List[dict]) -> None:
    path = OUTDIR / "percentage_metrics.csv"
    metrics = {
        "pct_sidewall_leak": "upstream_sidewall_over_inlet",
        "pct_sidewall_comp": "downstream_sidewall_over_outlet",
        "pct_liq_mem_diff": "liquid_membrane_diff_over_outlet",
    }
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "case",
                "type",
                "config",
                "T_C",
                "metric_key",
                "metric_name",
                "value",
            ]
        )
        for r in _sorted_results(all_results):
            typ, cfg = _split_case_group(r["case"])
            for key, name in metrics.items():
                val = r[key]
                w.writerow(
                    [
                        r["run"],
                        r["case"],
                        typ,
                        cfg,
                        r["T_C"],
                        key,
                        name,
                        "" if not np.isfinite(val) else val,
                    ]
                )
    if _RANK0:
        print(f"[saved] {path}")


def export_surface_breakdown_csv(all_results: List[dict]) -> None:
    path = OUTDIR / "surface_breakdown.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run",
                "case",
                "type",
                "config",
                "T_C",
                "surface",
                "flux",
                "is_upstream",
                "is_downstream",
            ]
        )
        for r in _sorted_results(all_results):
            typ, cfg = _split_case_group(r["case"])
            up_set = set(r["up_labels"])
            down_set = set(r["down_labels"])
            for label, val in zip(r["surface_labels"], r["surface_values"]):
                w.writerow(
                    [
                        r["run"],
                        r["case"],
                        typ,
                        cfg,
                        r["T_C"],
                        label,
                        val,
                        int(label in up_set),
                        int(label in down_set),
                    ]
                )
    if _RANK0:
        print(f"[saved] {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)

    T2K = {Tc: Tc + 273.15 for Tc in [500.0, 550.0, 600.0, 650.0, 700.0]}
    Y_FT_BY_TEMP_C = {
        500.0: 0.02914,
        550.0: 0.02919,
        600.0: 0.02925,
        650.0: 0.02930,
        700.0: 0.02936,
    }

    cases = {
        "normal_infinite": {"table": normal_infinite, "out_mode": "particle_flux_zero"},
        "normal_transparent": {"table": normal_transparent, "out_mode": "sieverts"},
        "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }

    # FLiBe permeability from inversion output (results/fitted_params.csv)
    flibe_perm_by_case_run = load_flibe_permeability()

    # Ni solubility from dry-run Arrhenius fits (results/dry_run_phi_arrhenius_fits.txt)
    ni_perm = load_ni_permeability()
    K_S_nickel_by_case = {}
    for case_name, cfg in cases.items():
        bc_type = cfg["out_mode"]
        prm = ni_perm[bc_type]
        phi_0 = prm["phi_0"] * 6.02214076e23  # mol -> particles
        E_phi = prm["E_phi_kJmol"] / 96.485332123  # kJ/mol -> eV
        perm = htm.Permeability(pre_exp=phi_0, act_energy=E_phi, law="sievert")
        K_S_nickel_by_case[case_name] = htm.Solubility(
            S_0=perm.pre_exp / D_nickel.pre_exp,
            E_S=perm.act_energy - D_nickel.act_energy,
            law="sievert",
        )

    all_results = run_all_cases(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        flibe_perm_by_case_run=flibe_perm_by_case_run,
        K_S_nickel_by_case=K_S_nickel_by_case,
    )

    if _RANK0:
        print("\n===== Sim vs Exp summary =====")
        for r in _sorted_results(all_results):
            rel = _safe_rel_err(r["J_sim"], r["J_exp"])
            print(
                f"{r['run']:>5s} | {r['case']:>20s} | T={r['T_C']:5.1f}C | "
                f"J_sim={r['J_sim']:.3e} | J_exp={r['J_exp']:.3e} | "
                f"rel_err={rel * 100:.1f}%"
                if np.isfinite(rel)
                else f"{r['run']:>5s} | {r['case']:>20s} | T={r['T_C']:5.1f}C | "
                f"J_sim={r['J_sim']:.3e} | J_exp={r['J_exp']:.3e}"
            )

    export_master_summary_csv(all_results)
    export_jsim_jexp_csv(all_results)
    export_percentage_metrics_csv(all_results)
    export_surface_breakdown_csv(all_results)

    if _RANK0:
        print(f"\nAll CSV outputs saved to: {OUTDIR.resolve()}")
