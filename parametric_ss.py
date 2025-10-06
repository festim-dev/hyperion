import festim as F
from mesh import generate_mesh, set_y_ft
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmshio
from mpi4py import MPI
import h_transport_materials as htm
from typing import Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt


def make_materials(D_solid, D_liquid, K_solid, permeability_liquid):
    # material parameters for solid
    D_0_solid = D_solid.pre_exp.magnitude  # m^2/s
    E_D_solid = D_solid.act_energy.magnitude  # ev/particle
    K_S_0_solid = K_solid.pre_exp.magnitude  # particle m^-3 Pa^-0.5
    E_K_S_solid = K_solid.act_energy.magnitude  # ev/particle

    # material parameters for liquid

    D_0_liquid = D_liquid.pre_exp.magnitude  # m^2/s
    E_D_liquid = D_liquid.act_energy.magnitude  # ev/particle

    K_S_liquid = htm.Solubility(
        S_0=permeability_liquid.pre_exp / D_liquid.pre_exp,
        E_S=permeability_liquid.act_energy - D_liquid.act_energy,
        law=permeability_liquid.law,
    )

    K_S_0_liquid = K_S_liquid.pre_exp.magnitude  # particle m^-3 Pa^-1
    E_K_S_liquid = K_S_liquid.act_energy.magnitude  # ev/particle

    # Define materials
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
        "prf_infinite",
        "transparent",
    ],
    mesh_size: float = 2e-4,
    penalty_term: float = 1e24,
    P_down: float = 5.0,
    out_bc: dict | None = None,
    y_ft: float | None = None,
) -> Tuple[F.HydrogenTransportProblemDiscontinuous, dict[str, list | CylindricalFlux]]:
    if y_ft is not None:
        set_y_ft(y_ft)

    generate_mesh(mesh_size=mesh_size)
    model_rank = 0
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        "mesh.msh", MPI.COMM_WORLD, model_rank
    )

    mat_solid, mat_liquid = make_materials(
        D_solid=D_nickel,
        D_liquid=D_flibe,
        K_solid=K_S_nickel,
        permeability_liquid=permeability_flibe,
    )

    # recompute liquid solubility (Henry) parameters for boundary conditions
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )
    # For Henry BC class, parameters are named H_0/E_H
    H_0_liq = K_S_liquid.pre_exp.magnitude
    E_H_liq = K_S_liquid.act_energy.magnitude
    K_S_0_Ni = K_S_nickel.pre_exp.magnitude
    E_S_Ni = K_S_nickel.act_energy.magnitude

    fluid_volume = F.VolumeSubdomain(id=1, material=mat_liquid)
    solid_volume = F.VolumeSubdomain(id=2, material=mat_solid)

    out_surf = F.SurfaceSubdomain(id=3)
    left_bc_liquid = F.SurfaceSubdomain(id=41)
    left_bc_top_Ni = F.SurfaceSubdomain(id=42)
    left_bc_middle_Ni = F.SurfaceSubdomain(id=43)
    left_bc_bottom_Ni = F.SurfaceSubdomain(id=44)
    top_Ni_bottom = F.SurfaceSubdomain(id=5)
    Ds_Ni_left = F.SurfaceSubdomain(id=6)
    Up_Ni_left = F.SurfaceSubdomain(id=7)
    Liquid_top = F.SurfaceSubdomain(id=8)
    mem_Ni_bottom = F.SurfaceSubdomain(id=9)
    bottom_Ni_top = F.SurfaceSubdomain(id=10)
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
        top_Ni_bottom,
        Ds_Ni_left,
        Up_Ni_left,
        Liquid_top,
        mem_Ni_bottom,
        bottom_Ni_top,
        liquid_solid_interface,
    ]

    my_model.method_interface = "penalty"
    interface = F.Interface(
        id=99, subdomains=[solid_volume, fluid_volume], penalty_term=penalty_term
    )
    my_model.interfaces = [interface]

    my_model.surface_to_volume = {
        out_surf: solid_volume,
        left_bc_liquid: fluid_volume,  # NOTE: this is fluid
        left_bc_top_Ni: solid_volume,
        left_bc_middle_Ni: solid_volume,
        left_bc_bottom_Ni: solid_volume,
        top_Ni_bottom: solid_volume,
        Ds_Ni_left: solid_volume,
        Up_Ni_left: solid_volume,
        Liquid_top: fluid_volume,
        mem_Ni_bottom: solid_volume,
        bottom_Ni_top: solid_volume,
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]

    my_model.temperature = temperature

    upstream_volume_surfaces = [mem_Ni_bottom, bottom_Ni_top, Up_Ni_left]
    downstream_volume_surfaces_Ni = [top_Ni_bottom, Ds_Ni_left]
    downstream_volume_surfaces = [top_Ni_bottom, Ds_Ni_left, Liquid_top]

    # transient BCs using time-dependent pressure for downstream sides

    upstream_surface_bcs = [
        F.SievertsBC(
            subdomain=s,
            species=H,
            pressure=P_up,
            S_0=K_S_0_Ni,
            E_S=E_S_Ni,
        )
        for s in upstream_volume_surfaces
    ]

    # downstream Ni surfaces: Sieverts w/ P_down
    downstream_surface_bcs_Ni = [
        F.SievertsBC(
            subdomain=s,
            species=H,
            pressure=P_down,
            S_0=K_S_0_Ni,
            E_S=E_S_Ni,
        )
        for s in downstream_volume_surfaces_Ni
    ]

    # liquid top: Henry w/ P_down
    def _henry_bc_for_liquid(subdomain):
        return F.HenrysBC(
            subdomain=subdomain,
            species=H,
            pressure=P_down,  # downstream pressure in Pa
            H_0=H_0_liq,  # Henry constant pre-exp
            E_H=E_H_liq,  # Henry activation energy [eV]
        )

    downstream_surface_bcs_liq = [_henry_bc_for_liquid(Liquid_top)]

    # optional out-surface BC (per-run control)
    out_bcs = []
    if out_bc is None:
        out_bc = {"type": "none"}
    if out_bc.get("type", "none").lower() == "sievert":
        out_bcs = [
            F.SievertsBC(
                subdomain=out_surf,
                species=H,
                pressure=float(out_bc.get("pressure", 0.0)),
                S_0=K_S_0_Ni,
                E_S=E_S_Ni,
            )
        ]
    elif out_bc.get("type", "none").lower() == "fixed_c":
        out_bcs = [
            F.FixedConcentrationBC(
                subdomain=out_surf,
                species=H,
                value=float(out_bc.get("value", 0.0)),
            )
        ]

    my_model.boundary_conditions = (
        upstream_surface_bcs
        + downstream_surface_bcs_Ni
        + downstream_surface_bcs_liq
        + out_bcs
    )

    # my_model.settings = F.Settings(
    #     atol=1e12,
    #     rtol=1e-13,
    #     transient=True,
    #     stepsize=dt,
    #     final_time=t_total,
    # )
    my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)

    # -------- flux monitors --------
    fluxes_in = [
        CylindricalFlux(field=H, surface=surf) for surf in upstream_volume_surfaces
    ]
    downstream_fluxes_total = [
        CylindricalFlux(field=H, surface=surf) for surf in downstream_volume_surfaces
    ]
    glovebox_flux = CylindricalFlux(field=H, surface=out_surf)
    flux_out_liquid = CylindricalFlux(field=H, surface=Liquid_top)
    flux_out_Ni_sidewall = CylindricalFlux(field=H, surface=Ds_Ni_left)
    flux_out_top_Ni_bottom = CylindricalFlux(field=H, surface=top_Ni_bottom)

    # field exports for visualization (optional)
    my_model.exports = [
        F.VTXSpeciesExport(
            field=H, filename="FLiBe_infinite_PRF_solid_ss.bp", subdomain=solid_volume
        ),
        F.VTXSpeciesExport(
            field=H, filename="FLiBe_infinite_PRF_liquid_ss.bp", subdomain=fluid_volume
        ),
    ]
    my_model.exports += downstream_fluxes_total
    my_model.exports += fluxes_in
    my_model.exports += [
        glovebox_flux,
        flux_out_liquid,
        flux_out_Ni_sidewall,
        flux_out_top_Ni_bottom,
    ]

    fluxes_dict = {
        "fluxes_in": fluxes_in,
        "downstream_fluxes_total": downstream_fluxes_total,
        "glovebox_flux": glovebox_flux,
        "flux_out_liquid": flux_out_liquid,
        "flux_out_Ni_sidewall": flux_out_Ni_sidewall,
        "flux_out_top_Ni_bottom": flux_out_top_Ni_bottom,
    }

    return my_model, fluxes_dict


def _get_flux_value(flux_obj) -> float:
    """
    Helper for steady-state: use .value if present; otherwise last sample in .data.
    """
    if hasattr(flux_obj, "value") and flux_obj.value is not None:
        return float(flux_obj.value)
    if hasattr(flux_obj, "data") and flux_obj.data:
        return float(np.asarray(flux_obj.data, dtype=float)[-1])
    try:
        return float(flux_obj)
    except Exception:
        return 0.0


def run_once(
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
    Solve steady-state once and return all relevant flux numbers for plotting.
    """
    my_model, all_fluxes = make_model(
        temperature=T_K,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        permeability_flibe=permeability_flibe,
        K_S_nickel=K_S_nickel,
        P_up=P_up,
        case="prf_infinite",
        P_down=P_down,
        out_bc=out_bc,
        y_ft=y_ft,
    )

    # time step & final time (seconds) for transient run

    my_model.initialise()
    my_model.run()  # steady state

    fluxes_in_list = all_fluxes["fluxes_in"]
    glovebox_flux = all_fluxes["glovebox_flux"]
    flux_out_liquid = all_fluxes["flux_out_liquid"]
    flux_out_Ni_sidewall = all_fluxes["flux_out_Ni_sidewall"]
    flux_out_top_Ni_bottom = all_fluxes["flux_out_top_Ni_bottom"]
    downstream_fluxes_total_list = all_fluxes["downstream_fluxes_total"]

    v_in_terms = np.array([_get_flux_value(f) for f in fluxes_in_list], dtype=float)
    total_flux_in = float(np.sum(v_in_terms))
    total_flux_glovebox = float(_get_flux_value(glovebox_flux))
    v_out_liquid = float(_get_flux_value(flux_out_liquid))
    v_downstream_sidewall = float(_get_flux_value(flux_out_Ni_sidewall))
    v_downstream_top = float(_get_flux_value(flux_out_top_Ni_bottom))
    total_downstream_flux = float(
        np.sum([_get_flux_value(f) for f in downstream_fluxes_total_list])
    )
    balance = total_flux_in + total_flux_glovebox + total_downstream_flux

    return dict(
        total_in=total_flux_in,
        glovebox=total_flux_glovebox,
        liquid_top=v_out_liquid,
        ni_side=v_downstream_sidewall,
        ni_top=v_downstream_top,
        total_downstream=total_downstream_flux,
        balance=balance,
    )


if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)
    # filter nickel and H
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")

    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    # liquid properties (edit if needed)
    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)
    permeability_flibe = htm.Permeability(pre_exp=1.2e13, act_energy=0.49, law="henry")

    # Temperatures: 500, 550, 600, 650, 700 °C
    temperatures_C = [500.0, 550.0, 600.0, 650.0, 700.0]
    temperatures_K = [Tc + 273.15 for Tc in temperatures_C]

    # Two simulation runs per temperature
    run_labels = ["Sim 1", "Sim 2"]

    # Per-temperature, per-run settings from the table (Pa)
    # Format: P_up in Pa, P_down in Pa
    run_params = {
        500.0: {
            "Sim 1": {"P_up": 1.11e5, "P_down": 4.55, "out_bc": {"type": "none"}},
            "Sim 2": {"P_up": 1.05e5, "P_down": 4.84, "out_bc": {"type": "none"}},
        },
        550.0: {
            "Sim 1": {"P_up": 1.10e5, "P_down": 7.10, "out_bc": {"type": "none"}},
            "Sim 2": {"P_up": 1.05e5, "P_down": 7.80, "out_bc": {"type": "none"}},
        },
        600.0: {
            "Sim 1": {"P_up": 1.05e5, "P_down": 9.36, "out_bc": {"type": "none"}},
            "Sim 2": {"P_up": 1.31e5, "P_down": 1.07e1, "out_bc": {"type": "none"}},
        },
        650.0: {
            "Sim 1": {"P_up": 1.05e5, "P_down": 1.51e1, "out_bc": {"type": "none"}},
            "Sim 2": {"P_up": 1.04e5, "P_down": 1.40e1, "out_bc": {"type": "none"}},
        },
        700.0: {
            "Sim 1": {"P_up": 1.03e5, "P_down": 1.97e1, "out_bc": {"type": "none"}},
            "Sim 2": {"P_up": 1.02e5, "P_down": 2.04e1, "out_bc": {"type": "none"}},
        },
    }

    # Experimental downstream totals [Run1, Run2] per temperature (fill as needed)
    exp_downstream = {
        500.0: [1.04e15, 1.09e15],
        550.0: [1.52e15, 1.69e15],
        600.0: [2.01e15, 2.38e15],
        650.0: [3.30e15, 3.00e15],
        700.0: [4.34e15, 4.38e15],
    }

    # Permeability Φ(T) for labels/legends
    def phi_at_T(T_K: float) -> float:
        try:
            return float(permeability_flibe.value(T_K).magnitude)
        except Exception:
            kB_eV = 8.617333262e-5
            return float(
                permeability_flibe.pre_exp
                * np.exp(-permeability_flibe.act_energy / (kB_eV * T_K))
            )

    phi_map = {Tc: phi_at_T(Tk) for Tc, Tk in zip(temperatures_C, temperatures_K)}

    # Run simulations and collect results
    model_totals = {run_case: [] for run_case in run_labels}
    per_temp_components = {
        Tc: {run_case: None for run_case in run_labels} for Tc in temperatures_C
    }

    # temperature-dependent FLiBe thickness [m]
    Y_FT_BY_TEMP_C = {
        500.0: 0.02914,
        550.0: 0.02919,
        600.0: 0.02925,
        650.0: 0.02930,
        700.0: 0.02936,
    }

    for Tc, Tk in zip(temperatures_C, temperatures_K):
        for run_case in run_labels:
            run_setup = run_params[Tc][run_case]
            res = run_once(
                T_K=Tk,
                P_up=run_setup["P_up"],
                P_down=run_setup["P_down"],
                D_flibe=D_flibe,
                D_nickel=D_nickel,
                permeability_flibe=permeability_flibe,
                K_S_nickel=K_S_nickel,
                out_bc=run_setup["out_bc"],
                y_ft=Y_FT_BY_TEMP_C[Tc],
            )
            model_totals[run_case].append(res["total_downstream"])
            per_temp_components[Tc][run_case] = res

    header = "Temp(°C) | Run | Model [H/s] | Exp [H/s] | Rel. err (%)"
    print(header)
    print("-" * len(header))
    for i, T in enumerate(temperatures_C):
        for r_idx, run_case in enumerate(["Sim 1", "Sim 2"]):
            model_val = model_totals[run_case][i]
            exp_val = exp_downstream[T][r_idx]
            rel_err = (
                100.0 * (model_val - exp_val) / exp_val if exp_val != 0 else np.nan
            )
            print(
                f"{int(T):>8} | {run_case:>5} | {model_val:>12.3e} | {exp_val:>9.3e} | {rel_err:>11.2f}"
            )
    print()

    # print(Y_FT_BY_TEMP_C[Tc])
    # grouped bars per temperature: Exp R1/R2 vs Sim1/Sim2
    temps = temperatures_C
    x = np.arange(len(temps), dtype=float)
    group_w = 0.8
    w = group_w / 4.0

    exp_r1 = [exp_downstream[T][0] for T in temps]
    exp_r2 = [exp_downstream[T][1] for T in temps]
    sim1 = [model_totals["Sim 1"][i] for i in range(len(temps))]
    sim2 = [model_totals["Sim 2"][i] for i in range(len(temps))]

    figA, axA = plt.subplots(figsize=(13, 6))
    axA.bar(x - 1.5 * w, exp_r1, width=w, label="Exp R1")
    axA.bar(x - 0.5 * w, exp_r2, width=w, label="Exp R2")
    axA.bar(x + 0.5 * w, sim1, width=w, label="Model Sim 1")
    axA.bar(x + 1.5 * w, sim2, width=w, label="Model Sim 2")

    axA.set_xticks(x)
    # axA.set_xticklabels([f"{int(T)}°C\nΦ={phi_map[T]:.2e}" for T in temps], rotation=0)
    axA.set_xticklabels(
        [
            f"{int(T)}°C\nΦ={phi_map[T]:.2e}\nthickness_FLiBe={Y_FT_BY_TEMP_C[T]:.5f} m"
            for T in temps
        ]
    )

    axA.set_ylabel("Total downstream flux [H/s]")
    axA.set_title("Total downstream flux comparison (500-700 °C)")
    axA.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axA.legend()
    figA.tight_layout()
    plt.show()

    # per-temperature flux breakdown (use Sim 1 by default)
    for Tc, Tk in zip(temperatures_C, temperatures_K):
        phi_here = phi_map[Tc]
        r1 = per_temp_components[Tc]["Sim 1"]
        labels = [
            f"Exp run 1 (T={int(Tc)}°C)",
            f"Exp run 2 (T={int(Tc)}°C)",
            "Flux in (upstream)",
            "Flux out to glovebox",
            "Total downstream flux",
            "Flux through membrane surface",
            "Flux through Ni downstream sidewall",
            "Flux through Ni downstream top surface",
            "Flux balance (in + glovebox + downstream)",
        ]
        values = [
            exp_downstream[Tc][0],
            exp_downstream[Tc][1],
            r1["total_in"],
            r1["glovebox"],
            r1["total_downstream"],
            r1["liquid_top"],
            r1["ni_side"],
            r1["ni_top"],
            r1["balance"],
        ]

        figB, axB = plt.subplots(figsize=(13, 5.5))
        xB = np.arange(len(labels), dtype=float)
        colors = ["tab:blue", "tab:blue"] + ["tab:orange"] * (len(labels) - 2)
        bars = axB.bar(xB, np.abs(values), color=colors)
        for bar, v in zip(bars, values):
            axB.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.02 if bar.get_height() > 0 else 1.0,
                f"{v:.2e}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
            if v < 0:
                bar.set_edgecolor("red")
                bar.set_linewidth(2)

        axB.set_xticks(xB)
        axB.set_xticklabels(labels, rotation=30, ha="right")
        axB.set_ylabel("Flux [H/s]")
        axB.set_title(f"Flux breakdown at {int(Tc)} °C")
        axB.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        from matplotlib.patches import Patch

        axB.legend(
            handles=[
                Patch(color="tab:blue", label="Experiment"),
                Patch(color="tab:orange", label=f"Model (Φ={phi_here:.3e})"),
            ],
            loc="best",
        )
        figB.tight_layout()
        plt.show()
