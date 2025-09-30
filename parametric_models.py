import festim as F
from mesh import generate_mesh
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmshio
from mpi4py import MPI
import festim as F
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
) -> Tuple[F.HydrogenTransportProblemDiscontinuous, dict[str, list[CylindricalFlux]]]:
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

    # time step & final time (seconds) for transient run
    dt = F.Stepsize(
        initial_value=10, growth_factor=1.1, cutback_factor=0.9, target_nb_iterations=4
    )
    t_total = 6e4

    # transient BCs using time-dependent pressure for downstream sides

    upstream_surface_bcs = [
        F.SievertsBC(
            subdomain=s,
            species=H,
            pressure=P_up,
            S_0=K_S_nickel.pre_exp.magnitude,
            E_S=K_S_nickel.act_energy.magnitude,
        )
        for s in upstream_volume_surfaces
    ]
    downstream_surface_bcs = [
        F.FixedConcentrationBC(
            subdomain=s, species=H, value=0.0
        )  # downstream partial pressure is ~5 Pa << P_up ~1e5 Pa
        for s in downstream_volume_surfaces_Ni + [Liquid_top]
    ]

    my_model.boundary_conditions = upstream_surface_bcs + downstream_surface_bcs

    # transient=True and set stepsize/final_time
    my_model.settings = F.Settings(
        atol=1e12,
        rtol=1e-13,
        transient=True,
        stepsize=dt,
        final_time=t_total,
    )

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
            field=H, filename="FLiBe_infinite_PRF_solid.bp", subdomain=solid_volume
        ),
        F.VTXSpeciesExport(
            field=H, filename="FLiBe_infinite_PRF_liquid.bp", subdomain=fluid_volume
        ),
    ]
    # add flux monitors to exports so they are evaluated at every time step
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


if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)
    # filter nickel and H
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")

    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    D_flibe = htm.Diffusivity(D_0=3.1e-7, E_D=0.52)
    permeability_flibe = htm.Permeability(pre_exp=1.2e-3, act_energy=0.0, law="henry")

    plt.figure(figsize=(12, 6))
    for temperature in [500, 700]:
        my_model, all_fluxes = make_model(
            temperature=temperature,
            D_flibe=D_flibe,
            D_nickel=D_nickel,
            permeability_flibe=permeability_flibe,
            K_S_nickel=K_S_nickel,
            P_up=1.11e5,
            case="prf_infinite",  # ignored for now
        )

        # time step & final time (seconds) for transient run

        my_model.initialise()
        my_model.run()

        fluxes_in = all_fluxes["fluxes_in"]
        glovebox_flux = all_fluxes["glovebox_flux"]
        flux_out_liquid = all_fluxes["flux_out_liquid"]
        flux_out_Ni_sidewall = all_fluxes["flux_out_Ni_sidewall"]
        flux_out_top_Ni_bottom = all_fluxes["flux_out_top_Ni_bottom"]

        t = np.asarray(glovebox_flux.t, dtype=float)  # seconds

        v_glovebox = np.asarray(glovebox_flux.data, dtype=float)
        v_out_liquid = np.asarray(flux_out_liquid.data, dtype=float)
        v_downstream_sidewall = np.asarray(flux_out_Ni_sidewall.data, dtype=float)
        v_downstream_top = np.asarray(flux_out_top_Ni_bottom.data, dtype=float)

        v_downstream_total = v_out_liquid + v_downstream_sidewall + v_downstream_top

        v_in_terms = [np.asarray(f.data, dtype=float) for f in fluxes_in]

        assert all(len(x) == len(t) for x in v_in_terms), (
            "Inlet flux series have mismatched lengths."
        )
        v_in_total = np.sum(np.stack(v_in_terms, axis=0), axis=0)

        v_balance = v_in_total + v_glovebox + v_downstream_total

        plt.plot(t, v_out_liquid + v_downstream_sidewall, label=f"T={temperature} K")

    plt.xlabel("Time (s)")
    plt.ylabel("Flux [H/s]")
    plt.title("Downstream flux")
    plt.legend(loc="best", ncol=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.show()
