from mesh import generate_mesh
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmshio
from mpi4py import MPI
import festim as F
import h_transport_materials as htm

import matplotlib.pyplot as plt
import numpy as np

set_log_level(LogLevel.INFO)

generate_mesh(mesh_size=2e-4)
model_rank = 0
mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "mesh_solid_only.msh", MPI.COMM_WORLD, model_rank
)

# filter nickel and H
diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")
solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")

# material parameters for Nickel
# D_solid = diffusivities_nickel[0].pre_exp.magnitude  # m^2/s
# E_D_solid = diffusivities_nickel[0].act_energy.magnitude  # ev/particle
# K_solid = solubilities_nickel[0].pre_exp.magnitude  # particle m^-3 Pa^-0.5
# E_K_S_solid = solubilities_nickel[0].act_energy.magnitude  # ev/particle

# print("solubilities_nickel")
# print(solubilities_nickel[0].value(773))
# print("diffusivities_nickel")
# print(diffusivities_nickel[0].value(773))
# print("permeabilities_nickel")
# print(solubilities_nickel[0].value(773) * diffusivities_nickel[0].value(773))
# exit()

D_solid = diffusivities_nickel[0].pre_exp.magnitude  # m^2/s
E_D_solid = diffusivities_nickel[0].act_energy.magnitude  # ev/particle
K_solid = solubilities_nickel[0].pre_exp.magnitude  # particle m^-3 Pa^-0.5
E_K_S_solid = solubilities_nickel[0].act_energy.magnitude  # ev/particle

# print("solubilities_nickel")
# print(solubilities_nickel[0])
# print("diffusivities_nickel")
# print(diffusivities_nickel[1])
# # print("permeabilities_nickel")
# # print(solubilities_nickel[0].value(773) * diffusivities_nickel[1].value(773))
# exit()

mat_solid = F.Material(
    D_0=D_solid,
    E_D=E_D_solid,
    K_S_0=K_solid,
    E_K_S=E_K_S_solid,
    solubility_law="sievert",
)

solid_volume = F.VolumeSubdomain(id=2, material=mat_solid)

out_surf = F.SurfaceSubdomain(id=3)
# left_bc_liquid = F.SurfaceSubdomain(id=41)
left_bc_top_Ni = F.SurfaceSubdomain(id=42)
left_bc_middle_Ni = F.SurfaceSubdomain(id=43)
left_bc_bottom_Ni = F.SurfaceSubdomain(id=44)
top_Ni_bottom = F.SurfaceSubdomain(id=5)
Ds_Ni_left = F.SurfaceSubdomain(id=6)
Up_Ni_left = F.SurfaceSubdomain(id=7)
mem_Ni_top = F.SurfaceSubdomain(id=8)
mem_Ni_bottom = F.SurfaceSubdomain(id=9)
bottom_Ni_top = F.SurfaceSubdomain(id=10)
liquid_solid_interface = F.SurfaceSubdomain(id=99)


my_model = F.HydrogenTransportProblemDiscontinuous()

my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")

# we need to pass the meshtags to the model directly
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [
    solid_volume,
    # fluid_volume,
    out_surf,
    # left_bc_liquid,
    left_bc_top_Ni,
    left_bc_middle_Ni,
    left_bc_bottom_Ni,
    top_Ni_bottom,
    Ds_Ni_left,
    Up_Ni_left,
    mem_Ni_top,
    mem_Ni_bottom,
    bottom_Ni_top,
    liquid_solid_interface,
]

my_model.surface_to_volume = {
    out_surf: solid_volume,
    # left_bc_liquid: solid_volume,
    left_bc_top_Ni: solid_volume,
    left_bc_middle_Ni: solid_volume,
    left_bc_bottom_Ni: solid_volume,
    top_Ni_bottom: solid_volume,
    Ds_Ni_left: solid_volume,
    Up_Ni_left: solid_volume,
    mem_Ni_top: solid_volume,
    mem_Ni_bottom: solid_volume,
    bottom_Ni_top: solid_volume,
}

H = F.Species("H", subdomains=my_model.volume_subdomains)
my_model.species = [H]

my_model.temperature = 773

upstream_volume_surfaces = [mem_Ni_bottom, bottom_Ni_top, Up_Ni_left]

downstream_volume_surfaces = [top_Ni_bottom, Ds_Ni_left, mem_Ni_top]


# case 2: Outside BC as isolated (no flux)
out_surface_bc = F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)

# p_glovebox = 1  # Pa
# out_surface_bc = F.SievertsBC(
#     subdomain=out_surf, species=H, pressure=p_glovebox, S_0=K_solid, E_S=E_K_S_solid
# )

P_up = 1.30e5  # Pa
P_down = 1.98e2  # Pa
my_model.boundary_conditions = (
    [
        F.SievertsBC(
            subdomain=s, species=H, pressure=P_up, S_0=K_solid, E_S=E_K_S_solid
        )  ###NOTE: E_s can not be 0.
        for s in upstream_volume_surfaces
    ]
    + [out_surface_bc]
    + [
        F.SievertsBC(
            subdomain=s, species=H, pressure=P_down, S_0=K_solid, E_S=E_K_S_solid
        )
        for s in downstream_volume_surfaces
    ]
)

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)


fluxes_in = [
    CylindricalFlux(field=H, surface=surf) for surf in upstream_volume_surfaces
]
downstream_fluxes = [
    CylindricalFlux(field=H, surface=surf) for surf in downstream_volume_surfaces
]
glovebox_flux = CylindricalFlux(field=H, surface=out_surf)

flux_out_membrane = CylindricalFlux(field=H, surface=mem_Ni_top)
flux_out_Ds_ni_left = CylindricalFlux(field=H, surface=Ds_Ni_left)
flux_out_top_Ni_bottom = CylindricalFlux(field=H, surface=top_Ni_bottom)


my_model.exports = [
    F.VTXSpeciesExport(
        field=H, filename="out-species_dry_run.bp", subdomain=solid_volume
    ),
]
my_model.exports += downstream_fluxes
my_model.exports += fluxes_in
my_model.exports += [glovebox_flux]
my_model.exports += [flux_out_membrane]
my_model.exports += [flux_out_Ds_ni_left]
my_model.exports += [flux_out_top_Ni_bottom]

my_model.initialise()
my_model.run()


total_flux_glovebox = glovebox_flux.value
total_downstream_flux = sum(flux.value for flux in downstream_fluxes)
total_flux_in = sum(flux.value for flux in fluxes_in)


print(f"Flux in: {total_flux_in:.4e} H/s")
print(f"Flux out to glovebox: {total_flux_glovebox:.4e} H/s")
print(f"Total downstream flux: {total_downstream_flux:.4e} H/s")
print(f"flux through membrane top: {flux_out_membrane.value:.4e} H/s")
print(f"flux through nickel downstream sidewall: {flux_out_Ds_ni_left.value:.4e} H/s")
print(
    f"flux through nickel downstream top surface: {flux_out_top_Ni_bottom.value:.4e} H/s"
)
print("-----")
print(
    f"Flux balance: {total_flux_in + total_flux_glovebox + total_downstream_flux:.4e} H/s"
)

# Build data
fluxes = {
    "experiment run 1 (P=5.557e13)": 4.5268e16,  # experimental value from run 1
    "experiment run 2 (P=5.457e13)": 4.0832e16,  # experimental value from run 2
    "Flux in (upstream)": float(total_flux_in),
    "Flux out to glovebox": float(total_flux_glovebox),
    "Total downstream flux": float(total_downstream_flux),
    "Flux through membrane surface": float(flux_out_membrane.value),
    "Flux through Ni downstream sidewall": float(flux_out_Ds_ni_left.value),
    "Flux through Ni downstream top surface": float(flux_out_top_Ni_bottom.value),
    "Flux balance (in + glovebox + downstream)": float(
        total_flux_in + total_flux_glovebox + total_downstream_flux
    ),
}

labels = list(fluxes.keys())
values = np.array([fluxes[k] for k in labels], dtype=float)

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12, 6))

# Colors: blue for ≥0, red for <0
colors = ["blue" if v >= 0 else "red" for v in values]

heights = np.abs(values)
bars = ax.bar(x, heights, color=colors)

y_offset = 0.05 * np.max(heights) if np.max(heights) > 0 else 1.0
for i, (bar, v) in enumerate(zip(bars, values)):
    y = bar.get_height() + (y_offset if v >= 0 else y_offset)
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        y,
        f"{v:.2e}",
        ha="center",
        va="bottom",
        rotation=0,
        fontsize=18,
    )

# Axes labels / ticks
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=18)
ax.set_ylabel("Flux [H/s]", fontsize=18)
ax.tick_params(axis="y", labelsize=18)
ax.yaxis.get_offset_text().set_fontsize(18)

# Permeability shown in the title (evaluate at 773 K)
P_773 = (
    solubilities_nickel[0].value(773) * diffusivities_nickel[0].value(773)
).magnitude
ax.set_title(
    f"Fluxes in the system (red = negative). Permeability at 773 K: {P_773:.3e}",
    fontsize=18,
)

ax.margins(y=0.15)
fig.tight_layout()
plt.show()
