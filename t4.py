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
    "mesh.msh", MPI.COMM_WORLD, model_rank
)

# filter nickel and H
diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")
solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")

# material parameters for Nickel
D_solid = diffusivities_nickel[0].pre_exp.magnitude  # m^2/s
E_D_solid = diffusivities_nickel[0].act_energy.magnitude  # ev/particle
K_solid = solubilities_nickel[0].pre_exp.magnitude  # particle m^-3 Pa^-0.5
E_K_S_solid = solubilities_nickel[0].act_energy.magnitude  # ev/particle


# print(f"K_solid: {K_solid:.3e}, E_K_S_solid: {E_K_S_solid:.3e}")
# print(f"D_solid: {D_solid:.3e}, E_D_solid: {E_D_solid:.3e}")

# material parameters for FLiBe
diffusivities_flibe = htm.diffusivities.filter(material="flibe").filter(isotope="h")
solubilities_flibe = htm.solubilities.filter(material="flibe").filter(isotope="h")

D_liquid = diffusivities_flibe[0].pre_exp.magnitude  # m^2/s
E_D_liquid = diffusivities_flibe[0].act_energy.magnitude  # ev/particle
K_liquid = solubilities_flibe[0].pre_exp.magnitude  # particle m^-3 Pa^-1
E_K_S_liquid = solubilities_flibe[
    0
].act_energy.magnitude  # ev/particle. NOTE: This is a negative value.

# print(diffusivities_flibe[0])
# print(f"K_liquid: {K_liquid:.3e}, E_K_S_liquid: {E_K_S_liquid:.3e}")
# print(f"D_liquid: {D_liquid:.3e}, E_D_liquid: {E_D_liquid:.3e}")

# exit()
# Define materials

mat_solid = F.Material(
    D_0=D_solid,
    E_D=E_D_solid,
    K_S_0=K_solid,
    E_K_S=E_K_S_solid,
    solubility_law="sievert",
)
mat_liquid = F.Material(
    D_0=D_liquid,
    E_D=E_D_liquid,
    K_S_0=K_liquid,
    E_K_S=E_K_S_liquid,
    solubility_law="henry",
)

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

# we need to pass the meshtags to the model directly
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
    id=99, subdomains=[solid_volume, fluid_volume], penalty_term=1e24
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

my_model.temperature = 773

upstream_volume_surfaces = [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]

downstream_volume_surfaces = [top_cap_Ni, top_sidewall_Ni]

# case 1: Outside BC as fixed concentration
# out_surface_bc = F.FixedConcentrationBC(subdomain=out_surf, species=H, value=0.0)

# case 2: Outside BC as isolated (no flux)
out_surface_bc = F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)

# case 3:
# glovebox BC as sieverts with P = 1 pa
# typical H2 partial pressure in glovebox is between 1e-6 atm to 1e-4 atm
# here we take 1 pa as an example
# p_glovebox = 10  # Pa
# out_surface_bc = F.SievertsBC(
#     subdomain=out_surf, species=H, pressure=p_glovebox, S_0=K_solid, E_S=E_K_S_solid
# )

# out_surface_bc = F.SurfaceReactionBC(
#     reactant=[H, H],
#     gas_pressure=1,
#     k_r0=1e-20,
#     E_kr=0.2,
#     k_d0=1e-4,
#     E_kd=0.1,
#     subdomain=out_surf,
# )

P_up = 1e5  # Pa

my_model.boundary_conditions = (
    [
        F.SievertsBC(
            subdomain=s, species=H, pressure=P_up, S_0=K_solid, E_S=E_K_S_solid
        )  ###NOTE: E_s can not be 0.
        for s in upstream_volume_surfaces
    ]
    + [out_surface_bc]
    + [
        F.FixedConcentrationBC(subdomain=s, species=H, value=0.0)
        for s in downstream_volume_surfaces
    ]
    +    [
        F.HenrysBC(
            subdomain=s, species=H, pressure=P_up, H_0=K_solid, E_H=E_K_S_solid
        )  ###NOTE: E_s can not be 0.
        for s in [liquid_surface]
    ]
)


my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)


fluxes_in = [
    CylindricalFlux(field=H, surface=surf) for surf in upstream_volume_surfaces
]
downstream_fluxes = [
    CylindricalFlux(field=H, surface=surf) for surf in downstream_volume_surfaces
]
glovebox_flux = CylindricalFlux(field=H, surface=out_surf)

flux_out_liquid = CylindricalFlux(field=H, surface=liquid_surface)
flux_out_Ds_ni_left = CylindricalFlux(field=H, surface=top_sidewall_Ni)
flux_out_top_cap_Ni = CylindricalFlux(field=H, surface=top_cap_Ni)

# my_model.exports = [
#     F.VTXSpeciesExport(
#         field=H, filename="out-species_solid_case_1.bp", subdomain=solid_volume
#     ),
#     F.VTXSpeciesExport(
#         field=H, filename="out-species_fluid_case_1.bp", subdomain=fluid_volume
#     ),
# ]
# my_model.exports = [
#     F.VTXSpeciesExport(
#         field=H, filename="out-species_solid_case_2.bp", subdomain=solid_volume
#     ),
#     F.VTXSpeciesExport(
#         field=H, filename="out-species_fluid_case_2.bp", subdomain=fluid_volume
#     ),
# ]
my_model.exports = [
    F.VTXSpeciesExport(
        field=H, filename="out-species_solid_case_3.bp", subdomain=solid_volume
    ),
    F.VTXSpeciesExport(
        field=H, filename="out-species_fluid_case_3.bp", subdomain=fluid_volume
    ),
]
my_model.exports += downstream_fluxes
my_model.exports += fluxes_in
my_model.exports += [glovebox_flux]
my_model.exports += [flux_out_liquid]
my_model.exports += [flux_out_Ds_ni_left]
my_model.exports += [flux_out_top_cap_Ni]

my_model.initialise()
my_model.run()


total_flux_glovebox = glovebox_flux.value
total_downstream_flux = sum(flux.value for flux in downstream_fluxes)
total_flux_in = sum(flux.value for flux in fluxes_in)


print(f"Flux in: {total_flux_in:.4e} H/s")
print(f"Flux out (glovebox): {total_flux_glovebox:.4e} H/s")
print(f"Flux balance: {total_flux_in + total_flux_glovebox:.4e} H/s")
print("-----")
print(f"Total downstream flux: {total_downstream_flux:.4e} H/s")
print(f"flux through liquid surface: {flux_out_liquid.value:.4e} H/s")
print(f"flux through nickel downstream vertically: {flux_out_Ds_ni_left.value:.4e} H/s")
print(f"flux through nickel downstream bottom: {flux_out_top_cap_Ni.value:.4e} H/s")

# Collect values into a dict
fluxes = {
    "Flux in": float(total_flux_in),
    "Flux out (glovebox)": float(total_flux_glovebox),
    "Flux balance": float(total_flux_in + total_flux_glovebox),
    "Total downstream flux": float(total_downstream_flux),
    "Liquid surface": float(flux_out_liquid.value),
    "Ni downstream (vertical)": float(flux_out_Ds_ni_left.value),
}

labels = list(fluxes.keys())
values = np.array([fluxes[k] for k in labels], dtype=float)

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(10, 5))

# Colors: blue for positive, red for negative
colors = ["blue" if v >= 0 else "red" for v in values]

# Bar plot with absolute values
ax.bar(x, np.abs(values), color=colors)

# Labeling
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("Flux [H/s]")
ax.set_title("fluxes in the system, red bars are negative values")

# Logarithmic y-scale
ax.set_yscale("log")

fig.tight_layout()
plt.show()


# replace this based on what we see in paraview
# c_flibe = 1.32e25
# c_ni = 2.17e25

# K_s_ni = solubilities_nickel[0].value(my_model.temperature)  # particle m^-3 Pa^-0.5
# K_s_flibe = solubilities_flibe[0].value(my_model.temperature)  # particle m^-3 Pa^-1

# expected_c_flibe = (c_ni / K_s_ni) ** 2 * K_s_flibe

# print("Expected concentration in FLiBe (particle/m^3): ", expected_c_flibe)
# print("Real concentration in FLiBe (particle/m^3): ", c_flibe)
