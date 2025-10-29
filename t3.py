from mesh import generate_mesh
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmshio
from mpi4py import MPI
import festim as F
import h_transport_materials as htm

import matplotlib.pyplot as plt
import numpy as np
import csv

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
# based on the dry run, the lower permeability seems to match better
# so we use diffusivities_nickel[1]
D_solid = diffusivities_nickel[1].pre_exp.magnitude  # m^2/s
E_D_solid = diffusivities_nickel[1].act_energy.magnitude  # ev/particle
K_solid = solubilities_nickel[0].pre_exp.magnitude  # particle m^-3 Pa^-0.5
E_K_S_solid = solubilities_nickel[0].act_energy.magnitude  # ev/particle

# material parameters for FLiBe
# we will fix the diffusivity based on literature, and change the solubility to match the experiment flux curve
diffusivities_flibe = htm.diffusivities.filter(material="flibe").filter(isotope="h")
solubilities_flibe = htm.solubilities.filter(material="flibe").filter(isotope="h")

D_liquid = diffusivities_flibe[0].pre_exp.magnitude  # m^2/s
E_D_liquid = diffusivities_flibe[0].act_energy.magnitude  # ev/particle
K_liquid = solubilities_flibe[0].pre_exp.magnitude  # particle m^-3 Pa^-1
E_K_S_liquid = solubilities_flibe[
    0
].act_energy.magnitude  # ev/particle. NOTE: This is a negative value.


print("solubilities_flibe")
print(solubilities_flibe[0])
print("diffusivities_flibe")
print(diffusivities_flibe[0])
# print("solubilities_FLiBe")
# print(solubilities_flibe[0].value(773))
# print("diffusivities_FLiBe")
# print(diffusivities_flibe[0].value(773))
# print("permeabilities_FLiBe")
# print(solubilities_flibe[0].value(773) * diffusivities_flibe[0].value(773))

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
    left_bc_liquid: fluid_volume,  # NOTE: this is fluid
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
downstream_volume_surfaces_Ni = [ top_cap_Ni, top_sidewall_Ni]
downstream_volume_surfaces = [ top_cap_Ni, top_sidewall_Ni, liquid_surface]

# constant upstream pressure
P_up = 1.11e5  # Pa


# time step & final time (seconds) for transient run
dt = F.Stepsize(
    initial_value=10, growth_factor=1.1, cutback_factor=0.9, target_nb_iterations=4
)
t_total = 6e4

# transient BCs using time-dependent pressure for downstream sides

upstream_surface_bcs = [
    F.SievertsBC(
        subdomain=s, species=H, pressure=P_up, S_0=K_solid, E_S=E_K_S_solid
    )  ###NOTE: E_s can not be 0.
    for s in upstream_volume_surfaces
]
downstream_surface_bcs = [
    F.FixedConcentrationBC(
        subdomain=s, species=H, value=0.0
    )  # downstream partial pressure is ~5 Pa << P_up ~1e5 Pa
    for s in downstream_volume_surfaces_Ni + [liquid_surface]
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
flux_out_liquid = CylindricalFlux(field=H, surface=liquid_surface)
flux_out_Ni_sidewall = CylindricalFlux(field=H, surface=top_sidewall_Ni)
flux_out_top_cap_Ni = CylindricalFlux(field=H, surface=top_cap_Ni)

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
    flux_out_top_cap_Ni,
]

# -------- initialise & run transient  --------
my_model.initialise()
my_model.show_progress_bar = True
my_model.run()

t = np.asarray(glovebox_flux.t, dtype=float)  # seconds

v_glovebox = np.asarray(glovebox_flux.data, dtype=float)
v_out_liquid = np.asarray(flux_out_liquid.data, dtype=float)
v_downstream_sidewall = np.asarray(flux_out_Ni_sidewall.data, dtype=float)
v_downstream_top = np.asarray(flux_out_top_cap_Ni.data, dtype=float)

v_downstream_total = v_out_liquid + v_downstream_sidewall + v_downstream_top

v_in_terms = [np.asarray(f.data, dtype=float) for f in fluxes_in]

assert all(len(x) == len(t) for x in v_in_terms), (
    "Inlet flux series have mismatched lengths."
)
v_in_total = np.sum(np.stack(v_in_terms, axis=0), axis=0)


v_balance = v_in_total + v_glovebox + v_downstream_total


all_fluxes = {
    "Flux_in_total": v_in_total,
    "Flux_downstream_total": v_downstream_total,
    "Flux_glovebox": v_glovebox,
    "Flux_liquid_top": v_out_liquid,
    "Flux_Ni_sidewall": v_downstream_sidewall,
    "Flux_Ni_top": v_downstream_top,
    "Flux_balance": v_balance,
}

with open("all_flux_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["time [s]"] + list(all_fluxes.keys())
    writer.writerow(header)

    for i in range(len(t)):
        row = [t[i]] + [vals[i] for vals in all_fluxes.values()]
        writer.writerow(row)


plt.figure(figsize=(12, 6))
# plt.plot(t, v_in_total, label="Flux into upstream space")
plt.plot(t, v_downstream_total, label="Flux outside downstream space")
# plt.plot(t, v_glovebox, label="Flux to glovebox")
plt.plot(t, v_out_liquid, label="Flux through liquid top")
plt.plot(t, v_downstream_sidewall, label="Flux through Ni downstream sidewall")
# plt.plot(t, v_downstream_top, label="Flux through Ni downstream top")
# plt.plot(t, v_balance, "--", label="Flux balance (in + glovebox + downstream)")

plt.xlabel("Time (s)")
plt.ylabel("Flux [H/s]")
plt.title("All flux monitors vs time")
plt.legend(loc="best", ncol=2)
plt.tight_layout()
plt.show()
