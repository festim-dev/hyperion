from mesh import generate_mesh
from dolfinx.log import set_log_level, LogLevel
set_log_level(LogLevel.INFO)

generate_mesh(mesh_size=2e-4)


from dolfinx.io import gmshio
from mpi4py import MPI

model_rank = 0

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "mesh.msh", MPI.COMM_WORLD, model_rank
)

import festim as F
# print(F.__version__)


import h_transport_materials as htm

#filter nickel and H
diffusivities_nickel= htm.diffusivities.filter(material="nickel").filter(isotope="h")
solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")

#material parameters for Nickel
D_solid= diffusivities_nickel[0].pre_exp.magnitude           #m^2/s
E_D_solid= diffusivities_nickel[0].act_energy.magnitude      #ev/particle
K_solid= solubilities_nickel[0].pre_exp.magnitude            #particle m^-3 Pa^-0.5
E_K_S_solid = solubilities_nickel[0].act_energy.magnitude    #ev/particle

#material parameters for FLiBe
diffusivities_flibe= htm.diffusivities.filter(material="flibe").filter(isotope="h")
solubilities_flibe = htm.solubilities.filter(material="flibe").filter(isotope="h")

D_liquid = diffusivities_flibe[0].pre_exp.magnitude           #m^2/s
E_D_liquid = diffusivities_flibe[0].act_energy.magnitude      #ev/particle
K_liquid = solubilities_flibe[0].pre_exp.magnitude            #particle m^-3 Pa^-1
E_K_S_liquid = solubilities_flibe[0].act_energy.magnitude     #ev/particle. NOTE: This is a negative value?

# Define materials

mat_solid  = F.Material(D_0=D_solid,  E_D=E_D_solid, K_S_0=K_solid,  E_K_S=E_K_S_solid, solubility_law="sievert")
mat_liquid = F.Material(D_0=D_liquid, E_D=E_D_liquid, K_S_0=K_liquid, E_K_S=E_K_S_liquid, solubility_law="henry")

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

# we need to pass the meshtags to the model directly
my_model.facet_meshtags = facet_tags
my_model.volume_meshtags = cell_tags

my_model.subdomains = [solid_volume, fluid_volume, out_surf, 
                       left_bc_liquid,left_bc_top_Ni, left_bc_middle_Ni,left_bc_bottom_Ni,
                       top_Ni_bottom, Ds_Ni_left, Up_Ni_left,
                       Liquid_top, mem_Ni_bottom, bottom_Ni_top,
                       liquid_solid_interface]

my_model.method_interface = "penalty"
interface = F.Interface(id=99, subdomains=[solid_volume, fluid_volume], penalty_term=1e28)

my_model.interfaces = [interface]

my_model.surface_to_volume = {
    out_surf: solid_volume,
    left_bc_liquid: solid_volume,
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

my_model.temperature = 773

upstream_volume_surfaces = [
    mem_Ni_bottom, bottom_Ni_top, Up_Ni_left
]

downstream_volume_surfaces = [
    top_Ni_bottom, Ds_Ni_left, Liquid_top
]

out_surface_bc = F.FixedConcentrationBC(
    subdomain=out_surf, species=H, value=0.0
)

P_up = 1e5  # Pa

my_model.boundary_conditions = [
   F.SievertsBC(subdomain=s, species=H, pressure=P_up, S_0=K_solid, E_S=E_K_S_solid)   ###NOTE: E_s can not be 0.
    for s in upstream_volume_surfaces
] + [out_surface_bc] + [
    F.FixedConcentrationBC(subdomain=s, species=H, value=0.0)
    for s in downstream_volume_surfaces]

my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)




# import ufl
# import numpy as np
# import dolfinx
# from mpi4py import MPI
# # here we define our own SurfaceFlux class that takes into account the cylindrical coordinate system


# class CylindricalFlux(F.SurfaceFlux):
#     azimuth_range: tuple[float, float] = (0.0, 2 * np.pi)

#     def compute(
#         self,
#         u: dolfinx.fem.Function | ufl.indexed.Indexed,
#         ds: ufl.Measure,
#         entity_maps=None,
#     ):
#         """Computes the value of the flux at the surface

#         J =int(- D * grad(c) . n * r dS)

#         Args:
#             u: field for which the flux is computed
#             ds: surface measure of the model
#             entity_maps: entity maps relating parent mesh and submesh
#         """
#         from scifem import assemble_scalar

#         # obtain mesh normal from field
#         # if case multispecies, solution is an index, use sub_function_space
#         if isinstance(u, ufl.indexed.Indexed):
#             mesh = self.field.sub_function_space.mesh
#         else:
#             mesh = u.function_space.mesh
#         n = ufl.FacetNormal(mesh)
#         x = ufl.SpatialCoordinate(mesh)
#         r = x[0]

#         flux = assemble_scalar(
#             dolfinx.fem.form(
#                 -self.D * r * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
#                 entity_maps=entity_maps,
#             )
#         )

#         flux *= self.azimuth_range[1] - self.azimuth_range[0]

#         self.value = flux
#         self.data.append(self.value)



# flux_in = CylindricalFlux(field=H, surface=upstream_volume_surfaces)
# flux_out_1 = CylindricalFlux(field=H, surface=downstream_volume_surfaces)
# flux_out_2 = CylindricalFlux(field=H, surface=out_surf)
# flux_out = F.CombinedFlux([flux_out_1, flux_out_2])

my_model.exports = [
    F.VTXSpeciesExport(field=H, filename="out-species_solid.bp", subdomain=solid_volume),
    F.VTXSpeciesExport(field=H, filename="out-species_fluid.bp", subdomain=fluid_volume),
]#+ [flux_in, flux_out]

my_model.initialise()
my_model.run()

# print(f"In flux: {flux_in.value}")
# print(f"Out flux: {flux_out.value}")

# # from dolfinx.io import XDMFFile

# # with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
# #     xdmf.write_mesh(mesh)
# #     xdmf.write_meshtags(cell_tags, mesh.geometry)
# # #exit()

# from dolfinx import plot
# import pyvista

# # fdim = mesh.topology.dim - 1
# # tdim = mesh.topology.dim
# # mesh.topology.create_connectivity(fdim, tdim)
# # topology, cell_types, x = plot.vtk_mesh(mesh, fdim, facet_tags.indices)

# # p = pyvista.Plotter()
# # grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# # grid.cell_data["Facet Marker"] = facet_tags.values
# # grid.set_active_scalars("Facet Marker")
# # p.add_mesh(grid, show_edges=True)
# # if pyvista.OFF_SCREEN:
# #     figure = p.screenshot("facet_marker.png")
# # p.show()


# topology, cell_types, x = plot.vtk_mesh(mesh, tdim, cell_tags.indices)
# p = pyvista.Plotter()
# grid = pyvista.UnstructuredGrid(topology, cell_types, x)
# grid.cell_data["Cell Marker"] = cell_tags.values
# grid.set_active_scalars("Cell Marker")
# p.add_mesh(grid, show_edges=False)
# if pyvista.OFF_SCREEN:
#     figure = p.screenshot("cell_marker.png")
# p.show()

# replace this based on what we see in paraview
c_flibe = 1.32e25
c_ni = 2.17e25

K_s_ni = solubilities_nickel[0].value(my_model.temperature)   # particle m^-3 Pa^-0.5
K_s_flibe = solubilities_flibe[0].value(my_model.temperature)  # particle m^-3 Pa^-1

expected_c_flibe = (c_ni / K_s_ni)**2 * K_s_flibe 

print("Expected concentration in FLiBe (particle/m^3): ", expected_c_flibe)
print("Real concentration in FLiBe (particle/m^3): ", c_flibe)