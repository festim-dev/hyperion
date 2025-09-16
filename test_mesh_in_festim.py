from mesh import generate_mesh


generate_mesh(mesh_size=2e-4)


from dolfinx.io import gmshio
from mpi4py import MPI

model_rank = 0

mesh, cell_tags, facet_tags = gmshio.read_from_msh(
    "mesh.msh", MPI.COMM_WORLD, model_rank
)

# import festim as F

# material = F.Material(D_0=1, E_D=0)

# top_volume = F.VolumeSubdomain(id=1, material=material)
# bottom_volume = F.VolumeSubdomain(id=2, material=material)

# tube_surf = F.SurfaceSubdomain(id=3)

# my_model = F.HydrogenTransportProblem()

# my_model.mesh = F.Mesh(mesh)

# # we need to pass the meshtags to the model directly
# my_model.facet_meshtags = facet_tags
# my_model.volume_meshtags = cell_tags

# my_model.subdomains = [top_volume, bottom_volume, tube_surf]

# H = F.Species("H")
# my_model.species = [H]

# my_model.temperature = 400

# my_model.boundary_conditions = [
#     F.FixedConcentrationBC(subdomain=tube_surf, value=1, species=H),
# ]

# my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

# my_model.exports = [
#     F.VTXSpeciesExport(field=H, filename="out-species.bp"),
# ]

# my_model.initialise()
# my_model.run()


# from dolfinx.io import XDMFFile

# with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as xdmf:
#     xdmf.write_mesh(mesh)
#     xdmf.write_meshtags(cell_tags, mesh.geometry)
# exit()

from dolfinx import plot
import pyvista

fdim = mesh.topology.dim - 1
tdim = mesh.topology.dim
mesh.topology.create_connectivity(fdim, tdim)
topology, cell_types, x = plot.vtk_mesh(mesh, fdim, facet_tags.indices)

p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Facet Marker"] = facet_tags.values
grid.set_active_scalars("Facet Marker")
p.add_mesh(grid, show_edges=True)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("facet_marker.png")
p.show()


topology, cell_types, x = plot.vtk_mesh(mesh, tdim, cell_tags.indices)
p = pyvista.Plotter()
grid = pyvista.UnstructuredGrid(topology, cell_types, x)
grid.cell_data["Cell Marker"] = cell_tags.values
grid.set_active_scalars("Cell Marker")
p.add_mesh(grid, show_edges=False)
if pyvista.OFF_SCREEN:
    figure = p.screenshot("cell_marker.png")
p.show()
