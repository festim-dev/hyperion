# --- Domain = SOLID ∪ LIQUID only (void removed) and void contacts become external BCs ---
import festim as F
import numpy as np
import dolfinx
from dolfinx.mesh import CellType
from dolfinx import mesh as _mesh, fem as _fem
import ufl
from mpi4py import MPI

# ---------------------------
# Geometry (meters)
# ---------------------------
x_in   = 0.468
x_out  = 0.492
y_bT   = 0.024
y_mB   = 0.264
y_mT   = 0.288
y_fT   = 0.3828
y_tIn  = 1.3092
y_tOut = 1.3332

# ---------------------------
# Helper
# ---------------------------
def in_range(z, a, b, eps=1e-12):
    """Closed interval test with a small tolerance."""
    return (z >= a - eps) & (z <= b + eps)

# ---------------------------
# 1) Build a base mesh on the full bounding box
# ---------------------------
nx, ny = 240, 650
base_mesh = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0.0, 0.0]), np.array([x_out, y_tOut])],
    [nx, ny],
    cell_type=CellType.triangle,
)
tdim = base_mesh.topology.dim

# ---------------------------
# 2) Keep only SOLID + LIQUID cells -> SUBMESH
# ---------------------------
def solid_cells(m):
    return dolfinx.mesh.locate_entities(
        m, tdim,
        lambda X: (
            in_range(X[1], 0.0, y_bT)                                       # bottom plate
            | (in_range(X[0], x_in, x_out) & in_range(X[1], y_bT, y_tIn))  # right wall
            | (in_range(X[1], y_mB, y_mT) & in_range(X[0], 0.0, x_in))     # thin mid plate
            | (in_range(X[1], y_tIn, y_tOut) & in_range(X[0], 0.0, x_out)) # top plate
        )
    )

def liquid_cells(m):
    return dolfinx.mesh.locate_entities(
        m, tdim,
        lambda X: in_range(X[0], 0.0, x_in) & in_range(X[1], y_mT, y_fT)
    )

keep = np.unique(np.hstack([solid_cells(base_mesh), liquid_cells(base_mesh)]).astype(np.int32))
try:
    submesh, cell_map, *_ = dolfinx.mesh.create_submesh(base_mesh, tdim, keep)
except ValueError:
    submesh, cell_map = dolfinx.mesh.create_submesh(base_mesh, tdim, keep)

# ---------------------------
# 3) Volume subdomains on the SUBMESH
# ---------------------------
class SolidVolume(F.VolumeSubdomain):
    def locate_subdomain_entities(self, m):
        return dolfinx.mesh.locate_entities(
            m, m.topology.dim,
            lambda X: (
                in_range(X[1], 0.0, y_bT)
                | (in_range(X[0], x_in, x_out) & in_range(X[1], y_bT, y_tIn))
                | (in_range(X[1], y_mB, y_mT) & in_range(X[0], 0.0, x_in))
                | (in_range(X[1], y_tIn, y_tOut) & in_range(X[0], 0.0, x_out))
            )
        )

class LiquidVolume(F.VolumeSubdomain):
    def locate_subdomain_entities(self, m):
        return dolfinx.mesh.locate_entities(
            m, m.topology.dim,
            lambda X: in_range(X[0], 0.0, x_in) & in_range(X[1], y_mT, y_fT)
        )

# ---------------------------
# 4) External boundaries on SUBMESH
# ---------------------------
class OuterBottom(F.SurfaceSubdomain):  # y = 0
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[1], 0.0, atol=1e-12))

class OuterTop(F.SurfaceSubdomain):     # y = y_tOut
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[1], y_tOut, atol=1e-12))

class OuterRight(F.SurfaceSubdomain):   # x = x_out
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], x_out, atol=1e-12))

# Left vertical pieces (x = 0)
class LeftBottomPlate(F.SurfaceSubdomain):      # y in [0, y_bT]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], 0.0, atol=1e-12) & in_range(X[1], 0.0, y_bT))

class LeftMiddlePlate(F.SurfaceSubdomain):      # y in [y_mB, y_mT]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], 0.0, atol=1e-12) & in_range(X[1], y_mB, y_mT))

class LeftLiquidBand(F.SurfaceSubdomain):       # y in [y_mT, y_fT]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], 0.0, atol=1e-12) & in_range(X[1], y_mT, y_fT))

class LeftTopPlate(F.SurfaceSubdomain):         # y in [y_tIn, y_tOut]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], 0.0, atol=1e-12) & in_range(X[1], y_tIn, y_tOut))

# External BCs on the submesh
class Bottom_Ni_top(F.SurfaceSubdomain):        # y = y_bT, x in [0,x_in]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[1], y_bT, atol=1e-12) & in_range(X[0], 0.0, x_in))

class Mem_Ni_bottom(F.SurfaceSubdomain):        # y = y_mB, x in [0,x_in]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[1], y_mB, atol=1e-12) & in_range(X[0], 0.0, x_in))

class Top_Ni_bottom(F.SurfaceSubdomain):        # y = y_tIn, x in [0,x_in]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[1], y_tIn, atol=1e-12) & in_range(X[0], 0.0, x_in))

class Up_Ni_Left(F.SurfaceSubdomain):           # x = x_in, y in [y_bT,y_mB]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], x_in, atol=1e-12) & in_range(X[1], y_bT, y_mB))

class Ds_Ni_Left(F.SurfaceSubdomain):           # x = x_in, y in [y_fT,y_tIn]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[0], x_in, atol=1e-12) & in_range(X[1], y_fT, y_tIn))

class LiquidTopBand(F.SurfaceSubdomain):        # y = y_fT, x in [0,x_in]
    def locate_boundary_facet_indices(self, m):
        return dolfinx.mesh.locate_entities_boundary(m, m.topology.dim-1,
            lambda X: np.isclose(X[1], y_fT, atol=1e-12) & in_range(X[0], 0.0, x_in))

# Instantiate boundary pieces (IDs)
Ni_ext_bot   = OuterBottom(id=11)
Ni_ext_top   = OuterTop(id=12)
Ni_ext_right = OuterRight(id=13)
Ni_left_bot  = LeftBottomPlate(id=14)
Ni_left_mid  = LeftMiddlePlate(id=16)
Liquid_left  = LeftLiquidBand(id=17)
Ni_left_top  = LeftTopPlate(id=18)

bott_Ni_top  = Bottom_Ni_top(id=201)
mem_Ni_bot   = Mem_Ni_bottom(id=202)
top_Ni_bot   = Top_Ni_bottom(id=203)
up_Ni_left   = Up_Ni_Left(id=204)
ds_Ni_left   = Ds_Ni_Left(id=205)
liq_top      = LiquidTopBand(id=206)

# ---------------------------
# 5) FESTIM model on SUBMESH
# ---------------------------
my_model = F.HydrogenTransportProblemDiscontinuous()
my_model.mesh = F.Mesh(submesh)

# Materials
D_solid, D_liquid = 2.0, 5.0
K_solid, K_liquid = 3.0, 6.0
mat_solid  = F.Material(D_0=D_solid,  E_D=0.0, K_S_0=K_solid,  E_K_S=0.0)
mat_liquid = F.Material(D_0=D_liquid, E_D=0.0, K_S_0=K_liquid, E_K_S=0.0)

solid_vol  = SolidVolume(id=1, material=mat_solid)
liquid_vol = LiquidVolume(id=2, material=mat_liquid)

my_model.subdomains = [
    solid_vol, liquid_vol,
    Ni_ext_bot, Ni_ext_top, Ni_ext_right,
    Ni_left_bot, Ni_left_mid, Liquid_left, Ni_left_top,
    bott_Ni_top, mem_Ni_bot, top_Ni_bot, up_Ni_left, ds_Ni_left, liq_top,
]

my_model.surface_to_volume = {
    Ni_ext_bot:   solid_vol,
    Ni_ext_top:   solid_vol,
    Ni_ext_right: solid_vol,
    Ni_left_bot:  solid_vol,
    Ni_left_mid:  solid_vol,
    Ni_left_top:  solid_vol,
    bott_Ni_top:  solid_vol,
    mem_Ni_bot:   solid_vol,
    top_Ni_bot:   solid_vol,
    up_Ni_left:   solid_vol,
    ds_Ni_left:   solid_vol,
    Liquid_left:  liquid_vol,
    liq_top:      liquid_vol,
}

# Only ONE internal interface: liquid <-> solid
my_model.interfaces = [F.Interface(id=99, subdomains=[liquid_vol, solid_vol])]
#my_model.interfaces = []

# Species in solid + liquid only
H = F.Species("H", subdomains=[solid_vol, liquid_vol])
my_model.species = [H]

# ---------------------------
# MMS exact solutions & sources
# ---------------------------


K_s_Ni   = K_solid   # K_{s,Ni}
K_H_FLiBe = K_liquid # K_{H,FLiBe}

#def exact_solution_Ni(mod):
#    return lambda x: 1.5 + 0.6*mod.sin(2*mod.pi*(x[0] + 0.25)) + 0.6*mod.cos(2*mod.pi*x[1])


#def exact_solution_liquid(mod):
#    return lambda x: K_H_FLiBe * (exact_solution_Ni(mod)(x) / K_s_Ni)**2

#def exact_solution_liquid(mod):
#    return exact_solution_Ni(mod) 


# ---- parameters you already have ----
# D_solid, D_liquid, K_s_Ni, K_H_FLiBe, x_in, x_out  等
# ------------------------------------

# interface value for v to ensure flux continuity
v0 = (D_solid * K_s_Ni) / (2.0 * D_liquid * K_H_FLiBe)

# choose a small relative amplitude so fields stay positive
amp = 0.25                       
Lx  = x_out                      # length scale to make amplitude dimensionless
A   = (amp * v0) / Lx            # has units 1/m so that A*(x-x_in) is dimensionless* v0

def v_exact(mod):
    # v(x,y) equals v0 on the interface (x = x_in), but has nonzero normal derivative
    return lambda x: v0 + A * (x[0] - x_in) * mod.sin(2*mod.pi * x[1])

def exact_solution_Ni(mod):
    v = v_exact(mod)
    return lambda x: K_s_Ni * v(x)                 # c_S = K_s * v

def exact_solution_liquid(mod):
    v = v_exact(mod)
    return lambda x: K_H_FLiBe * (v(x)**2)         # c_L = K_H * v^2

# UFL versions for FESTIM
exact_solution_Ni_ufl     = exact_solution_Ni(ufl)
exact_solution_liquid_ufl = exact_solution_liquid(ufl)

# MMS volume sources (steady): S = - div(D * grad(c_exact))
f_Ni     = lambda x: -ufl.div(D_solid  * ufl.grad(exact_solution_Ni_ufl(x)))
f_liquid = lambda x: -ufl.div(D_liquid * ufl.grad(exact_solution_liquid_ufl(x)))

# Sources
my_model.sources = [
    F.ParticleSource(f_liquid, volume=liquid_vol, species=H),
    F.ParticleSource(f_Ni, volume=solid_vol,  species=H),
]

# External Dirichlet BCs = exact values
solid_BCs = [Ni_ext_bot, Ni_ext_top, Ni_ext_right, Ni_left_bot, Ni_left_mid,
             Ni_left_top, bott_Ni_top, mem_Ni_bot, top_Ni_bot, up_Ni_left, ds_Ni_left]
liquid_BCs = [Liquid_left, liq_top]

my_model.boundary_conditions = (
    [F.FixedConcentrationBC(subdomain=surface, value=exact_solution_Ni_ufl, species=H) for surface in solid_BCs] +
    [F.FixedConcentrationBC(subdomain=surface, value=exact_solution_liquid_ufl, species=H) for surface in liquid_BCs]
)

# ---------------------------
# Solve (steady) — tight tolerances for MMS
# ---------------------------
my_model.temperature = 773.0
my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)
my_model.initialise()
my_model.run()
print("Solve done (MMS with interface relation applied).")

from dolfinx import fem
import ufl, numpy as np

u_solid  = H.subdomain_to_post_processing_solution[solid_vol]
u_liquid = H.subdomain_to_post_processing_solution[liquid_vol]

exact_solid_fn  = fem.Function(u_solid.function_space);  exact_solid_fn.interpolate(exact_solution_Ni(np))
exact_liquid_fn = fem.Function(u_liquid.function_space); exact_liquid_fn.interpolate(exact_solution_liquid(np))

dx_S = ufl.Measure("dx", domain=u_solid.function_space.mesh)
dx_L = ufl.Measure("dx", domain=u_liquid.function_space.mesh)

eS = fem.assemble_scalar(fem.form(ufl.inner(u_solid  - exact_solid_fn,  u_solid  - exact_solid_fn)  * dx_S))
eL = fem.assemble_scalar(fem.form(ufl.inner(u_liquid - exact_liquid_fn, u_liquid - exact_liquid_fn) * dx_L))

print(f"L2 error (solid)  = {np.sqrt(eS):.3e}")
print(f"L2 error (liquid) = {np.sqrt(eL):.3e}")
print("solid [min,max]  =", float(u_solid.x.array.min()),  float(u_solid.x.array.max()))
print("liquid [min,max] =", float(u_liquid.x.array.min()), float(u_liquid.x.array.max()))


# --------- Visualization ---------
import numpy as np
import os
import pyvista
import dolfinx
from dolfinx import fem
from dolfinx.plot import vtk_mesh

pyvista.OFF_SCREEN = False   # show an on-screen window

def get_ugrid(computed_solution: dolfinx.fem.Function, label: str):
    """Convert a dolfinx Function into a PyVista grid.
    If the function is nodal (CG), store as point_data; if cell-wise (DG), store as cell_data.
    If neither matches, interpolate to CG1 for robust plotting."""
    topo, cell_types, geom = vtk_mesh(computed_solution.function_space)
    grid = pyvista.UnstructuredGrid(topo, cell_types, geom)

    arr = computed_solution.x.array.real
    if arr.size == grid.n_points:
        grid.point_data[label] = arr
    elif arr.size == grid.n_cells:
        grid.cell_data[label] = arr
    else:
        V1 = fem.functionspace(computed_solution.function_space.mesh, ("CG", 1))
        tmp = fem.Function(V1)
        tmp.interpolate(computed_solution)
        topo, cell_types, geom = vtk_mesh(V1)
        grid = pyvista.UnstructuredGrid(topo, cell_types, geom)
        grid.point_data[label] = tmp.x.array.real

    grid.set_active_scalars(label)
    return grid

# --- Numerical fields ---
u_grid_Ni    = get_ugrid(H.subdomain_to_post_processing_solution[solid_vol],  "c")
u_grid_liquid = get_ugrid(H.subdomain_to_post_processing_solution[liquid_vol], "c")

# --- Exact fields interpolated onto the same spaces ---
exact_Ni = dolfinx.fem.Function(
    H.subdomain_to_post_processing_solution[solid_vol].function_space
)
exact_Ni.interpolate(exact_solution_Ni(np))

exact_liquid = dolfinx.fem.Function(
    H.subdomain_to_post_processing_solution[liquid_vol].function_space
)
exact_liquid.interpolate(exact_solution_liquid(np))

u_grid_exact_Ni     = get_ugrid(exact_Ni,     "c_exact")
u_grid_exact_liquid = get_ugrid(exact_liquid, "c_exact")

# --- Shared color limits across both panels (so colors mean the same value) ---
def _vals(grid, key):
    if key in grid.point_data: return grid.point_data[key]
    if key in grid.cell_data:  return grid.cell_data[key]
    return np.array([])

_all = np.concatenate([
    _vals(u_grid_Ni, "c"),
    _vals(u_grid_liquid, "c"),
    _vals(u_grid_exact_Ni, "c_exact"),
    _vals(u_grid_exact_liquid, "c_exact"),
])
vmin, vmax = float(np.nanmin(_all)), float(np.nanmax(_all))

u_num_s = H.subdomain_to_post_processing_solution[solid_vol]
u_num_l = H.subdomain_to_post_processing_solution[liquid_vol]
import numpy as np
print("solid NaNs:", np.isnan(u_num_s.x.array).sum())
print("liquid NaNs:", np.isnan(u_num_l.x.array).sum())


# --- Plot: left = numerical, right = exact ---
u_plotter = pyvista.Plotter(shape=(1, 2))

u_plotter.subplot(0, 0)
u_plotter.add_mesh(u_grid_Ni,    show_edges=True,  clim=[vmin, vmax])
u_plotter.add_mesh(u_grid_liquid, show_edges=True,  clim=[vmin, vmax])
u_plotter.view_xy()
u_plotter.add_text("Numerical", font_size=12)

u_plotter.subplot(0, 1)
u_plotter.add_mesh(u_grid_exact_Ni,     show_edges=False, clim=[vmin, vmax])
u_plotter.add_mesh(u_grid_exact_liquid, show_edges=False, clim=[vmin, vmax])
u_plotter.view_xy()
u_plotter.add_text("Exact", font_size=12)

# Optional: print bounds for quick geometry sanity-check
print("Ni bounds:    ", u_grid_Ni.bounds)
print("Liquid bounds:", u_grid_liquid.bounds)

# Show window and also save a screenshot
out_png = "discontinuity_concentration.png"
u_plotter.show(
    title="FESTIM MMS",
    window_size=[1200, 600],
    screenshot=out_png,
    auto_close=True
)
print("Saved to:", os.path.abspath(out_png))
