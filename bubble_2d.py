import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F

import ufl
from dolfinx import fem
from dolfinx import geometry
from dolfinx.fem import assemble_scalar

import dolfinx
import pyvista
from dolfinx.plot import vtk_mesh


OUTDIR = "plots"

COMM = MPI.COMM_WORLD
RANK = COMM.rank

MESH_FILE = "2Dbubble.msh"
_read = gmshio.read_from_msh(MESH_FILE, COMM, rank=0, gdim=2)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags

# if RANK == 0:
#     print("Done reading mesh.")
#     print("cell_tags:", np.unique(cell_tags.values))
#     print("facet_tags:", np.unique(facet_tags.values))


def facet_measure(tag_id: int) -> float:
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    one = fem.Constant(mesh, 1.0)
    val_local = assemble_scalar(fem.form(one * ds(tag_id)))
    return COMM.allreduce(val_local, op=MPI.SUM)


L35 = facet_measure(35)  # liquid-gas surface (hole) length in 2D
L36 = facet_measure(36)  # solid-gas surface (hole) length in 2D

if RANK == 0:
    print(f"L35={L35:.6e} m, L36={L36:.6e} m  (2D -> these are lengths)")

try:
    pyvista.start_xvfb()
except Exception:
    pass


def get_ugrid(computed_solution, label: str):
    topo, cell_types, geom = vtk_mesh(computed_solution.function_space)
    grid = pyvista.UnstructuredGrid(topo, cell_types, geom)
    grid.point_data[label] = computed_solution.x.array.real
    grid.set_active_scalars(label)
    return grid


def save_H_plot_clean_single(H, liquid_volume, solid_volume, p_b: float, outdir="."):
    u_liq = H.subdomain_to_post_processing_solution[liquid_volume]
    u_sol = H.subdomain_to_post_processing_solution[solid_volume]

    if RANK == 0:
        print("liq min/max:", u_liq.x.array.min(), u_liq.x.array.max())
        print("sol min/max:", u_sol.x.array.min(), u_sol.x.array.max())

    grid_liq = get_ugrid(u_liq, "H")
    grid_sol = get_ugrid(u_sol, "H")

    # shared color limits
    vmin = float(min(grid_liq["H"].min(), grid_sol["H"].min()))
    vmax = float(max(grid_liq["H"].max(), grid_sol["H"].max()))

    p = pyvista.Plotter(off_screen=True, window_size=(900, 900))
    p.set_background("white")

    scalar_bar_args = dict(
        title="H",
        vertical=False,
        position_x=0.22,
        position_y=0.05,
        width=0.56,
        height=0.06,
        label_font_size=16,
        title_font_size=18,
        n_labels=4,
        fmt="%.2e",
    )

    p.add_mesh(
        grid_liq,
        scalars="H",
        clim=(vmin, vmax),
        show_edges=False,
        scalar_bar_args=scalar_bar_args,
    )
    p.add_mesh(grid_sol, scalars="H", clim=(vmin, vmax), show_edges=False)

    p.view_xy()
    p.camera.parallel_projection = True
    p.reset_camera()

    p.hide_axes()

    fname = f"{outdir}/H_pb_{p_b:.3e}.png"
    p.screenshot(fname)
    p.close()
    return fname


def sample_profile_across_interface(
    H, liquid_volume, solid_volume, x0, y_min, y_max, n=400
):
    u_liq = H.subdomain_to_post_processing_solution[liquid_volume]
    u_sol = H.subdomain_to_post_processing_solution[solid_volume]

    ys = np.linspace(y_min, y_max, n)
    pts = np.zeros((3, n), dtype=np.float64)
    pts[0, :] = x0
    pts[1, :] = ys

    def eval_on_function(u, pts):
        ptsT = np.ascontiguousarray(pts.T, dtype=np.float64)  # (N, 3)

        tree = geometry.bb_tree(
            u.function_space.mesh, u.function_space.mesh.topology.dim
        )
        cand = geometry.compute_collisions_points(tree, ptsT)
        coll = geometry.compute_colliding_cells(u.function_space.mesh, cand, ptsT)

        cells = np.array(
            [
                coll.links(i)[0] if len(coll.links(i)) > 0 else -1
                for i in range(ptsT.shape[0])
            ],
            dtype=np.int32,
        )

        vals = np.full(ptsT.shape[0], np.nan, dtype=np.float64)
        ok = cells >= 0
        if np.any(ok):
            v = u.eval(ptsT[ok], cells[ok])
            vals[ok] = v.reshape(-1)
        return vals

    c_liq = eval_on_function(u_liq, pts)
    c_sol = eval_on_function(u_sol, pts)
    return ys, c_liq, c_sol


temperature = 973.15  # K


D_0_solid, E_D_solid = 3e-3, 1e-30
D_0_liquid, E_D_liquid = 1e-5, 1e-30


K_S_0_liquid, E_K_S_liquid = 1e-5, 1e-30
K_S_0_solid, E_K_S_solid = 4e-5, 1e-30

P_top = 1e5
P_bottom = 1e-3


PENALTY = 1e7


ATOL = 1e-10
RTOL = 1e-8

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

solid_volume = F.VolumeSubdomain(id=1, material=mat_solid)
liquid_volume = F.VolumeSubdomain(id=2, material=mat_liquid)

top = F.SurfaceSubdomain(id=33)
bottom = F.SurfaceSubdomain(id=34)
liquid_gas_surface = F.SurfaceSubdomain(id=35)
solid_gas_surface = F.SurfaceSubdomain(id=36)
liquid_solid_interface = F.SurfaceSubdomain(id=101)


def solve_and_plot_for_pb(p_b: float, outdir="plots"):
    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cartesian")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags

    my_model.subdomains = [
        solid_volume,
        liquid_volume,
        top,
        bottom,
        liquid_gas_surface,
        solid_gas_surface,
        liquid_solid_interface,
    ]

    my_model.method_interface = "penalty"
    my_model.interfaces = [
        F.Interface(
            id=101, subdomains=[solid_volume, liquid_volume], penalty_term=PENALTY
        )
    ]

    my_model.surface_to_volume = {
        top: liquid_volume,
        bottom: solid_volume,
        liquid_gas_surface: liquid_volume,
        solid_gas_surface: solid_volume,
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = temperature

    bc_top = [
        F.HenrysBC(
            subdomain=top,
            species=H,
            pressure=P_top,
            H_0=K_S_0_liquid,
            E_H=E_K_S_liquid,
        )
    ]

    bc_bottom = [
        F.SievertsBC(
            subdomain=bottom,
            species=H,
            pressure=P_bottom,
            S_0=K_S_0_solid,
            E_S=E_K_S_solid,
        )
    ]

    bc_liquid_gas = [
        F.HenrysBC(
            subdomain=liquid_gas_surface,
            species=H,
            pressure=float(p_b),
            H_0=K_S_0_liquid,
            E_H=E_K_S_liquid,
        )
    ]

    bc_solid_gas = [
        F.SievertsBC(
            subdomain=solid_gas_surface,
            species=H,
            pressure=float(p_b),
            S_0=K_S_0_solid,
            E_S=E_K_S_solid,
        )
    ]

    my_model.boundary_conditions = bc_top + bc_bottom + bc_liquid_gas + bc_solid_gas
    my_model.settings = F.Settings(atol=ATOL, rtol=RTOL, transient=False)

    # Flux density exporters
    flux_lg = F.SurfaceFlux(field=H, surface=liquid_gas_surface, filename=None)
    flux_sg = F.SurfaceFlux(field=H, surface=solid_gas_surface, filename=None)
    my_model.exports = [flux_lg, flux_sg]

    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    my_model.initialise()
    my_model.run()

    if RANK == 0:
        import os

        os.makedirs(outdir, exist_ok=True)
    COMM.barrier()
    png = save_H_plot_clean_single(H, liquid_volume, solid_volume, p_b, outdir=outdir)

    # Return flux densities
    j_liquid_gas = float(flux_lg.value)
    j_solid_gas = float(flux_sg.value)
    return j_liquid_gas, j_solid_gas, png, H


R_gas = 8.314  # J/(mol*K)
t_b = 0.001
V_b = t_b * L36  # m^3, arbitrary bubble volume for ideal
dt = 0.1  # time step for dynamic update (s)

# pb_list = [1e-12, 1e-9, 1e-6, 1e-3, 1.0]
# if RANK == 0:
#     print("\nRunning p_b sweep and saving plots to ./plots ...\n")

# for pb in pb_list:
#     j_liquid_gas, j_solid_gas, png, H = solve_and_plot_for_pb(pb, outdir="plots")
#     N_b = pb * V_b / (R_gas * temperature)
#     # Convert flux density -> molar flow (mol/s) using lengths (2D) * thickness
#     F_mol_s = j_liquid_gas * L35 + j_solid_gas * L36

#     N_b = max(N_b + dt * F_mol_s, 0.0)
#     p_b = N_b * R_gas * temperature / V_b
#     ys, cL, cS = sample_profile_across_interface(
#         H,
#         liquid_volume,
#         solid_volume,
#         x0=0.039,  # choose x location
#         y_min=-0.01,  # bottom of solid
#         y_max=0.01,  # top of liquid
#         n=600,
#     )

#     if RANK == 0:
#         import matplotlib.pyplot as plt

#         plt.figure()
#         plt.plot(ys, cL, label="liquid")
#         plt.plot(ys, cS, label="solid")
#         plt.axvline(0.0, linestyle="--")  # if your interface is at y=0
#         plt.xlabel("y (m)")
#         plt.ylabel("H concentration")
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"{OUTDIR}/profile_across_interface_pb_{pb:.3e}.png", dpi=200)
#         plt.close()

#     if RANK == 0:
#         print(
#             f"pb={pb:.3e} Pa  j_liquid_gas={j_liquid_gas:.3e}  j_solid_gas={j_solid_gas:.3e}  p_b~{p_b:.3e} Pa, N={N_b:.3e} mol  saved={png}"
#         )

# if RANK == 0:
#     print("\nDone. Check the PNGs in ./plots/")
t_total = 200
dt = 0.1

p_b = 1e-12
N_b = p_b * V_b / (R_gas * temperature)

t_hist = []
pb_hist = []

H_last = None
p_b_last = None

if RANK == 0:
    print("\nRunning dynamic p_b update\n")

for n in range(t_total):
    t_now = n * dt

    j_liquid_gas, j_solid_gas, png, H = solve_and_plot_for_pb(p_b, outdir="plots")

    H_last = H
    p_b_last = p_b

    Fnet = j_liquid_gas * L35 + j_solid_gas * L36

    N_b = max(N_b + dt * Fnet, 0.0)
    p_b = N_b * R_gas * temperature / V_b

    if RANK == 0:
        t_hist.append(t_now)
        pb_hist.append(p_b)

        print(
            f"t={t_now:.3f} s  p_b={p_b:.3e} Pa  "
            f"j_liquid_gas={j_liquid_gas:.3e}  "
            f"j_solid_gas={j_solid_gas:.3e}  "
            f"F~{Fnet:.3e} mol/s"
        )

    if abs(Fnet) < 1e-12:
        if RANK == 0:
            print(f"\nBreak at t={t_now:.3f} s (|Fnet| small)\n")
        break


# ---------------- FINAL PLOTS ONLY ----------------

if RANK == 0 and H_last is not None:
    import matplotlib.pyplot as plt

    print("\nGenerating final plots...\n")

    # ---- Profile at x0 = 0.039 ----
    ys_1, cL_1, cS_1 = sample_profile_across_interface(
        H_last,
        liquid_volume,
        solid_volume,
        x0=0.039,
        y_min=-0.01,
        y_max=0.01,
        n=600,
    )

    plt.figure()
    plt.plot(ys_1, cL_1, label="liquid")
    plt.plot(ys_1, cS_1, label="solid")
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("y (m)")
    plt.ylabel("H concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FINAL_profile_x0_0.039.png", dpi=200)
    plt.close()

    # ---- Profile at x0 = 0.01 ----
    ys_2, cL_2, cS_2 = sample_profile_across_interface(
        H_last,
        liquid_volume,
        solid_volume,
        x0=0.01,
        y_min=-0.01,
        y_max=0.01,
        n=600,
    )

    plt.figure()
    plt.plot(ys_2, cL_2, label="liquid")
    plt.plot(ys_2, cS_2, label="solid")
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("y (m)")
    plt.ylabel("H concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FINAL_profile_x0_0.01.png", dpi=200)
    plt.close()

    # ---- t vs p_b ----
    plt.figure()
    plt.plot(t_hist, pb_hist)
    plt.xlabel("t (s)")
    plt.ylabel("p_b (Pa)")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/FINAL_pb_vs_time.png", dpi=200)
    plt.close()

    print("Saved:")
    print(f"{OUTDIR}/FINAL_profile_x0_0.039.png")
    print(f"{OUTDIR}/FINAL_profile_x0_0.01.png")
    print(f"{OUTDIR}/FINAL_pb_vs_time.png")

    print("\nDone.\n")
