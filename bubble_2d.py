import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F

import ufl
from dolfinx import fem
from dolfinx import geometry
from dolfinx.fem import assemble_scalar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import gc
from petsc4py import PETSc


OUTDIR = "plots"

COMM = MPI.COMM_WORLD
RANK = COMM.rank

MESH_FILE = "2Dbubble.msh"
_read = gmshio.read_from_msh(MESH_FILE, COMM, rank=0, gdim=2)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags


def facet_measure(tag_id: int) -> float:
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    one = fem.Constant(mesh, 1.0)
    val_local = assemble_scalar(fem.form(one * ds(tag_id)))
    return COMM.allreduce(val_local, op=MPI.SUM)


L35 = facet_measure(35)  # liquid-gas surface (hole) length in 2D
L36 = facet_measure(36)  # solid-gas surface (hole) length in 2D

# if RANK == 0:
#     print(f"L35={L35:.6e} m, L36={L36:.6e} m ")


def T_label_from_temperature(T):
    return f"T{int(round(T))}K"


def csv_path_for_T(outdir, T):
    return os.path.join(outdir, f"time_series_{T_label_from_temperature(T)}.csv")


def ensure_csv_header(csv_path):
    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("t,p_b,H_liq_near_ls,H_liq_near_lg,H_sol_near_ls,H_sol_near_sg\n")


def append_row(csv_path, row):
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(f"{v:.16e}" for v in row) + "\n")


def read_last_row(csv_path):
    if (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0:
        return None
    with open(csv_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        if end == 0:
            return None
        size = min(8192, end)
        f.seek(end - size)
        tail = f.read().decode("utf-8", errors="ignore").strip().splitlines()
    for line in reversed(tail):
        if line and (not line.startswith("t,")):
            parts = line.split(",")
            if len(parts) >= 6:
                return np.array([float(x) for x in parts[:6]], dtype=float)
    return None


def eval_point_on_function(u, x: float, y: float) -> float:
    pts = np.array([[x, y, 0.0]], dtype=np.float64)  # (1,3)

    tree = geometry.bb_tree(u.function_space.mesh, u.function_space.mesh.topology.dim)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(u.function_space.mesh, cand, pts)

    links = coll.links(0)
    if len(links) == 0:
        return float("nan")

    cell = np.array([links[0]], dtype=np.int32)
    v = u.eval(pts, cell)
    return float(v.reshape(-1)[0])


def sample_4_points(H, liquid_volume, solid_volume, pts_dict):
    u_liq = H.subdomain_to_post_processing_solution[liquid_volume]
    u_sol = H.subdomain_to_post_processing_solution[solid_volume]

    p1 = pts_dict["liq_near_ls"]
    p2 = pts_dict["liq_near_lg"]
    p3 = pts_dict["sol_near_ls"]
    p4 = pts_dict["sol_near_sg"]

    c1 = eval_point_on_function(u_liq, p1[0], p1[1])
    c2 = eval_point_on_function(u_liq, p2[0], p2[1])
    c3 = eval_point_on_function(u_sol, p3[0], p3[1])
    c4 = eval_point_on_function(u_sol, p4[0], p4[1])

    return c1, c2, c3, c4


temperature = 773.15  # K

D_0_solid, E_D_solid = 3e-3, 0.2
K_S_0_solid, E_K_S_solid = 4e-5, 0.2

D_0_liquid, E_D_liquid = 1e-5, 0.2
K_S_0_liquid, E_K_S_liquid = 1e-5, 0.2


# D_0_solid, E_D_solid = 3e-3, 1e-3
# K_S_0_solid, E_K_S_solid = 4e-5, 1e-3

# D_0_liquid, E_D_liquid = 1e-5, 1e-3
# K_S_0_liquid, E_K_S_liquid = 1e-5, 1e-3


P_top = 1e5
P_bottom = 1e-3

PENALTY = 1e5

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


def solve_for_pb(p_b: float):
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

    flux_lg = F.SurfaceFlux(field=H, surface=liquid_gas_surface, filename=None)
    flux_sg = F.SurfaceFlux(field=H, surface=solid_gas_surface, filename=None)
    my_model.exports = [flux_lg, flux_sg]

    my_model.initialise()
    my_model.run()

    j_liquid_gas = float(flux_lg.value)
    j_solid_gas = float(flux_sg.value)

    del my_model, flux_lg, flux_sg

    return j_liquid_gas, j_solid_gas, H


POINTS = {
    "liq_near_ls": (0.01, 0.00204),  # liquid near liquid-solid interface
    "liq_near_lg": (0.039, 0.00304),  # liquid near liquid-gas interface (hole)
    "sol_near_ls": (0.01, 0.00203),  # solid near liquid-solid interface
    "sol_near_sg": (0.039, 0.00203),  # solid near solid-gas interface (hole)
}


R_gas = 8.314  # J/(mol*K)
t_b = 0.001
V_b = t_b * L36

t_total = 600
dt = 5

if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

csv_path = csv_path_for_T(OUTDIR, temperature)

t_start = 0.0
p_b = 3.157e02

if RANK == 0:
    ensure_csv_header(csv_path)
    last = read_last_row(csv_path)
    if last is not None:
        t_start = float(last[0] + dt)
        p_b = float(last[1])
        print(f"\nResuming from CSV: {csv_path}")
        print(f"Last saved t={last[0]:.3f} s, p_b={last[1]:.3e} Pa")
        print(f"Restarting at t={t_start:.3f} s, p_b={p_b:.3e} Pa\n")
    else:
        print(f"\nStarting new run, appending to CSV: {csv_path}\n")

t_start = COMM.bcast(t_start if RANK == 0 else None, root=0)
p_b = COMM.bcast(p_b if RANK == 0 else None, root=0)

N_b = p_b * V_b / (R_gas * temperature)

t_hist, pb_hist = [], []
H1_hist, H2_hist, H3_hist, H4_hist = [], [], [], []

H_last = None

CLEAN_EVERY = 5  # PETSc cleanup period

if RANK == 0:
    print("\nRunning dynamic p_b update\n")

for n in range(t_total):
    t_now = t_start + n * dt

    j_liquid_gas, j_solid_gas, H = solve_for_pb(p_b)
    H_last = H

    Fnet = j_liquid_gas * L35 + j_solid_gas * L36

    N_b = max(N_b + dt * Fnet, 0.0)
    p_b = N_b * R_gas * temperature / V_b

    if RANK == 0:
        c1, c2, c3, c4 = sample_4_points(H, liquid_volume, solid_volume, POINTS)

        t_hist.append(t_now)
        pb_hist.append(p_b)
        H1_hist.append(c1)
        H2_hist.append(c2)
        H3_hist.append(c3)
        H4_hist.append(c4)

        append_row(csv_path, [t_now, p_b, c1, c2, c3, c4])

        print(
            f"t={t_now:.3f} s  p_b={p_b:.3e} Pa  "
            f"j_lg={j_liquid_gas:.3e}  j_sg={j_solid_gas:.3e}  "
            f"F={Fnet:.3e} mol/s  "
            f"H_pts=[{c1:.3e}, {c2:.3e}, {c3:.3e}, {c4:.3e}]"
        )

    del H
    H_last = None

    # Periodic cleanup to reduce PETSc/MPI resource accumulation
    if (n % CLEAN_EVERY) == 0:
        gc.collect()
        PETSc.garbage_cleanup(COMM)

    if abs(Fnet) < 1e-11:
        if RANK == 0:
            print(f"\nBreak at t={t_now:.3f} s (|Fnet| small)\n")
        break


if RANK == 0 and len(t_hist) > 0:
    os.makedirs(OUTDIR, exist_ok=True)

    T_label = T_label_from_temperature(temperature)

    t_arr = np.array(t_hist, dtype=float)
    pb_arr = np.array(pb_hist, dtype=float)
    h1 = np.array(H1_hist, dtype=float)
    h2 = np.array(H2_hist, dtype=float)
    h3 = np.array(H3_hist, dtype=float)
    h4 = np.array(H4_hist, dtype=float)

    def save_plot(x, y, fname, ylabel):
        os.makedirs(OUTDIR, exist_ok=True)
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("t (s)")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, fname), dpi=200)
        plt.close()

    save_plot(t_arr, h1, f"H_liq_near_ls_vs_time_{T_label}.png", "H")
    save_plot(t_arr, h2, f"H_liq_near_lg_vs_time_{T_label}.png", "H")
    save_plot(t_arr, h3, f"H_sol_near_ls_vs_time_{T_label}.png", "H")
    save_plot(t_arr, h4, f"H_sol_near_sg_vs_time_{T_label}.png", "H")
    save_plot(t_arr, pb_arr, f"pb_vs_time_{T_label}.png", "p_b (Pa)")

    print("\nSaved results to:", OUTDIR)
    print(f"time_series_{T_label}.csv")
    print(f"H_liq_near_ls_vs_time_{T_label}.png")
    print(f"H_liq_near_lg_vs_time_{T_label}.png")
    print(f"H_sol_near_ls_vs_time_{T_label}.png")
    print(f"H_sol_near_sg_vs_time_{T_label}.png")
    print(f"pb_vs_time_{T_label}.png")
    print("\nDone.\n")
