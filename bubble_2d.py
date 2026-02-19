import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F

import ufl
from dolfinx import fem, geometry
from dolfinx.fem import assemble_scalar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

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


L35 = facet_measure(35)
L36 = facet_measure(36)


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
    pts = np.array([[x, y, 0.0]], dtype=np.float64)
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


POINTS = {
    "liq_near_ls": (0.01, 0.00204),
    "liq_near_lg": (0.039, 0.00304),
    "sol_near_ls": (0.01, 0.00203),
    "sol_near_sg": (0.039, 0.00203),
}

temperature = 773.15

D_0_solid, E_D_solid = 3e-3, 0.2
K_S_0_solid, E_K_S_solid = 4e-5, 0.2

D_0_liquid, E_D_liquid = 1e-5, 0.2
K_S_0_liquid, E_K_S_liquid = 1e-5, 0.2

P_top = 1e5
P_bottom = 1e-3
PENALTY = 1e4

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

R_gas = 8.314
t_b = 0.001
V_b = t_b * L36

t_total = 2e4
dt = 1

if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

csv_path = csv_path_for_T(OUTDIR, temperature)

t_start = 0.0
p_b0 = 1e-12  # initial pressure

if RANK == 0:
    ensure_csv_header(csv_path)
    last = read_last_row(csv_path)
    if last is not None:
        t_start = float(last[0] + dt)
        p_b0 = float(last[1])
        print(f"\nResuming from CSV: {csv_path}")
        print(f"Last saved t={last[0]:.3f} s, p_b={last[1]:.3e} Pa")
        print(f"Restarting at t={t_start:.3f} s, p_b={p_b0:.3e} Pa\n")
    else:
        print(f"\nStarting new run, appending to CSV: {csv_path}\n")


kB_eV = 8.617333262145e-5


def K_henry_T():
    return K_S_0_liquid * np.exp(-E_K_S_liquid / (kB_eV * temperature))


def K_sievert_T():
    return K_S_0_solid * np.exp(-E_K_S_solid / (kB_eV * temperature))


K_H_T = float(K_henry_T())
K_S_T = float(K_sievert_T())

USE_LENGTHS = True


class Custom2DProblem(F.HydrogenTransportProblemDiscontinuous):
    def iterate(self):
        super().iterate()

        j_liquid_gas = float(self.flux_lg.value)
        j_solid_gas = float(self.flux_sg.value)

        if USE_LENGTHS:
            Fnet = j_liquid_gas * L35 + j_solid_gas * L36
        else:
            Fnet = j_liquid_gas + j_solid_gas

        self.N_b = max(self.N_b + float(self.dt.value) * Fnet, 0.0)
        new_pb = max(self.N_b * R_gas * temperature / V_b, 0.0)

        bc_liquid_gas[0].value = new_pb * K_H_T
        bc_solid_gas[0].value = (new_pb**0.5) * K_S_T if new_pb > 0.0 else 0.0

        self.bc_forms[self.idx_bclg] = self.create_dirichletbc_form(bc_liquid_gas[0])
        self.bc_forms[self.idx_bcsg] = self.create_dirichletbc_form(bc_solid_gas[0])

        if RANK == 0:
            c1, c2, c3, c4 = sample_4_points(
                self.H, self.liquid_volume, self.solid_volume, self.points
            )
            append_row(
                self.csv_path,
                [float(self.t.value), float(new_pb), c1, c2, c3, c4],
            )

            # print(
            #     f"t={float(self.t.value):.3f} s  p_b={float(pb_box[0]):.3e} Pa  "
            #     f"j_lg={j_liquid_gas:.3e}  j_sg={j_solid_gas:.3e}  "
            #     f"F={Fnet:.3e}  "
            #     f"H_pts=[{c1:.3e}, {c2:.3e}, {c3:.3e}, {c4:.3e}]"
            # )

        # if abs(Fnet) < 1e-11:
        #     self.settings.final_time = float(self.t.value)


my_model = Custom2DProblem()
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
    F.Interface(id=101, subdomains=[solid_volume, liquid_volume], penalty_term=PENALTY)
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
        subdomain=top, species=H, pressure=P_top, H_0=K_S_0_liquid, E_H=E_K_S_liquid
    )
]
bc_bottom = [
    F.SievertsBC(
        subdomain=bottom, species=H, pressure=P_bottom, S_0=K_S_0_solid, E_S=E_K_S_solid
    )
]

bc_liquid_gas = [
    F.FixedConcentrationBC(subdomain=liquid_gas_surface, species=H, value=p_b0 * K_H_T)
]
bc_solid_gas = [
    F.FixedConcentrationBC(
        subdomain=solid_gas_surface,
        species=H,
        value=(p_b0**0.5) * K_S_T if p_b0 > 0.0 else 0.0,
    )
]

my_model.boundary_conditions = bc_top + bc_bottom + bc_liquid_gas + bc_solid_gas

my_model.settings = F.Settings(atol=ATOL, rtol=RTOL, transient=True, final_time=t_total)
my_model.settings.stepsize = F.Stepsize(
    initial_value=dt, growth_factor=1.1, cutback_factor=0.9, target_nb_iterations=10
)

flux_lg = F.SurfaceFlux(field=H, surface=liquid_gas_surface, filename=None)
flux_sg = F.SurfaceFlux(field=H, surface=solid_gas_surface, filename=None)
my_model.exports = [flux_lg, flux_sg]

my_model.initialise()

my_model.t.value = t_start
my_model.N_b = p_b0 * V_b / (R_gas * temperature)

my_model.csv_path = csv_path
my_model.points = POINTS
my_model.liquid_volume = liquid_volume
my_model.solid_volume = solid_volume
my_model.H = H
my_model.flux_lg = flux_lg
my_model.flux_sg = flux_sg

my_model.idx_bclg = 2
my_model.idx_bcsg = 3

my_model.run()
