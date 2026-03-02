import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F

import ufl
from dolfinx import fem
from dolfinx.fem import assemble_scalar
import dolfinx

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


class Custom2DProblem(F.HydrogenTransportProblemDiscontinuous):
    def iterate(self):
        global old_N_b
        super().iterate()

        j_liquid_gas = float(flux_lg.value)
        j_solid_gas = float(flux_sg.value)

        Fnet = j_liquid_gas * L35 + j_solid_gas * L36

        new_N_b = max(old_N_b + float(self.dt.value) * Fnet, 0.0)
        new_pb = max(new_N_b * R_gas * temperature / V_b, 0.0)

        bc_liquid_gas.value = new_pb * K_H_T
        bc_solid_gas.value = (new_pb**0.5) * K_S_T if new_pb > 0.0 else 0.0

        idx_bclg = self.boundary_conditions.index(bc_liquid_gas)
        idx_bcsg = self.boundary_conditions.index(bc_solid_gas)
        self.bc_forms[idx_bclg] = self.create_dirichletbc_form(bc_liquid_gas)
        self.bc_forms[idx_bcsg] = self.create_dirichletbc_form(bc_solid_gas)

        old_N_b = new_N_b
        all_pbs.append(new_pb)


if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

fig_pb, ax_pb = None, None
if RANK == 0:
    fig_pb, ax_pb = plt.subplots(1, 1)

D_0_solid, E_D_solid = 2, 0.0
K_S_0_solid, E_K_S_solid = 3, 0.2

D_0_liquid, E_D_liquid = 1, 0.0
K_S_0_liquid, E_K_S_liquid = 2, 0.2

PENALTY = 1e5

ATOL = 1e-10
RTOL = 1e-8

R_gas = 8.314
V_b = 10000

t_total = 1000

c_up = 1.0
c_down = 0.0


Te_Cs = (500, 550, 600, 650, 700)
temperatures = tuple(float(T) + 273.15 for T in Te_Cs)

for temperature in temperatures:
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

    K_H_T = float(
        mat_liquid.get_solubility_coefficient(mesh=mesh, temperature=temperature)
    )
    K_S_T = float(
        mat_solid.get_solubility_coefficient(mesh=mesh, temperature=temperature)
    )

    solid_volume = F.VolumeSubdomain(id=1, material=mat_solid)
    liquid_volume = F.VolumeSubdomain(id=2, material=mat_liquid)

    top = F.SurfaceSubdomain(id=33)
    bottom = F.SurfaceSubdomain(id=34)
    liquid_gas_surface = F.SurfaceSubdomain(id=35)
    solid_gas_surface = F.SurfaceSubdomain(id=36)
    liquid_solid_interface = F.SurfaceSubdomain(id=101)

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
    my_model.temperature = float(temperature)

    bc_top = [F.FixedConcentrationBC(subdomain=top, species=H, value=c_up)]
    bc_bottom = [
        F.FixedConcentrationBC(
            subdomain=bottom,
            species=H,
            value=c_down,
        )
    ]
    p_b0 = 0.0
    all_pbs = []
    bc_liquid_gas = F.FixedConcentrationBC(
        subdomain=liquid_gas_surface, species=H, value=p_b0 * K_H_T
    )
    bc_solid_gas = F.FixedConcentrationBC(
        subdomain=solid_gas_surface,
        species=H,
        value=(p_b0**0.5) * K_S_T if p_b0 > 0.0 else 0.0,
    )

    my_model.boundary_conditions = bc_top + bc_bottom + [bc_liquid_gas, bc_solid_gas]

    my_model.settings = F.Settings(
        atol=ATOL, rtol=RTOL, transient=True, final_time=t_total
    )
    my_model.settings.stepsize = F.Stepsize(
        initial_value=1,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=10,
    )

    flux_lg = F.SurfaceFlux(field=H, surface=liquid_gas_surface, filename=None)
    flux_sg = F.SurfaceFlux(field=H, surface=solid_gas_surface, filename=None)
    my_model.exports = [flux_lg, flux_sg]

    my_model.initialise()

    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    old_N_b = p_b0 * V_b / (R_gas * temperature)

    my_model.run()

    if RANK == 0 and len(all_pbs) > 0:
        t_arr = np.array(flux_lg.t, dtype=float)
        pb_arr = np.array(all_pbs, dtype=float)

        csv_pb = os.path.join(OUTDIR, f"pb_{T_label_from_temperature(temperature)}.csv")
        np.savetxt(
            csv_pb,
            np.column_stack([t_arr, pb_arr]),
            delimiter=",",
            header="t,p_b",
            comments="",
        )

        if ax_pb is not None:
            ax_pb.plot(t_arr, pb_arr, label=T_label_from_temperature(temperature))

if RANK == 0 and ax_pb is not None:
    ax_pb.set_xlabel("Time (s)")
    ax_pb.set_ylabel("Bubble pressure (Pa)")
    ax_pb.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig_pb.tight_layout()
    fig_pb.savefig(os.path.join(OUTDIR, "pb_all_temperatures.png"), dpi=200)
    plt.close(fig_pb)
