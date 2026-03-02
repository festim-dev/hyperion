import numpy as np
from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F

import ufl
from dolfinx import fem
from dolfinx.fem import assemble_scalar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

OUTDIR = "plots"

COMM = MPI.COMM_WORLD
RANK = COMM.rank


def T_label_from_temperature(T):
    return f"T{int(round(T))}K"


MESH_FILE = "2D.msh"
_read = gmshio.read_from_msh(MESH_FILE, COMM, rank=0, gdim=2)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags


def facet_measure(tag_id: int) -> float:
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
    one = fem.Constant(mesh, 1.0)
    val_local = assemble_scalar(fem.form(one * ds(tag_id)))
    return COMM.allreduce(val_local, op=MPI.SUM)


L_top = facet_measure(33)
L_bottom = facet_measure(34)

if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

D_0_solid, E_D_solid = 2, 0.0
K_S_0_solid, E_K_S_solid = 3, 0.2

D_0_liquid, E_D_liquid = 1, 0.0
K_S_0_liquid, E_K_S_liquid = 2, 0.2

PENALTY = 1e5

ATOL = 1e-10
RTOL = 1e-8

t_total = 1000

c_up = 1.0
c_down = 0.0

Te_Cs = (500, 550, 600, 650, 700)
temperatures = tuple(float(T) + 273.15 for T in Te_Cs)

fig_nb, ax_nb = (None, None)
if RANK == 0:
    fig_nb, ax_nb = plt.subplots(1, 1)
    fig_flux, ax_flux = plt.subplots(1, 1)

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

    solid_volume = F.VolumeSubdomain(id=1, material=mat_solid)
    liquid_volume = F.VolumeSubdomain(id=2, material=mat_liquid)

    top = F.SurfaceSubdomain(id=33)
    bottom = F.SurfaceSubdomain(id=34)
    liquid_solid_interface = F.SurfaceSubdomain(id=101)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cartesian")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags

    my_model.subdomains = [
        solid_volume,
        liquid_volume,
        top,
        bottom,
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
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = float(temperature)

    bc_top = F.FixedConcentrationBC(subdomain=top, species=H, value=c_up)
    bc_bottom = F.FixedConcentrationBC(subdomain=bottom, species=H, value=c_down)
    my_model.boundary_conditions = [bc_top, bc_bottom]

    my_model.settings = F.Settings(
        atol=ATOL, rtol=RTOL, transient=True, final_time=t_total
    )
    my_model.settings.stepsize = F.Stepsize(
        initial_value=1,
        growth_factor=1.1,
        cutback_factor=0.9,
        target_nb_iterations=10,
    )

    flux_down = F.SurfaceFlux(field=H, surface=bottom, filename=None)
    my_model.exports = [flux_down]

    my_model.initialise()
    my_model.run()

    if RANK == 0:
        t_arr = np.array(flux_down.t, dtype=float)
        j_arr = np.array(flux_down.data, dtype=float)

        csv_flux = os.path.join(
            OUTDIR, f"noBubble_fluxDown_{T_label_from_temperature(temperature)}.csv"
        )
        np.savetxt(
            csv_flux,
            np.column_stack([t_arr, j_arr]),
            delimiter=",",
            header="t,flux_down",
            comments="",
        )

        if ax_nb is not None:
            ax_nb.plot(t_arr, j_arr, label=T_label_from_temperature(temperature))

if RANK == 0 and ax_nb is not None:
    ax_nb.set_xlabel("Time (s)")
    ax_nb.set_ylabel("Downstream flux (no bubble)")
    ax_nb.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig_nb.tight_layout()
    fig_nb.savefig(
        os.path.join(OUTDIR, "noBubble_fluxDown_all_temperatures.png"), dpi=200
    )
    plt.close(fig_nb)
