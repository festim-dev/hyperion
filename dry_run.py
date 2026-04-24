# This is the input file for the dry run case.
# It reads the experimental data from exp_data.py, sets up the model with different material parameters and boundary conditions,
# runs the simulations, and compares the results with the experimental fluxes.

from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import festim as F
import h_transport_materials as htm

import matplotlib.pyplot as plt
import numpy as np
import os

from exp_data import dry_run

from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerErrorbar
import morethemes as mt
from drawarrow import fig_arrow

mt.set_theme("lumen")

COMM = MPI.COMM_WORLD
RANK = COMM.rank

OUTDIR = "results"
if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

model_rank = 0
_read = gmshio.read_from_msh("mesh_solid_only.msh", MPI.COMM_WORLD, model_rank)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags

diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")
D_solid = diffusivities_nickel[-1]

KJ_MOL_TO_EV = 1.0 / 96.485  # 1 kJ/mol in eV
N_A = 6.02214076e23  # Avogadro constant [mol^-1]


def kj_mol_to_ev(x):
    return x * KJ_MOL_TO_EV


def mol_to_particles(x):
    return x * N_A


out_surf = F.SurfaceSubdomain(id=3)

left_bc_top_Ni = F.SurfaceSubdomain(id=42)
left_bc_middle_Ni = F.SurfaceSubdomain(id=43)
left_bc_bottom_Ni = F.SurfaceSubdomain(id=44)

top_cap_Ni = F.SurfaceSubdomain(id=5)
top_sidewall_Ni = F.SurfaceSubdomain(id=6)
bottom_sidewall_Ni = F.SurfaceSubdomain(id=7)

mem_Ni_top = F.SurfaceSubdomain(id=8)
mem_Ni_bottom = F.SurfaceSubdomain(id=9)
bottom_cap_Ni = F.SurfaceSubdomain(id=10)

upstream_volume_surfaces = [mem_Ni_bottom, bottom_cap_Ni, bottom_sidewall_Ni]
downstream_volume_surfaces = [top_cap_Ni, top_sidewall_Ni, mem_Ni_top]

# all non-volume subdomains, used to build subdomains list and surface_to_volume map
all_surface_subdomains = [
    out_surf,
    left_bc_top_Ni,
    left_bc_middle_Ni,
    left_bc_bottom_Ni,
    top_cap_Ni,
    top_sidewall_Ni,
    bottom_sidewall_Ni,
    mem_Ni_top,
    mem_Ni_bottom,
    bottom_cap_Ni,
]


def make_materials(phi_0, E_phi):
    """
    Build a FESTIM material from permeability parameters.

    phi_0 : permeability pre-exponential [particles / (m s Pa^0.5)]
    E_phi : permeability activation energy [eV]
    """
    D_0_solid = D_solid.pre_exp.magnitude
    E_D_solid = D_solid.act_energy.magnitude

    K_S_solid = htm.Solubility(
        S_0=phi_0 / D_0_solid, E_S=E_phi - E_D_solid, law="sievert"
    )

    mat_solid = F.Material(
        D_0=D_0_solid,
        E_D=E_D_solid,
        K_S_0=K_S_solid.pre_exp.magnitude,
        E_K_S=K_S_solid.act_energy.magnitude,
        solubility_law="sievert",
    )
    return mat_solid


def make_outsurf_bc(out_surf, mode: str, H: F.Species):
    """
    Build the boundary condition for the outer surface.

    mode:
      - "flux0": zero particle flux (ideal coating)
      - "conc0": zero concentration (fully permeable / uncoated)
    """
    if mode == "flux0":
        return F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)
    if mode == "conc0":
        return F.FixedConcentrationBC(subdomain=out_surf, species=H, value=0.0)
    raise ValueError(f"Unknown outsurf mode: {mode!r}")


def run_one_temperature(
    T_K: float, P_up: float, P_down: float, outsurf_mode: str, mat_solid: F.Material
):
    solid_volume = F.VolumeSubdomain(id=2, material=mat_solid)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags

    my_model.subdomains = [solid_volume] + all_surface_subdomains
    my_model.surface_to_volume = {s: solid_volume for s in all_surface_subdomains}

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = float(T_K)

    def sieverts_bcs(surfaces, pressure):
        return [
            F.SievertsBC(
                subdomain=s,
                species=H,
                pressure=float(pressure),
                S_0=float(mat_solid.K_S_0),
                E_S=float(mat_solid.E_K_S),
            )
            for s in surfaces
        ]

    my_model.boundary_conditions = (
        sieverts_bcs(upstream_volume_surfaces, P_up)
        + [make_outsurf_bc(out_surf, outsurf_mode, H)]
        + sieverts_bcs(downstream_volume_surfaces, P_down)
    )

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    downstream_fluxes = [
        CylindricalFlux(field=H, surface=s) for s in downstream_volume_surfaces
    ]
    my_model.exports = downstream_fluxes

    my_model.initialise()
    my_model.run()

    return float(sum(flux.value for flux in downstream_fluxes))


# Experimental cases: (T [C], run label, P_up [Pa], P_down [Pa], flux [H/s], flux_err [H/s])
# flux_err converted from k=2 to k=1 (1-sigma) by dividing by 2.
exp_cases = [
    (T, run, pup, pdown, flux, flux_err / 2.0)
    for T, run, pup, pdown, flux, flux_err in dry_run
]

T_C = np.array([c[0] for c in exp_cases], dtype=float)
run_labels = [c[1] for c in exp_cases]
exp_flux = np.array([c[4] for c in exp_cases], dtype=float)
exp_flux_err = np.array([c[5] for c in exp_cases], dtype=float)

idx1 = np.array([i for i, r in enumerate(run_labels) if r == "Run 1"], dtype=int)
idx2 = np.array([i for i, r in enumerate(run_labels) if r == "Run 2"], dtype=int)

modes = ["flux0", "conc0"]

# Literature permeability data (phi_0 in mol-based units, E_phi in kJ/mol)
manual_mats = [
    ("Lee", make_materials(mol_to_particles(4.52e-7), kj_mol_to_ev(55.3))),
    ("Yamanishi", make_materials(mol_to_particles(7.08e-7), kj_mol_to_ev(54.8))),
    ("Robertson", make_materials(mol_to_particles(3.22e-7), kj_mol_to_ev(54.6))),
]

results = {}

for name, mat in manual_mats:
    for mode in modes:
        flux_list = []
        for t_c, run_id, P_up, P_down, _f_exp, _f_err in exp_cases:
            T_K = t_c + 273.15
            f_model = run_one_temperature(
                T_K=T_K,
                P_up=P_up,
                P_down=P_down,
                outsurf_mode=mode,
                mat_solid=mat,
            )
            flux_list.append(f_model)
        results[(name, mode)] = np.array(flux_list, dtype=float)


def _plot_one_run_on_ax(ax, run_name: str, idx: np.ndarray, author_colors: dict):
    temps_C = T_C[idx]
    temps_K = temps_C + 273.15

    exp_y = exp_flux[idx]
    exp_yerr = exp_flux_err[idx]

    x = temps_K.astype(float)

    authors = [name for name, _ in manual_mats]
    n_authors = len(authors)
    author_offsets = np.linspace(-8.0, 8.0, n_authors)

    alpha_authors = 0.5
    line_y_offset_factor = 0.05

    for i, author in enumerate(authors):
        c = author_colors[author]
        x_group = x + author_offsets[i]

        y_ideal = results[(author, "flux0")][idx]
        y_uncoated = results[(author, "conc0")][idx]

        for xi, yi1, yi2 in zip(x_group, y_ideal, y_uncoated):
            ax.plot(
                [xi, xi],
                [yi1 * (1 - line_y_offset_factor), yi2 * (1 + line_y_offset_factor)],
                color=c,
                linewidth=1.6,
                zorder=2,
                alpha=alpha_authors,
            )

        ax.plot(
            x_group,
            y_ideal,
            linestyle="None",
            marker="s",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.5,
            zorder=3,
            alpha=alpha_authors,
        )
        ax.plot(
            x_group,
            y_uncoated,
            linestyle="None",
            marker="^",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor=c,
            markeredgewidth=1.5,
            zorder=3,
            alpha=alpha_authors,
        )

    ax.errorbar(
        x,
        exp_y,
        yerr=exp_yerr,
        # fmt="o",
        linewidth=0,
        color="red",
        markersize=26,
        capsize=6,
        elinewidth=2.0,
        capthick=2.0,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2.0,
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(T)}" for T in temps_K], fontsize=15)
    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=15)
    ax.set_yscale("log")
    ax.set_ylim(ymin=1e16)
    ax.yaxis.get_offset_text().set_fontsize(15)


def bar_panel_runs(fname: str):
    authors = [name for name, _ in manual_mats]

    color_cycler = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    author_colors = {
        authors[0]: color_cycler[0],
        authors[1]: color_cycler[1],
        authors[2]: color_cycler[2],
    }

    fig, ax = plt.subplots(1, 1)

    ax.set_xlabel("Temperature [K]", fontsize=15)
    ax.set_ylabel("Downstream flux [H/s]", fontsize=15)

    _plot_one_run_on_ax(ax, "Run 1", idx1, author_colors)
    ax.tick_params(axis="x", labelbottom=True)

    author_handles = [
        Line2D([0], [0], color=author_colors[name], lw=2, label=name)
        for name, _ in manual_mats
    ]
    style_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="0.3",
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=1.8,
            markersize=9,
            label="ideal coating",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="0.3",
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=1.8,
            markersize=10,
            label="uncoated",
        ),
    ]

    exp_handle = ax.errorbar(
        [],
        [],
        yerr=[[1], [1]],
        fmt="o",
        color="red",
        ecolor="red",
        markersize=13,
        capsize=6,
        elinewidth=2.0,
        capthick=2.0,
        markerfacecolor="none",
        markeredgecolor="red",
        markeredgewidth=2.0,
        linestyle="None",
        label=r"Experiment",
    )

    handles = author_handles + style_handles + [exp_handle]

    authors_annotation_pos = {
        authors[0]: (873, 5e16),
        authors[1]: (873, 22e16),
        authors[2]: (873, 4e16),
    }

    n_authors = len(authors)
    author_offsets = np.linspace(-8.0, 8.0, n_authors)

    for handle in author_handles:
        name = handle.get_label()
        x_pos = authors_annotation_pos[name][0] + author_offsets[authors.index(name)]
        y_pos = authors_annotation_pos[name][1]

        plt.annotate(
            name,
            xy=(x_pos, y_pos),
            fontsize=10,
            ha="center",
            va="bottom",
            color=author_colors[name],
            alpha=0.7,
        )

    plt.annotate(
        "Experiment", xy=(930, 3e16), fontsize=10, ha="center", va="bottom", color="red"
    )

    for head_pos in [(0.57, 0.55), (0.92, 0.65)]:
        fig_arrow(
            tail_position=(0.75, 0.35),
            head_position=head_pos,
            fill_head=False,
            color="red",
            radius=0.2,
            alpha=0.5,
            width=1.5,
            head_width=5,
            head_length=5,
        )

    plt.annotate(
        "Ideal coating",
        xy=(785, 3.3e16),
        fontsize=10,
        ha="left",
        va="bottom",
        color="tab:grey",
    )
    plt.annotate(
        "Uncoated",
        xy=(785, 2.2e16),
        fontsize=10,
        ha="left",
        va="bottom",
        color="tab:grey",
    )

    # fig.legend(
    #     handles,
    #     [h.get_label() for h in handles],
    #     loc="upper center",
    #     ncol=3,
    #     bbox_to_anchor=(0.5, 0.94),
    #     frameon=True,
    #     columnspacing=2.0,
    #     handletextpad=0.8,
    #     borderpad=0.5,
    #     handler_map={type(exp_handle): HandlerErrorbar(numpoints=1)},
    # )

    fig.subplots_adjust(left=0.12, right=0.985, top=0.84, bottom=0.09, hspace=0.12)

    fpath = os.path.join(OUTDIR, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    # plt.show()

    if RANK == 0:
        print(f"Saved: {fpath}")


if RANK == 0:
    bar_panel_runs("dry_run_comparison.pdf")
