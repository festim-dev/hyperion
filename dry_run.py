from mesh import generate_mesh
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import festim as F
import h_transport_materials as htm

import matplotlib.pyplot as plt
import numpy as np
import os

COMM = MPI.COMM_WORLD
RANK = COMM.rank

OUTDIR = "dry_run_outputs"
if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

model_rank = 0
_read = gmshio.read_from_msh("mesh_solid_only.msh", MPI.COMM_WORLD, model_rank)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags

diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")
D_solid = diffusivities_nickel[-1]


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


def make_materials(P_0, E_P):
    D_0_solid = D_solid.pre_exp.magnitude
    E_D_solid = D_solid.act_energy.magnitude

    K_S_solid = htm.Solubility(S_0=P_0 / D_0_solid, E_S=E_P - E_D_solid, law="sievert")

    K_S_0_solid = K_S_solid.pre_exp.magnitude
    E_K_S_solid = K_S_solid.act_energy.magnitude

    mat_solid = F.Material(
        D_0=D_0_solid,
        E_D=E_D_solid,
        K_S_0=K_S_0_solid,
        E_K_S=E_K_S_solid,
        solubility_law="sievert",
    )
    return mat_solid


def make_outsurf_bc(out_surf, mode: str, H: F.Species):
    """
    mode:
      - "flux0": impose outsurface particle flux = 0
      - "conc0": impose concentration = 0 at outsurface
    """
    if mode == "flux0":
        return F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)

    if mode == "conc0":
        return F.FixedConcentrationBC(subdomain=out_surf, species=H, value=0.0)


def run_one_temperature(
    T_K: float, P_up: float, P_down: float, outsurf_mode: str, mat_solid: F.Material
):
    solid_volume = F.VolumeSubdomain(id=2, material=mat_solid)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags

    my_model.subdomains = [
        solid_volume,
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

    my_model.surface_to_volume = {
        out_surf: solid_volume,
        left_bc_top_Ni: solid_volume,
        left_bc_middle_Ni: solid_volume,
        left_bc_bottom_Ni: solid_volume,
        top_cap_Ni: solid_volume,
        top_sidewall_Ni: solid_volume,
        bottom_sidewall_Ni: solid_volume,
        mem_Ni_top: solid_volume,
        mem_Ni_bottom: solid_volume,
        bottom_cap_Ni: solid_volume,
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = float(T_K)

    out_surface_bc = make_outsurf_bc(out_surf, outsurf_mode, H)

    my_model.boundary_conditions = (
        [
            F.SievertsBC(
                subdomain=s,
                species=H,
                pressure=float(P_up),
                S_0=float(mat_solid.K_S_0),
                E_S=float(mat_solid.E_K_S),
            )
            for s in upstream_volume_surfaces
        ]
        + [out_surface_bc]
        + [
            F.SievertsBC(
                subdomain=s,
                species=H,
                pressure=float(P_down),
                S_0=float(mat_solid.K_S_0),
                E_S=float(mat_solid.E_K_S),
            )
            for s in downstream_volume_surfaces
        ]
    )

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    downstream_fluxes = [
        CylindricalFlux(field=H, surface=s) for s in downstream_volume_surfaces
    ]
    my_model.exports = downstream_fluxes

    my_model.initialise()
    my_model.run()

    total_downstream_flux = float(sum(flux.value for flux in downstream_fluxes))
    return total_downstream_flux


exp_cases = [
    (500.0, "Run 1", 1.30e5, 1.98e2, 4.52e16),
    (500.0, "Run 2", 1.10e5, 1.79e2, 4.08e16),
    (600.0, "Run 1", 1.30e5, 4.59e2, 1.03e17),
    (600.0, "Run 2", 1.10e5, 4.02e2, 9.19e16),
    (700.0, "Run 1", 1.30e5, 8.16e2, 1.85e17),
    (700.0, "Run 2", 1.10e5, 7.36e2, 1.65e17),
]
exp_cases = [(T, run, pup, pdown, flux / 2.0) for T, run, pup, pdown, flux in exp_cases]

T_C = np.array([c[0] for c in exp_cases], dtype=float)
run_labels = [c[1] for c in exp_cases]
exp_flux = np.array([c[4] for c in exp_cases], dtype=float)

idx1 = np.array([i for i, r in enumerate(run_labels) if r == "Run 1"], dtype=int)
idx2 = np.array([i for i, r in enumerate(run_labels) if r == "Run 2"], dtype=int)

modes = ["flux0", "conc0"]

results = {}

# permeability_data = [
#     ("Lee", 4.52e-7, 55.3),
#     ("Yamanishi et al.", 7.08e-7, 54.8),
#     ("Masui et al.", 5.21e-7, 54.4),
#     ("Ebisuzaki et al.", 4.05e-7, 55.1)
# ("Robertson", 3.22e-7, 54.6)
# ]


KJ_MOL_TO_EV = 1.0 / 96.485  # 1 kJ/mol = 0.010364 eV


def kj_mol_to_ev(x):
    return x * KJ_MOL_TO_EV


N_A = 6.02214076e23  # particles/mol


def mol_to_particles(x):
    return x * N_A


manual_mats = [
    ("Lee", make_materials(mol_to_particles(4.52e-7), kj_mol_to_ev(55.3))),
    ("Yamanishi", make_materials(mol_to_particles(7.08e-7), kj_mol_to_ev(54.8))),
    # ("Masui", make_materials(mol_to_particles(5.21e-7), kj_mol_to_ev(54.4))),
    # ("Ebisuzaki", make_materials(mol_to_particles(4.05e-07), kj_mol_to_ev(55.1))),
    ("Robertson", make_materials(mol_to_particles(3.22e-07), kj_mol_to_ev(54.6))),
]

# manual_mats = [
#     (
#         "Run1, ideal coating",
#         make_materials(mol_to_particles(5.196e-8), kj_mol_to_ev(45.502)),
#     ),
#     (
#         "Run1, no coating",
#         make_materials(mol_to_particles(4.516e-07), kj_mol_to_ev(57.592)),
#     ),
#     # (
#     #     "Run2, ideal coating",
#     #     make_materials(mol_to_particles(4.831e-08), kj_mol_to_ev(45.157)),
#     # ),
#     # (
#     #     "Run2, no coating",
#     #     make_materials(mol_to_particles(4.687e-07), kj_mol_to_ev(57.913)),
#     # ),
# ]

for name, mat in manual_mats:
    for mode in modes:
        flux_list = []
        for t_c, run_id, P_up, P_down, _f_exp in exp_cases:
            T_K = t_c + 273.15
            f_model = run_one_temperature(
                T_K=T_K,
                P_up=P_up,
                P_down=P_down,
                outsurf_mode=mode,
                mat_solid=mat,
            )
            flux_list.append(f_model)
            # print(f_model)

        results[(name, mode)] = np.array(flux_list, dtype=float)


# def bar_run(run_name: str, idx: np.ndarray, fname: str):
#     temps = T_C[idx]
#     exp_y = exp_flux[idx]

#     x = np.arange(len(temps))

#     fig, ax = plt.subplots(figsize=(9, 5))

#     authors = [name for name, _ in manual_mats]
#     cmap = plt.get_cmap("tab10")
#     author_colors = {author: cmap(i) for i, author in enumerate(authors)}

#     sim_series = []
#     for name, _mat in manual_mats:
#         for mode in modes:
#             y = results[(name, mode)][idx]
#             sim_series.append((name, mode, y))

#     n_series = len(sim_series)
#     group_width = 0.8
#     bar_w = group_width / n_series
#     offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

#     for i, (name, mode, y) in enumerate(sim_series):
#         filled = mode == "flux0"

#         ax.bar(
#             x + offsets[i],
#             y,
#             width=bar_w,
#             color=author_colors[name] if filled else "none",
#             edgecolor=author_colors[name],
#             linewidth=1.5,
#             label=f"{name} | {'ideal coating' if filled else 'no coating'}",
#             zorder=2,
#         )

#     ax.scatter(
#         x,
#         exp_y,
#         color="red",
#         s=80,
#         marker="o",
#         label="Experiment",
#         zorder=5,
#     )

#     ax.set_xticks(x)
#     ax.set_xticklabels([f"{int(t)}°C" for t in temps])

#     ax.set_xlabel("Temperature")
#     ax.set_ylabel("Downstream flux [H/s]")
#     # ax.set_title(f"{run_name}: Experimental vs Simulation")

#     ax.grid(True, axis="y", alpha=0.3)

#     ax.legend(ncol=2, fontsize=9)

#     fig.tight_layout()

#     fpath = os.path.join(OUTDIR, fname)
#     fig.savefig(fpath, dpi=300)
#     plt.close(fig)

#     if RANK == 0:
#         print(f"Saved: {fpath}")


# if RANK == 0:
#     bar_run("Run 1", idx1, "barchart_run1_exp_vs_sim.svg")
#     bar_run("Run 2", idx2, "barchart_run2_exp_vs_sim.svg")


def _plot_one_run_on_ax(ax, run_name: str, idx: np.ndarray, author_colors: dict):
    temps = T_C[idx]
    exp_y = exp_flux[idx]

    x = np.arange(len(temps))

    sim_series = []
    for name, _mat in manual_mats:
        for mode in modes:
            y = results[(name, mode)][idx]
            sim_series.append((name, mode, y))

    n_series = len(sim_series)
    group_width = 0.8
    bar_w = group_width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

    for i, (name, mode, y) in enumerate(sim_series):
        filled = mode == "flux0"

        ax.bar(
            x + offsets[i],
            y,
            width=bar_w,
            color=author_colors[name] if filled else "none",
            edgecolor=author_colors[name],
            linewidth=1.5,
            label=f"{name} | {'ideal coating' if filled else 'no coating'}",
            zorder=2,
        )

    ax.scatter(
        x,
        exp_y,
        color="red",
        s=80,
        marker="o",
        label="Experiment",
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(t)}°C" for t in temps], fontsize=15)
    # ax.set_xlabel("Temperature")
    # ax.set_title(run_name)
    ax.grid(True, axis="y", alpha=0.3)


def bar_panel_runs(fname="barchart_runs_panel_exp_vs_sim.svg"):
    authors = [name for name, _ in manual_mats]
    cmap = plt.get_cmap("tab10")
    author_colors = {author: cmap(i) for i, author in enumerate(authors)}

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 8.2), sharex=True, sharey=True)
    # fig.text(0.5, 0.9, "(a)", ha="center", va="bottom", fontsize=15, fontweight="bold")
    # fig.text(0.5, 0.5, "(b)", ha="center", va="bottom", fontsize=15, fontweight="bold")

    fig.supylabel("Downstream flux [H/s]", fontsize=15)
    axes[1].set_xlabel("Temperature", fontsize=15)

    _plot_one_run_on_ax(axes[0], "Run 1", idx1, author_colors)
    _plot_one_run_on_ax(axes[1], "Run 2", idx2, author_colors)

    # axes[0].set_xlabel("")  # remove top x label
    # axes[1].set_xlabel("Temperature", fontsize=15)

    # axes[0].set_ylabel("Downstream flux [H/s]", fontsize=15)

    # build unique legend handles/labels from both axes
    handles = []
    labels = []
    for ax in axes:
        ax.tick_params(axis="both", labelsize=15)
        ax.yaxis.get_offset_text().set_fontsize(15)
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)

    axes[0].text(
        0.5,
        1.01,
        "(a)",
        transform=axes[0].transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

    axes[1].text(
        0.5,
        1.01,
        "(b)",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        fontsize=12,
        frameon=True,
        columnspacing=1.6,
        handletextpad=0.7,
        borderpad=0.5,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.84, bottom=0.09, hspace=0.12)

    # axes[0].text(
    #     0.02,
    #     0.95,
    #     "(a)",
    #     transform=axes[0].transAxes,
    #     fontsize=15,
    #     fontweight="bold",
    #     va="center_baseline",
    # )

    # axes[1].text(
    #     0.02,
    #     0.95,
    #     "(b)",
    #     transform=axes[1].transAxes,
    #     fontsize=15,
    #     fontweight="bold",
    #     va="center_baseline",
    # )

    fpath = os.path.join(OUTDIR, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    if RANK == 0:
        print(f"Saved: {fpath}")


if RANK == 0:
    bar_panel_runs("barchart_run1_run2_panel_exp_vs_sim.svg")
