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

generate_mesh(mesh_size=2e-4)
model_rank = 0
_read = gmshio.read_from_msh("mesh_solid_only.msh", MPI.COMM_WORLD, model_rank)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags

diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")
perms_nickel = htm.permeabilities.filter(material="nickel").filter(isotope="h")

D_solid_obj = diffusivities_nickel[-1]


wanted = ["lee", "yamanishi", "masui"]
candidates = []
for p in perms_nickel:
    a = str(getattr(p, "author", "")).lower()
    if any(w in a for w in wanted):
        candidates.append(p)


def pick_unique_three(perms):
    picked = {}
    for p in perms:
        a = str(getattr(p, "author", "")).lower()
        for key in wanted:
            if key in a and key not in picked:
                picked[key] = p
    return [picked[k] for k in wanted if k in picked]


chosen_perms = pick_unique_three(candidates)

solid_volume = F.VolumeSubdomain(id=2, material=None)

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
liquid_solid_interface = F.SurfaceSubdomain(id=99)

upstream_volume_surfaces = [mem_Ni_bottom, bottom_cap_Ni, bottom_sidewall_Ni]
downstream_volume_surfaces = [top_cap_Ni, top_sidewall_Ni, mem_Ni_top]


def make_outsurf_bc(mode: str, H: F.Species):
    """
    mode:
      - "flux0": impose outsurface particle flux = 0
      - "conc0": impose concentration = 0 at outsurface
    """
    mode = mode.lower().strip()
    if mode == "flux0":
        return F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)

    if mode == "conc0":
        if hasattr(F, "DirichletBC"):
            return F.DirichletBC(subdomain=out_surf, species=H, value=0.0)
        if hasattr(F, "FixedConcentrationBC"):
            return F.FixedConcentrationBC(subdomain=out_surf, species=H, value=0.0)
        raise AttributeError(
            "Cannot find a concentration Dirichlet BC in festim. "
            "Tried F.DirichletBC and F.FixedConcentrationBC."
        )

    raise ValueError(f"Unknown outsurface BC mode: {mode}. Use 'flux0' or 'conc0'.")


def make_solid_material_from_perm(D_obj, perm_obj) -> F.Material:
    K_obj = htm.Solubility(
        S_0=perm_obj.pre_exp / D_obj.pre_exp,
        E_S=perm_obj.act_energy - D_obj.act_energy,
        law=getattr(perm_obj, "law", "sievert"),
    )
    return F.Material(
        D_0=D_obj.pre_exp.magnitude,
        E_D=D_obj.act_energy.magnitude,
        K_S_0=K_obj.pre_exp.magnitude,
        E_K_S=K_obj.act_energy.magnitude,
        solubility_law="sievert",
    )


# -----------------------------
def run_one_temperature(
    T_K: float, P_up: float, P_down: float, outsurf_mode: str, mat_solid: F.Material
):
    solid_volume.material = mat_solid

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
        liquid_solid_interface,
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

    out_surface_bc = make_outsurf_bc(outsurf_mode, H)

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

for perm in chosen_perms:
    mat = make_solid_material_from_perm(D_solid_obj, perm)

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

        results[(perm.author, mode)] = np.array(flux_list, dtype=float)


def plot_run(run_name: str, idx: np.ndarray, fname: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        T_C[idx],
        exp_flux[idx],
        marker="o",
        linewidth=2,
        label=f"Experiment ({run_name})",
    )

    for perm in chosen_perms:
        for mode in modes:
            y = results[(perm.author, mode)][idx]
            label = f"{perm.author} | out: {'flux=0' if mode == 'flux0' else 'C=0'}"
            ax.plot(
                T_C[idx],
                y,
                marker="s" if mode == "flux0" else "^",
                linewidth=2,
                label=label,
            )

    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Downstream flux [H/s]")
    ax.set_title(
        f"{run_name}: downstream flux (3 permeabilities × 2 out-surface BC) vs exp"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=9)
    fig.tight_layout()

    fpath = os.path.join(OUTDIR, fname)
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {fpath}")


if RANK == 0:
    plot_run("Run 1", idx1, "compare_run1_7curves.png")
    plot_run("Run 2", idx2, "compare_run2_7curves.png")
