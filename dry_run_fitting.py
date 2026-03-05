import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F
import h_transport_materials as htm

from cylindrical_flux import CylindricalFlux


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


exp_cases = [
    (500.0, "Run 1", 1.30e5, 1.98e2, 4.52e16),
    (500.0, "Run 2", 1.10e5, 1.79e2, 4.08e16),
    (600.0, "Run 1", 1.30e5, 4.59e2, 1.03e17),
    (600.0, "Run 2", 1.10e5, 4.02e2, 9.19e16),
    (700.0, "Run 1", 1.30e5, 8.16e2, 1.85e17),
    (700.0, "Run 2", 1.10e5, 7.36e2, 1.65e17),
]

exp_cases = [(T, run, pup, pdown, flux / 2.0) for T, run, pup, pdown, flux in exp_cases]


KJ_MOL_TO_EV = 1.0 / 96.485
N_A = 6.02214076e23
K_B_EV_PER_K = 8.617333262145e-5


def mol_to_particles(x):
    return x * N_A


def kjmol_to_ev(x):
    return x * KJ_MOL_TO_EV


def ev_to_kjmol(x):
    return x / KJ_MOL_TO_EV


def make_material(P0_mol_based, EP_kJmol):
    D0 = D_solid.pre_exp.magnitude
    ED = D_solid.act_energy.magnitude

    P0_particles = mol_to_particles(P0_mol_based)
    EP_eV = kjmol_to_ev(EP_kJmol)

    K_S = htm.Solubility(S_0=P0_particles / D0, E_S=EP_eV - ED, law="sievert")
    mat = F.Material(
        D_0=D0,
        E_D=ED,
        K_S_0=K_S.pre_exp.magnitude,
        E_K_S=K_S.act_energy.magnitude,
        solubility_law="sievert",
    )
    return mat


def make_outsurf_bc(mode, H):
    if mode == "flux0":
        return F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)
    if mode == "conc0":
        return F.FixedConcentrationBC(subdomain=out_surf, species=H, value=0.0)
    raise ValueError(mode)


def run_one(T_K, P_up, P_down, mode, mat):
    solid = F.VolumeSubdomain(id=2, material=mat)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags

    my_model.subdomains = [
        solid,
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
        out_surf: solid,
        left_bc_top_Ni: solid,
        left_bc_middle_Ni: solid,
        left_bc_bottom_Ni: solid,
        top_cap_Ni: solid,
        top_sidewall_Ni: solid,
        bottom_sidewall_Ni: solid,
        mem_Ni_top: solid,
        mem_Ni_bottom: solid,
        bottom_cap_Ni: solid,
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = float(T_K)

    bc_out = make_outsurf_bc(mode, H)

    my_model.boundary_conditions = (
        [
            F.SievertsBC(
                subdomain=s,
                species=H,
                pressure=float(P_up),
                S_0=float(mat.K_S_0),
                E_S=float(mat.E_K_S),
            )
            for s in upstream_volume_surfaces
        ]
        + [bc_out]
        + [
            F.SievertsBC(
                subdomain=s,
                species=H,
                pressure=float(P_down),
                S_0=float(mat.K_S_0),
                E_S=float(mat.E_K_S),
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

    return float(sum(f.value for f in downstream_fluxes))


def restore_Peff_all(mode, P_ref):
    mat_ref = make_material(P_ref, 0.0)

    rows = []
    for T_C, run_id, Pup, Pdown, flux_exp in exp_cases:
        T_K = T_C + 273.15
        flux_ref = run_one(T_K, Pup, Pdown, mode, mat_ref)
        P_eff = P_ref * (flux_exp / flux_ref)

        rows.append((T_C, run_id, Pup, Pdown, flux_exp, flux_ref, P_eff))

        if RANK == 0:
            print(
                f"[{mode}] {T_C:.0f}C {run_id}: flux_ref={flux_ref:.3e}, P_eff={P_eff:.3e}"
            )

    return rows


def fit_arrhenius(T_C_vals, P_vals):
    T_K = np.array(T_C_vals, dtype=float) + 273.15
    x = 1.0 / T_K
    y = np.log(np.array(P_vals, dtype=float))

    # y = a + b x
    b, a = np.polyfit(x, y, 1)

    P0 = float(np.exp(a))
    EP_eV = float(-b * K_B_EV_PER_K)
    EP_kJmol = float(ev_to_kjmol(EP_eV))
    return P0, EP_kJmol


if __name__ == "__main__":
    P_REF = 1e-7

    modes = ["flux0", "conc0"]

    all_results = {}

    for mode in modes:
        if RANK == 0:
            print(f"\n=== Restoring P_eff(T) for mode={mode} (no bisection) ===")
        rows = restore_Peff_all(mode, P_REF)
        all_results[mode] = rows

    if RANK != 0:
        raise SystemExit

    csv_path = os.path.join(OUTDIR, "restored_Peff.csv")
    with open(csv_path, "w") as f:
        f.write("mode,T_C,run,P_up,P_down,flux_exp,flux_ref(Pref),P_eff\n")
        for mode in modes:
            for T_C, run_id, Pup, Pdown, flux_exp, flux_ref, P_eff in all_results[mode]:
                f.write(
                    f"{mode},{T_C},{run_id},{Pup:.8e},{Pdown:.8e},{flux_exp:.8e},{flux_ref:.8e},{P_eff:.8e}\n"
                )
    print(f"Saved: {csv_path}")

    fit_lines = []
    for mode in modes:
        for run_id in ["Run 1", "Run 2"]:
            T_list = [r[0] for r in all_results[mode] if r[1] == run_id]
            P_list = [r[6] for r in all_results[mode] if r[1] == run_id]

            P0, EP = fit_arrhenius(T_list, P_list)
            fit_lines.append((mode, run_id, P0, EP))

    # Print final expressions
    print("\n=== Final permeability expressions (separate fits) ===")
    print("Form:  P(T) = P0 * exp( -EP / (kB*T) )   with EP in kJ/mol")
    for mode, run_id, P0, EP in fit_lines:
        print(
            f"{mode:5s} | {run_id}:  P(T) = {P0:.3e} * exp( -{EP:.3f} kJ/mol / (kB*T) )"
        )

    txt_path = os.path.join(OUTDIR, "P_arrhenius_expressions.txt")
    with open(txt_path, "w") as f:
        f.write("P(T) = P0 * exp( -EP / (kB*T) )  (EP reported in kJ/mol)\n")
        for mode, run_id, P0, EP in fit_lines:
            f.write(f"{mode} | {run_id}: P0={P0:.10e}, EP_kJmol={EP:.6f}\n")
    print(f"Saved: {txt_path}")

    fig, ax = plt.subplots(figsize=(7, 4))
    markers = {
        ("flux0", "Run 1"): "o",
        ("flux0", "Run 2"): "s",
        ("conc0", "Run 1"): "^",
        ("conc0", "Run 2"): "D",
    }

    for mode in modes:
        for run_id in ["Run 1", "Run 2"]:
            T_list = np.array(
                [r[0] for r in all_results[mode] if r[1] == run_id], float
            )
            P_list = np.array(
                [r[6] for r in all_results[mode] if r[1] == run_id], float
            )
            invT = 1.0 / (T_list + 273.15)
            ax.scatter(
                invT, P_list, marker=markers[(mode, run_id)], label=f"{mode} {run_id}"
            )

    ax.set_yscale("log")
    ax.set_xlabel("1/T [1/K]")
    ax.set_ylabel("Restored P_eff")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fpath = os.path.join(OUTDIR, "restored_Peff_vs_invT.png")
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {fpath}")
