"""
Restore effective permeability (phi_eff) from experimental downstream flux data.

For each experimental case (temperature, upstream/downstream pressure), a single
steady-state simulation is run with an arbitrary reference permeability (phi_ref).
Because flux scales linearly with permeability under Sievert's law, phi_eff is
recovered by simple proportionality:

    phi_eff = phi_ref * (flux_exp / flux_sim_at_phi_ref)

Arrhenius fits are then applied to phi_eff(T) for each boundary condition mode
and run, yielding phi_0 and E_phi for use in subsequent simulations.

Outputs (saved to results/):
    dry_run_phi_per_case.csv       -- phi_eff for every experimental case
    dry_run_phi_arrhenius_fits.txt -- Arrhenius fit parameters
    dry_run_phi_vs_invT.png        -- phi_eff vs 1/T plot
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
import festim as F
import h_transport_materials as htm

from cylindrical_flux import CylindricalFlux
from exp_data import dry_run

COMM = MPI.COMM_WORLD
RANK = COMM.rank

OUTDIR = "results"
if RANK == 0:
    os.makedirs(OUTDIR, exist_ok=True)

# exp_cases drops flux_err (k=2) since it is not needed here
exp_cases = [(T, run, pup, pdown, flux) for T, run, pup, pdown, flux, _ in dry_run]

model_rank = 0
_read = gmshio.read_from_msh("mesh_solid_only.msh", MPI.COMM_WORLD, model_rank)
mesh = _read.mesh
cell_tags = _read.cell_tags
facet_tags = _read.facet_tags

diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(isotope="h")
D_solid = diffusivities_nickel[-1]

KJ_MOL_TO_EV = 1.0 / 96.485  # 1 kJ/mol in eV
N_A = 6.02214076e23  # Avogadro constant [mol^-1]
K_B_EV_PER_K = 8.617333262145e-5  # Boltzmann constant [eV/K]


def mol_to_particles(x):
    return x * N_A


def kjmol_to_ev(x):
    return x * KJ_MOL_TO_EV


def ev_to_kjmol(x):
    return x / KJ_MOL_TO_EV


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


def make_material(phi_0_mol_based, E_phi_kJmol):
    """
    Build a FESTIM material from permeability parameters.

    phi_0_mol_based : permeability pre-exponential in mol-based units
    E_phi_kJmol     : permeability activation energy [kJ/mol]
    """
    D0 = D_solid.pre_exp.magnitude
    ED = D_solid.act_energy.magnitude

    phi_0_particles = mol_to_particles(phi_0_mol_based)
    E_phi_eV = kjmol_to_ev(E_phi_kJmol)

    K_S = htm.Solubility(S_0=phi_0_particles / D0, E_S=E_phi_eV - ED, law="sievert")
    return F.Material(
        D_0=D0,
        E_D=ED,
        K_S_0=K_S.pre_exp.magnitude,
        E_K_S=K_S.act_energy.magnitude,
        solubility_law="sievert",
    )


def make_outsurf_bc(mode, H):
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


def run_one(T_K, P_up, P_down, mode, mat):
    solid = F.VolumeSubdomain(id=2, material=mat)

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags
    my_model.subdomains = [solid] + all_surface_subdomains
    my_model.surface_to_volume = {s: solid for s in all_surface_subdomains}

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = float(T_K)

    def sieverts_bcs(surfaces, pressure):
        return [
            F.SievertsBC(
                subdomain=s,
                species=H,
                pressure=float(pressure),
                S_0=float(mat.K_S_0),
                E_S=float(mat.E_K_S),
            )
            for s in surfaces
        ]

    my_model.boundary_conditions = (
        sieverts_bcs(upstream_volume_surfaces, P_up)
        + [make_outsurf_bc(mode, H)]
        + sieverts_bcs(downstream_volume_surfaces, P_down)
    )

    my_model.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    downstream_fluxes = [
        CylindricalFlux(field=H, surface=s) for s in downstream_volume_surfaces
    ]
    my_model.exports = downstream_fluxes

    my_model.initialise()
    my_model.run()

    return float(sum(f.value for f in downstream_fluxes))


def restore_phi_eff_all(mode, phi_ref):
    """For each experimental case, scale phi_ref linearly to match the measured flux."""
    mat_ref = make_material(phi_ref, 0.0)

    rows = []
    for T_C, run_id, Pup, Pdown, flux_exp in exp_cases:
        T_K = T_C + 273.15
        flux_ref = run_one(T_K, Pup, Pdown, mode, mat_ref)
        phi_eff = phi_ref * (flux_exp / flux_ref)

        rows.append((T_C, run_id, Pup, Pdown, flux_exp, flux_ref, phi_eff))

        if RANK == 0:
            print(
                f"[{mode}] {T_C:.0f}C {run_id}: flux_ref={flux_ref:.3e}, phi_eff={phi_eff:.3e}"
            )

    return rows


def fit_arrhenius(T_C_vals, phi_vals):
    """Fit log(phi) vs 1/T to extract pre-exponential and activation energy."""
    T_K = np.array(T_C_vals, dtype=float) + 273.15
    x = 1.0 / T_K
    y = np.log(np.array(phi_vals, dtype=float))

    b, a = np.polyfit(x, y, 1)

    phi_0 = float(np.exp(a))
    E_phi_eV = float(-b * K_B_EV_PER_K)
    E_phi_kJmol = float(ev_to_kjmol(E_phi_eV))
    return phi_0, E_phi_kJmol


if __name__ == "__main__":
    phi_REF = 1e-7
    modes = ["flux0", "conc0"]
    all_results = {}

    for mode in modes:
        if RANK == 0:
            print(f"\n=== Restoring phi_eff(T) for mode={mode} ===")
        all_results[mode] = restore_phi_eff_all(mode, phi_REF)

    if RANK != 0:
        raise SystemExit

    # Save restored phi_eff per experimental case
    csv_path = os.path.join(OUTDIR, "dry_run_phi_per_case.csv")
    with open(csv_path, "w") as f:
        f.write("mode,T_C,run,P_up,P_down,flux_exp,flux_ref(phi_ref),phi_eff\n")
        for mode in modes:
            for T_C, run_id, Pup, Pdown, flux_exp, flux_ref, phi_eff in all_results[
                mode
            ]:
                f.write(
                    f"{mode},{T_C},{run_id},{Pup:.8e},{Pdown:.8e},"
                    f"{flux_exp:.8e},{flux_ref:.8e},{phi_eff:.8e}\n"
                )
    print(f"Saved: {csv_path}")

    # Arrhenius fits per mode and run
    fit_lines = []
    for mode in modes:
        for run_id in ["Run 1", "Run 2"]:
            T_list = [r[0] for r in all_results[mode] if r[1] == run_id]
            phi_list = [r[6] for r in all_results[mode] if r[1] == run_id]
            phi_0, E_phi = fit_arrhenius(T_list, phi_list)
            fit_lines.append((mode, run_id, phi_0, E_phi))

    print("\n=== Permeability Arrhenius fits (separate per mode and run) ===")
    print("Form:  phi(T) = phi_0 * exp( -E_phi / (kB*T) )   with E_phi in kJ/mol")
    for mode, run_id, phi_0, E_phi in fit_lines:
        print(
            f"{mode:5s} | {run_id}:  phi(T) = {phi_0:.3e} * exp( -{E_phi:.3f} kJ/mol / (kB*T) )"
        )

    txt_path = os.path.join(OUTDIR, "dry_run_phi_arrhenius_fits.txt")
    with open(txt_path, "w") as f:
        f.write("phi(T) = phi_0 * exp( -E_phi / (kB*T) )  (E_phi reported in kJ/mol)\n")
        for mode, run_id, phi_0, E_phi in fit_lines:
            f.write(f"{mode} | {run_id}: phi_0={phi_0:.10e}, E_phi_kJmol={E_phi:.6f}\n")
    print(f"Saved: {txt_path}")

    # Plot phi_eff vs 1/T
    markers = {
        ("flux0", "Run 1"): "o",
        ("flux0", "Run 2"): "s",
        ("conc0", "Run 1"): "^",
        ("conc0", "Run 2"): "D",
    }

    fig, ax = plt.subplots(figsize=(7, 4))
    for mode in modes:
        for run_id in ["Run 1", "Run 2"]:
            T_arr = np.array([r[0] for r in all_results[mode] if r[1] == run_id], float)
            phi_arr = np.array(
                [r[6] for r in all_results[mode] if r[1] == run_id], float
            )
            ax.scatter(
                1.0 / (T_arr + 273.15),
                phi_arr,
                marker=markers[(mode, run_id)],
                label=f"{mode} {run_id}",
            )

    ax.set_yscale("log")
    ax.set_xlabel("1/T [1/K]")
    ax.set_ylabel(r"Restored $\Phi_\mathrm{eff}$")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fpath = os.path.join(OUTDIR, "dry_run_phi_vs_invT.png")
    fig.savefig(fpath, dpi=300)
    plt.close(fig)
    print(f"Saved: {fpath}")
