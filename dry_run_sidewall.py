"""Side-wall bypass in the dry (empty) cell.

The salt-filled figures (`sidewall_loss.pdf`, `sidewall_contribution.pdf`) report
how much of the flux avoids the primary path.  This is the same breakdown for the
dry run, where there is no salt at all, so what remains is the bypass of the
nickel envelope on its own -- the geometric leak the referee asks us to isolate.

Definitions mirror comparison.py exactly, with the membrane taking the role that
the salt free surface plays in the filled cell:

    sidewall loss          100 * (1 - J[mem_Ni_bottom] / J_in )
    sidewall contribution  100 * (1 - J[mem_Ni_top]    / J_out)

Because the empty cell is a single material with Dirichlet boundaries, every
surface flux scales with the nickel permeability and it cancels in these ratios:
the numbers depend only on the geometry, the pressure pair and the outer-wall
condition, not on temperature through Phi_Ni.  Any temperature trend seen here
therefore comes from the measured downstream back-pressure alone -- which is the
contrast worth drawing against the filled cell, where the trend is driven by the
FLiBe/Ni permeability ratio.

Outputs:
    results/dry_run_sidewall_metrics.csv
    results/dry_run_sidewall_loss.pdf/.png
    results/dry_run_sidewall_contribution.pdf/.png
"""

from pathlib import Path
import csv

import festim as F
import h_transport_materials as htm
import matplotlib.pyplot as plt
import numpy as np
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

from cylindrical_flux import CylindricalFlux
from exp_data import dry_run, load_ni_permeability

import morethemes as mt
mt.set_theme("lumen")

OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

N_A = 6.02214076e23
KJ_MOL_TO_EV = 1.0 / 96.485

_read = gmshio.read_from_msh("mesh_solid_only.msh", MPI.COMM_WORLD, 0)
mesh, cell_tags, facet_tags = _read.mesh, _read.cell_tags, _read.facet_tags

D_solid = htm.diffusivities.filter(material="nickel").filter(isotope="h")[-1]

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

UPSTREAM = [mem_Ni_bottom, bottom_cap_Ni, bottom_sidewall_Ni]
DOWNSTREAM = [top_cap_Ni, top_sidewall_Ni, mem_Ni_top]
ALL_SURFACES = [out_surf, left_bc_top_Ni, left_bc_middle_Ni, left_bc_bottom_Ni,
                top_cap_Ni, top_sidewall_Ni, bottom_sidewall_Ni,
                mem_Ni_top, mem_Ni_bottom, bottom_cap_Ni]
LABEL = {id(mem_Ni_bottom): "mem_Ni_bottom", id(bottom_cap_Ni): "bottom_cap_Ni",
         id(bottom_sidewall_Ni): "bottom_sidewall_Ni", id(top_cap_Ni): "top_cap_Ni",
         id(top_sidewall_Ni): "top_sidewall_Ni", id(mem_Ni_top): "mem_Ni_top"}

MODES = {"flux0": "Ideal coating", "conc0": "Uncoated"}
MODE_MARKER = {"flux0": "s", "conc0": "^"}
RUN_COLOR = {"Run 1": "red", "Run 2": "black"}


def _material(phi_0_particles, E_phi_eV):
    K_S = htm.Solubility(S_0=phi_0_particles / D_solid.pre_exp.magnitude,
                         E_S=E_phi_eV - D_solid.act_energy.magnitude, law="sievert")
    return F.Material(D_0=D_solid.pre_exp.magnitude,
                      E_D=D_solid.act_energy.magnitude,
                      K_S_0=K_S.pre_exp.magnitude, E_K_S=K_S.act_energy.magnitude,
                      solubility_law="sievert")


def run_one(T_K, P_up, P_down, mode, mat):
    solid = F.VolumeSubdomain(id=2, material=mat)
    m = F.HydrogenTransportProblemDiscontinuous()
    m.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    m.facet_meshtags, m.volume_meshtags = facet_tags, cell_tags
    m.subdomains = [solid] + ALL_SURFACES
    m.surface_to_volume = {s: solid for s in ALL_SURFACES}
    H = F.Species("H", subdomains=m.volume_subdomains)
    m.species = [H]
    m.temperature = float(T_K)

    def sieverts(surfaces, pressure):
        return [F.SievertsBC(subdomain=s, species=H, pressure=float(pressure),
                             S_0=float(mat.K_S_0), E_S=float(mat.E_K_S))
                for s in surfaces]

    outer = (F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)
             if mode == "flux0"
             else F.FixedConcentrationBC(subdomain=out_surf, species=H, value=0.0))
    m.boundary_conditions = (sieverts(UPSTREAM, P_up) + [outer]
                             + sieverts(DOWNSTREAM, P_down))
    m.settings = F.Settings(atol=1e-10, rtol=1e-10, transient=False)

    fluxes = {LABEL[id(s)]: CylindricalFlux(field=H, surface=s)
              for s in UPSTREAM + DOWNSTREAM}
    gb = CylindricalFlux(field=H, surface=out_surf)
    m.exports = list(fluxes.values()) + [gb]
    m.initialise()
    m.run()
    vals = {k: float(v.value) for k, v in fluxes.items()}
    vals["glovebox"] = float(gb.value)
    return vals


def main():
    ni = load_ni_permeability()
    mode_to_bc = {"flux0": "particle_flux_zero", "conc0": "sieverts"}
    rows = []
    for mode in MODES:
        prm = ni[mode_to_bc[mode]]
        mat = _material(prm["phi_0"] * N_A, prm["E_phi_kJmol"] * KJ_MOL_TO_EV)
        for T_C, run, P_up, P_down, J_exp, _err in dry_run:
            v = run_one(T_C + 273.15, P_up, P_down, mode, mat)
            J_in = abs(sum(v[LABEL[id(s)]] for s in UPSTREAM))
            J_out = abs(sum(v[LABEL[id(s)]] for s in DOWNSTREAM))
            leak = 100.0 * (1 - abs(v["mem_Ni_bottom"]) / J_in) if J_in > 0 else np.nan
            comp = 100.0 * (1 - abs(v["mem_Ni_top"]) / J_out) if J_out > 0 else np.nan
            rows.append(dict(mode=mode, case=MODES[mode], run=run, T_C=T_C,
                             T_K=T_C + 273.15, P_up=P_up, P_down=P_down,
                             J_in=J_in, J_out=J_out, J_exp=J_exp,
                             pct_sidewall_leak=leak, pct_sidewall_comp=comp,
                             **{k: v[k] for k in sorted(v)}))
            print(f"  {MODES[mode]:>14} {run} {T_C:.0f}C: "
                  f"leak={leak:6.2f}%  comp={comp:6.2f}%", flush=True)

    path = OUTDIR / "dry_run_sidewall_metrics.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"[saved] {path}")
    # plot() and plot_per_case() are defined below main(); Python resolves them
    # at call time, so the ordering is only a readability choice.
    plot(rows)
    plot_per_case(rows)
    plot_combined(rows)


# Paper literature palette (plot_perm_fits.py); marker still carries the
# boundary condition, as in the salt-filled sidewall figures.
MODE_COLOR = {"flux0": "#4C72B0", "conc0": "#DD8452"}


def plot(rows, runs=("Run 1",)):
    """Re-plot from the stored metrics; `runs` selects which pressure levels show."""
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15,
                         "axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 15})

    def make(metric, ylabel, outfile):
        fig, ax = plt.subplots(figsize=(8, 6.2))
        for run in runs:
            for mode in MODES:
                sub = sorted([r for r in rows if r["run"] == run and r["mode"] == mode],
                             key=lambda r: r["T_K"])
                if not sub:
                    continue
                lab = MODES[mode] if len(runs) == 1 else \
                    f"{sub[0]['P_up'] / 1e5:.2f} bar - {MODES[mode]}"
                ax.plot([r["T_K"] for r in sub], [r[metric] for r in sub],
                        color=MODE_COLOR[mode], marker=MODE_MARKER[mode],
                        linestyle="-", linewidth=1.8, markersize=10,
                        markerfacecolor="none", markeredgewidth=1.8, label=lab)
        ax.set_xlabel("Temperature [K]")
        ax.set_xlim(768, 978)
        ax.set_xticks([773, 823, 873, 923, 973])
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
                  ncol=len(runs) * 2, frameon=True)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(f"{outfile}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {outfile}.pdf (+ .png)")

    make("pct_sidewall_leak", "Upstream sidewall flux ratio [%]",
         OUTDIR / "dry_run_sidewall_loss")
    make("pct_sidewall_comp", "Net sidewall contribution\nto downstream flux [%]",
         OUTDIR / "dry_run_sidewall_contribution")


def plot_per_case(rows, run="Run 1"):
    """One figure per boundary condition, so each metric gets its own scale.

    On a shared 0-70 % axis the coated case is squashed against the bottom and its
    flatness -- the point of the figure -- is invisible.
    """
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15,
                         "axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 15})

    NOTE = {"flux0": "constant to $3\\times10^{-13}$",
            "conc0": "varies with $\\sqrt{P_\\mathrm{down}/P_\\mathrm{up}}$"}

    for mode in MODES:
        sub = sorted([r for r in rows if r["run"] == run and r["mode"] == mode],
                     key=lambda r: r["T_K"])
        if not sub:
            continue
        color = MODE_COLOR[mode]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.3))
        for ax, key, title in (
                (axes[0], "pct_sidewall_leak", "(a)  Upstream side"),
                (axes[1], "pct_sidewall_comp", "(b)  Downstream side")):
            v = [r[key] for r in sub]
            ax.plot([r["T_K"] for r in sub], v, color=color,
                    marker=MODE_MARKER[mode], linestyle="-", linewidth=1.8,
                    markersize=10, markerfacecolor="none", markeredgewidth=1.8)
            ax.set_xlabel("Temperature [K]")
            ax.set_xlim(768, 978)
            ax.set_xticks([773, 823, 873, 923, 973])
            ax.set_title(title)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            lo, hi = min(v), max(v)
            pad = max(0.08 * (hi - lo), 0.25)          # keep a flat line off the frame
            ax.set_ylim(lo - pad, hi + pad)
            ax.text(0.5, 0.06, NOTE[mode], transform=ax.transAxes, ha="center",
                    fontsize=12, color="0.35")
        # axis names inherited from the salt-filled figures
        axes[0].set_ylabel("Upstream sidewall flux ratio [%]")
        axes[1].set_ylabel("Net sidewall contribution\nto downstream flux [%]")

        h = Line2D([0], [0], color=color, marker=MODE_MARKER[mode], lw=1.8,
                   ms=10, markerfacecolor="none", markeredgewidth=1.8,
                   label=MODES[mode])
        fig.legend(handles=[h], loc="upper center", bbox_to_anchor=(0.5, 1.005),
                   frameon=True, borderpad=0.5)
        fig.tight_layout(rect=[0, 0, 1, 0.88])
        stem = OUTDIR / f"dry_run_sidewall_{'ideal' if mode == 'flux0' else 'uncoated'}"
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {stem}.pdf (+ .png)")


MODE_LS = {"flux0": "-", "conc0": "-"}


def plot_combined(rows, run="Run 1"):
    """Dry-run reference for the two pathway metrics used for the filled cell.

    One panel per metric, matching the salt-filled figures, with both outer-wall
    conditions overlaid.  The contrast between them is the message, so the two
    share an axis rather than being magnified onto separate ones.
    """
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 15,
                         "axes.labelsize": 15, "xtick.labelsize": 15,
                         "ytick.labelsize": 15, "legend.fontsize": 15})

    PANELS = (("pct_sidewall_leak", "Upstream sidewall flux ratio [%]",
               "(a)  Upstream side"),
              ("pct_sidewall_comp", "Net sidewall contribution to downstream flux [%]",
               "(b)  Downstream side"))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    for ax, (key, ylab, title) in zip(axes, PANELS):
        for mode in MODES:
            sub = sorted([r for r in rows if r["run"] == run and r["mode"] == mode],
                         key=lambda r: r["T_K"])
            ax.plot([r["T_K"] for r in sub], [r[key] for r in sub],
                    color=MODE_COLOR[mode], marker=MODE_MARKER[mode],
                    linestyle=MODE_LS[mode], linewidth=1.8, markersize=10,
                    markerfacecolor="none", markeredgewidth=1.8)
        ax.set_xlabel("Temperature [K]")
        ax.set_xlim(768, 978)
        ax.set_xticks([773, 823, 873, 923, 973])
        ax.set_ylabel(ylab, fontsize=13)
        ax.set_title(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.margins(y=0.12)

    handles = [Line2D([0], [0], color=MODE_COLOR[m], marker=MODE_MARKER[m],
                      linestyle=MODE_LS[m], lw=1.8, ms=10,
                      markerfacecolor="none", markeredgewidth=1.8,
                      label=MODES[m]) for m in MODES]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=2, frameon=True, columnspacing=2.0, borderpad=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    stem = OUTDIR / "dry_run_sidewall_reference"
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {stem}.pdf (+ .png)")


if __name__ == "__main__":
    main()
