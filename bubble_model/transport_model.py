"""Axisymmetric (cylindrical) permeation model on the real cell geometry
from ../mesh.py, with the bubble represented as a gas pocket at the
FLiBe/Ni interface.

Physical picture
-----------------
A hydrogen bubble is a gas pocket trapped at the interface. Its effect on
the flux is geometric: it blocks direct H exchange over the area it
covers, while transport through the gas phase itself is negligible (gas
density is ~1000x smaller than the dissolved-H concentrations on either
side). H2 from FLiBe must instead go around the bubble, laterally through
the open annular part of the interface.

Boundary conditions
--------------------
- liquid_gas_surface / solid_gas_surface (FLiBe/Ni side of the bubble):
  no flux, since the gas pocket blocks H atoms from crossing there.
- liquid_Ni_interface (the open annular ring r > r_b, plus the vertical
  side at r = X_IN): normal Henry/Sievert penalty equilibrium.

alpha is realized as a static mesh parameter (bubble radius r_b, with
alpha = (r_b/X_IN)^2) fixed for one run; a time-dependent coverage history
is built downstream by combining runs at different alpha.

Physical parameters
--------------------
- FLiBe diffusivity:   htm.Diffusivity(D_0=2.5e-7 m²/s, E_D=0.24 eV)
                       (Calderoni-style FLiBe, literature)
- FLiBe permeability:  htm pre-exp 4.16e13 atoms/m/s/Pa, E=0.466 eV
                       (Calderoni FLiBe permeability)
- Ni permeability:     dry-run fit (independent of this experiment),
                       phi_0 = 9.03e-7 mol/m/s/Pa^0.5, E = 57.6 kJ/mol
- α_max ≈ 0.26 from r_b = 0.020 m on a disk of radius X_IN = 0.039 m
- Initial condition:   C(x,y,0) = 0 (cell empty at t=0)
- Upstream:           H2 at P_up = 1.32e5 Pa, applied as Henry at the
                       liquid_surface and Sievert at top_cap_Ni at t > 0
- Downstream:          C = 0 at mid_membrane_Ni and bottom_cap_Ni (sweep
                       gas treated as pure carrier, P_H2_down ≈ 0)
"""
from __future__ import annotations
import gc
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import h_transport_materials as htm
import festim as F
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.io import gmsh as gmshio

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from cylindrical_flux import CylindricalFlux  # noqa: E402
from exp_data import D_nickel, load_ni_permeability  # noqa: E402

from generate_mesh import generate_axisym_bubble_mesh, X_IN, Y_mT  # noqa: E402

# Cache: mesh-file-name -> (mesh, cell_tags, facet_tags).  FESTIM creates new
# MPI communicators every time the mesh is re-read; caching reads avoids
# leaking communicators across fit iterations.
_MESH_CACHE: dict[str, tuple] = {}


def _dispose_model(m):
    """Release FESTIM/dolfinx/PETSc resources held by a finished model."""
    if m is None:
        return
    try:
        for e in getattr(m, "exports", []) or []:
            for attr in ("data", "field", "surface"):
                if hasattr(e, attr):
                    setattr(e, attr, None)
        for attr in ("exports", "interfaces", "subdomains",
                     "boundary_conditions"):
            setattr(m, attr, [])
        for attr in ("_forms", "_function_spaces", "_solvers", "_timers"):
            if hasattr(m, attr):
                setattr(m, attr, None)
        m.mesh = None
    except Exception:
        pass
    gc.collect()
    try:
        PETSc.garbage_cleanup()
    except Exception:
        pass
    try:
        MPI.COMM_WORLD.barrier()
    except Exception:
        pass

KB_EV = 8.617333262e-5
N_A = 6.02214076e23
EV_PER_KJMOL = 1.0 / 96.485332

# ── Physical material parameters (NOT fit to this experiment) ────────────────
# FLiBe diffusivity (Calderoni / htm default)
D_FLIBE_0 = 2.5e-7
E_D_FLIBE = 0.24

# FLiBe permeability (Calderoni-style, from htm) — H atoms per m/s/Pa
PHI_F0_ATOMS = 41587400565660.95
E_PHIF = 0.4655730255084721

# Ni permeability (from the user's dry-run Arrhenius fit, an
# independent experiment that doesn't include the FLiBe layer)
def _ni_perm():
    ni = load_ni_permeability(
        REPO_ROOT / "results" / "dry_run_phi_arrhenius_fits.txt"
    )
    prm = ni["sieverts"]
    return prm["phi_0"] * N_A, prm["E_phi_kJmol"] * EV_PER_KJMOL

PHI_NI_0_ATOMS, E_PHINI = _ni_perm()

# FLiBe top surface y-coordinate (thermal expansion) at each T
Y_FT_BY_TC = {500.0: 0.02914, 550.0: 0.02919, 600.0: 0.02925,
              650.0: 0.02930, 700.0: 0.02936}


def _mag(x):
    return x.magnitude if hasattr(x, "magnitude") else float(x)


# ─────────────────────────────────────────────────────────────────────────────
# Build the FESTIM model.  alpha is realised as a geometric mesh parameter:
#   alpha = (r_b / X_IN)^2   (since both disks share the same azimuth)
# We generate one mesh per chosen alpha and run a no-bubble-dynamics transient.
# ─────────────────────────────────────────────────────────────────────────────
def alpha_to_rb(alpha: float) -> float:
    """Inverse of  alpha = (r_b / X_IN)^2  ."""
    return math.sqrt(max(alpha, 0.0)) * X_IN


def build_model(T_C: float, P_up: float, alpha: float, t_b: float = 5e-4,
                mesh_size: float = 6e-4):
    """Build a transient FESTIM 2.x model on the axisymmetric mesh.

    The bubble is *static* in the mesh; its size sets the coverage `alpha`.
    The bubble surfaces (liquid_gas, solid_gas) get NO BC -> no-flux Neumann,
    which is the correct boundary for a gas pocket that blocks H atoms.
    """
    T_K = T_C + 273.15
    y_ft = Y_FT_BY_TC[T_C]

    if alpha <= 0:
        # Degenerate: no bubble.  Use a vanishingly thin bubble (alpha=1e-4)
        # so the mesh remains valid; effect is negligible.
        alpha = 1e-4
    r_b = alpha_to_rb(alpha)

    fname = (f"axisym_T{int(T_C)}_a{alpha:.3f}_tb{int(t_b*1e6)}"
             f"_ms{int(mesh_size*1e6)}.msh")
    if not Path(fname).exists():
        generate_axisym_bubble_mesh(y_ft=y_ft, r_b=r_b, t_b=t_b,
                                    mesh_size=mesh_size, fname=fname)

    # Cache mesh reads to avoid leaking MPI communicators across calls.
    if fname not in _MESH_CACHE:
        _read = gmshio.read_from_msh(fname, MPI.COMM_WORLD, rank=0, gdim=2)
        _MESH_CACHE[fname] = (_read.mesh, _read.cell_tags, _read.facet_tags)
    mesh, cell_tags, facet_tags = _MESH_CACHE[fname]

    # Materials
    D_flibe = htm.Diffusivity(D_0=D_FLIBE_0, E_D=E_D_FLIBE)
    perm_flibe = htm.Permeability(pre_exp=PHI_F0_ATOMS, act_energy=E_PHIF,
                                  law="henry")
    perm_Ni = htm.Permeability(pre_exp=PHI_NI_0_ATOMS, act_energy=E_PHINI,
                               law="sievert")

    K_S_liquid = htm.Solubility(
        S_0=perm_flibe.pre_exp / D_flibe.pre_exp,
        E_S=perm_flibe.act_energy - D_flibe.act_energy,
        law="henry",
    )
    K_S_nickel = htm.Solubility(
        S_0=perm_Ni.pre_exp / D_nickel.pre_exp,
        E_S=perm_Ni.act_energy - D_nickel.act_energy,
        law="sievert",
    )

    mat_liquid = F.Material(
        D_0=_mag(D_flibe.pre_exp), E_D=_mag(D_flibe.act_energy),
        K_S_0=_mag(K_S_liquid.pre_exp), E_K_S=_mag(K_S_liquid.act_energy),
        solubility_law="henry",
    )
    mat_solid = F.Material(
        D_0=_mag(D_nickel.pre_exp), E_D=_mag(D_nickel.act_energy),
        K_S_0=_mag(K_S_nickel.pre_exp), E_K_S=_mag(K_S_nickel.act_energy),
        solubility_law="sievert",
    )

    K_S_0_Ni = _mag(K_S_nickel.pre_exp)
    E_S_Ni = _mag(K_S_nickel.act_energy)
    H_0_liq = _mag(K_S_liquid.pre_exp)
    E_H_liq = _mag(K_S_liquid.act_energy)

    # Subdomains
    fluid = F.VolumeSubdomain(id=1, material=mat_liquid)
    solid = F.VolumeSubdomain(id=2, material=mat_solid)
    out_surf = F.SurfaceSubdomain(id=3)
    top_cap = F.SurfaceSubdomain(id=5)
    liquid_surface = F.SurfaceSubdomain(id=8)
    mid_membrane = F.SurfaceSubdomain(id=9)
    bottom_cap = F.SurfaceSubdomain(id=10)
    # NB: liquid_gas (tag 35) and solid_gas (tag 36) are NOT added to
    # subdomains.  Without an associated SurfaceSubdomain and BC, FESTIM
    # treats them as no-flux boundaries — exactly the bubble physics we want.
    all_surfs = [out_surf, top_cap, liquid_surface, mid_membrane, bottom_cap]

    model = F.HydrogenTransportProblemDiscontinuous()
    model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    model.facet_meshtags = facet_tags
    model.volume_meshtags = cell_tags
    model.subdomains = [solid, fluid] + all_surfs
    model.method_interface = "penalty"
    model.interfaces = [
        F.Interface(id=99, subdomains=[solid, fluid], penalty_term=1e22)
    ]
    model.surface_to_volume = {
        out_surf: solid, top_cap: solid,
        liquid_surface: fluid, mid_membrane: solid, bottom_cap: solid,
    }

    H = F.Species("H", subdomains=model.volume_subdomains)
    model.species = [H]
    model.temperature = T_K

    # BCs
    bcs = [
        F.SievertsBC(subdomain=top_cap, species=H, pressure=P_up,
                     S_0=K_S_0_Ni, E_S=E_S_Ni),
        F.HenrysBC(subdomain=liquid_surface, species=H, pressure=P_up,
                   H_0=H_0_liq, E_H=E_H_liq),
        # Downstream: C = 0 (sweep gas treated as pure carrier, P_H2_down ≈ 0).
        # We use FixedConcentrationBC rather than SievertsBC(P=0) because UFL
        # collapses a zero-pressure Sievert expression and loses mesh info.
        F.FixedConcentrationBC(subdomain=mid_membrane, species=H, value=0.0),
        F.FixedConcentrationBC(subdomain=bottom_cap,   species=H, value=0.0),
    ]
    model.boundary_conditions = bcs
    model.settings = F.Settings(atol=1e12, rtol=1e-12, transient=True)

    flux_mid = CylindricalFlux(field=H, surface=mid_membrane)
    flux_botcap = CylindricalFlux(field=H, surface=bottom_cap)
    model.exports = [flux_mid, flux_botcap]

    A_mid = math.pi * X_IN ** 2

    return dict(
        model=model, T_K=T_K, A_mid=A_mid, alpha=alpha,
        flux_mid=flux_mid, flux_botcap=flux_botcap,
    )


def run_axisym(T_C: float, P_up: float = 1.32e5, alpha: float = 0.0,
               t_end_s: float = 4 * 3600, dt_value: float = 120.0,
               mesh_size: float = 6e-4):
    """Run the FESTIM transient for one (T, alpha) combination."""
    state = build_model(T_C=T_C, P_up=P_up, alpha=alpha, mesh_size=mesh_size)
    m = state["model"]
    m.settings.final_time = t_end_s
    # Conservative time stepping: start small (BC step at t=0 is stiff) and
    # grow gently.  Cutback aggressively on non-convergence.
    m.settings.stepsize = F.Stepsize(
        initial_value=5.0, growth_factor=1.1,
        cutback_factor=0.5, target_nb_iterations=4,
    )
    m.initialise()
    t0 = time.time()
    m.run()

    t = np.asarray(state["flux_mid"].t, dtype=float).copy()
    J_mid = np.asarray(state["flux_mid"].data, dtype=float).copy()
    J_botcap = np.asarray(state["flux_botcap"].data, dtype=float).copy()
    # CylindricalFlux returns the integral of -D·grad(C)·n̂·r over the
    # surface, where n̂ is the FacetNormal on the Ni submesh.  The downstream
    # surfaces (mid_membrane, bottom_cap) act as sinks; for the physical
    # convention "flux leaving the cell is positive" we take the absolute
    # value of the integral.  (FESTIM submesh facet normals do not always
    # align with the geometric outward normal — see GitHub issues on
    # discontinuous mesh orientation.)
    J_total_atoms = np.abs(J_mid + J_botcap)
    J_total_molH2_s = J_total_atoms / (2.0 * N_A)
    A_mid = state["A_mid"]
    J_total_per_A = J_total_molH2_s / A_mid

    out = dict(
        t=t, J_total_per_A=J_total_per_A, J_mid_atoms=J_mid,
        J_botcap_atoms=J_botcap, T_K=state["T_K"], alpha=state["alpha"],
        time_s=time.time() - t0,
    )
    # release model resources so MPI communicators don't leak in long fit loops
    _dispose_model(m)
    return out


