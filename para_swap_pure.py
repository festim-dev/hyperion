"""
Pointwise inversion of FLiBe permeability (phi_flibe) from SWAP experimental flux data.

For each experimental point, a bisection search finds the phi_flibe value that makes
the simulated downstream flux match the measured flux. Uncertainty in phi_flibe is
propagated from the experimental flux uncertainty via a finite-difference derivative.

An Arrhenius fit is applied to the recovered phi_flibe(T) points per boundary condition
mode and run, with weighted least squares using the propagated 1-sigma uncertainties.

Both swap_infinite and swap_transparent cases are inverted and overlaid on the same plot.
Each case uses its own Ni solubility derived from the corresponding entry in
results/dry_run_phi_arrhenius_fits.txt (produced by dry_run_fitting.py).

Outputs (saved to results/):
    dual_pointwise_lnphi_invT.png  -- Arrhenius plot with both cases and fit bands
    inverted_points.csv            -- phi_eff per experimental point
    fitted_params.csv              -- Arrhenius fit parameters (phi_0, E, R2)
    logs/fit_summary.txt           -- per-point fit diagnostics
"""

import gc
import math
import csv
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.log import LogLevel, set_log_level
from dolfinx.io import gmsh as gmshio
import festim as F
import h_transport_materials as htm

from cylindrical_flux import CylindricalFlux
from mesh import generate_mesh, set_y_ft
from exp_data import (
    swap_infinite,
    swap_transparent,
    swap_flux_err,
    load_ni_permeability,
    D_nickel,
)

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# ── Constants ─────────────────────────────────────────────────────────────────

kB_eV = 8.617333262e-5  # Boltzmann constant [eV/K]
NA = 6.02214076e23  # Avogadro constant [mol^-1]

OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

_RANK0 = MPI.COMM_WORLD.rank == 0
_mesh_cache: dict[str, tuple] = {}

# ── Helpers ───────────────────────────────────────────────────────────────────


def _ensure_and_save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if _RANK0:
        print(f"[saved] {out_path}")
    plt.close(fig)


def kJmol_to_eV(E_kJmol: float) -> float:
    return float(E_kJmol) / 96.485332123


def mol_to_particles(x: float) -> float:
    return x * NA


# ── Model disposal ────────────────────────────────────────────────────────────


def _dispose_model(m) -> None:
    if m is None:
        return
    try:
        for e in getattr(m, "exports", []) or []:
            for attr in ("data", "field", "surface"):
                if hasattr(e, attr):
                    setattr(e, attr, None)
        for attr in ("exports", "interfaces", "subdomains", "boundary_conditions"):
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


# ── Mesh I/O ──────────────────────────────────────────────────────────────────


def _mesh_key_from_yft(y_ft: float) -> str:
    return f"mesh_{float(y_ft):.5f}.msh"


def load_or_make_mesh(mesh_file: str, mesh_size: float, model_rank: int = 0):
    if not Path(mesh_file).exists():
        if _RANK0:
            print(f"[mesh] creating '{mesh_file}' (size={mesh_size:g})")
        generate_mesh(mesh_size=mesh_size, fname=mesh_file)

    if mesh_file in _mesh_cache:
        return _mesh_cache[mesh_file]

    _read = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, model_rank)
    _mesh_cache[mesh_file] = (_read.mesh, _read.cell_tags, _read.facet_tags)
    return _mesh_cache[mesh_file]


# ── Ni solubility from permeability ──────────────────────────────────────────


def _ni_solubility_for_bc(bc_type: str) -> htm.Solubility:
    """
    Load Ni permeability from dry_run_phi_arrhenius_fits.txt and derive
    solubility via S(T) = phi(T) / D(T), using the shared D_nickel from exp_data.

    bc_type: 'particle_flux_zero' (ideal coating) or 'sieverts' (uncoated).
    """
    ni_perm = load_ni_permeability()
    prm = ni_perm[bc_type]
    phi_0_particles = mol_to_particles(prm["phi_0"])
    E_phi_eV = kJmol_to_eV(prm["E_phi_kJmol"])

    perm = htm.Permeability(pre_exp=phi_0_particles, act_energy=E_phi_eV, law="sievert")
    return htm.Solubility(
        S_0=perm.pre_exp / D_nickel.pre_exp,
        E_S=perm.act_energy - D_nickel.act_energy,
        law="sievert",
    )


# ── Materials ─────────────────────────────────────────────────────────────────


def make_materials(D_solid, D_liquid, K_solid, permeability_liquid):
    K_S_liquid = htm.Solubility(
        S_0=permeability_liquid.pre_exp / D_liquid.pre_exp,
        E_S=permeability_liquid.act_energy - D_liquid.act_energy,
        law=permeability_liquid.law,
    )
    mat_solid = F.Material(
        D_0=D_solid.pre_exp.magnitude,
        E_D=D_solid.act_energy.magnitude,
        K_S_0=K_solid.pre_exp.magnitude,
        E_K_S=K_solid.act_energy.magnitude,
        solubility_law="sievert",
    )
    mat_liquid = F.Material(
        D_0=D_liquid.pre_exp.magnitude,
        E_D=D_liquid.act_energy.magnitude,
        K_S_0=K_S_liquid.pre_exp.magnitude,
        E_K_S=K_S_liquid.act_energy.magnitude,
        solubility_law="henry",
    )
    return mat_solid, mat_liquid


# ── Model builder ─────────────────────────────────────────────────────────────


def make_model(
    D_flibe: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    K_S_nickel: htm.Solubility,
    temperature: float,
    P_up: float,
    P_down: float = 5.0,
    mesh_size: float = 2e-4,
    penalty_term: float = 1e22,
    out_bc: dict | None = None,
    y_ft: float | None = None,
) -> Tuple[F.HydrogenTransportProblemDiscontinuous, dict]:
    if y_ft is None:
        raise ValueError("y_ft must be provided")

    y_ft_5 = float(f"{float(y_ft):.5f}")
    set_y_ft(y_ft_5)

    mesh, cell_tags, facet_tags = load_or_make_mesh(
        _mesh_key_from_yft(y_ft_5), mesh_size
    )
    mat_solid, mat_liquid = make_materials(
        D_nickel, D_flibe, K_S_nickel, permeability_flibe
    )

    K_S_0_Ni = mat_solid.K_S_0
    E_S_Ni = mat_solid.E_K_S
    H_0_liq = mat_liquid.K_S_0
    E_H_liq = mat_liquid.E_K_S

    fluid_volume = F.VolumeSubdomain(id=1, material=mat_liquid)
    solid_volume = F.VolumeSubdomain(id=2, material=mat_solid)

    out_surf = F.SurfaceSubdomain(id=3)
    left_bc_liquid = F.SurfaceSubdomain(id=41)
    left_bc_top_Ni = F.SurfaceSubdomain(id=42)
    left_bc_middle_Ni = F.SurfaceSubdomain(id=43)
    left_bc_bottom_Ni = F.SurfaceSubdomain(id=44)
    top_cap_Ni = F.SurfaceSubdomain(id=5)
    top_sidewall_Ni = F.SurfaceSubdomain(id=6)
    bottom_sidewall_Ni = F.SurfaceSubdomain(id=7)
    liquid_surface = F.SurfaceSubdomain(id=8)
    mid_membrane_Ni = F.SurfaceSubdomain(id=9)
    bottom_cap_Ni = F.SurfaceSubdomain(id=10)
    liquid_solid_interface = F.SurfaceSubdomain(id=99)

    all_surface_subdomains = [
        out_surf,
        left_bc_liquid,
        left_bc_top_Ni,
        left_bc_middle_Ni,
        left_bc_bottom_Ni,
        top_cap_Ni,
        top_sidewall_Ni,
        bottom_sidewall_Ni,
        liquid_surface,
        mid_membrane_Ni,
        bottom_cap_Ni,
        liquid_solid_interface,
    ]

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags
    my_model.subdomains = [solid_volume, fluid_volume] + all_surface_subdomains
    my_model.method_interface = "penalty"
    my_model.interfaces = [
        F.Interface(
            id=99, subdomains=[solid_volume, fluid_volume], penalty_term=penalty_term
        )
    ]
    my_model.surface_to_volume = {
        out_surf: solid_volume,
        left_bc_liquid: fluid_volume,
        left_bc_top_Ni: solid_volume,
        left_bc_middle_Ni: solid_volume,
        left_bc_bottom_Ni: solid_volume,
        top_cap_Ni: solid_volume,
        top_sidewall_Ni: solid_volume,
        bottom_sidewall_Ni: solid_volume,
        liquid_surface: fluid_volume,
        mid_membrane_Ni: solid_volume,
        bottom_cap_Ni: solid_volume,
    }

    H = F.Species("H", subdomains=my_model.volume_subdomains)
    my_model.species = [H]
    my_model.temperature = temperature

    downstream_bcs = [
        F.SievertsBC(subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni)
        for s in [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
    ]
    upstream_bcs = [
        F.SievertsBC(subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni)
        for s in [top_cap_Ni, top_sidewall_Ni]
    ] + [
        F.HenrysBC(
            subdomain=liquid_surface, species=H, pressure=P_up, H_0=H_0_liq, E_H=E_H_liq
        )
    ]

    out_bc = out_bc or {"type": "none"}
    t = out_bc.get("type", "none").lower()
    if t == "sieverts":
        outer_bcs = [
            F.SievertsBC(
                subdomain=out_surf,
                species=H,
                pressure=float(out_bc.get("pressure", 0.0)),
                S_0=K_S_0_Ni,
                E_S=E_S_Ni,
            )
        ]
    elif t == "particle_flux_zero":
        outer_bcs = [F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)]
    else:
        outer_bcs = []

    my_model.boundary_conditions = upstream_bcs + downstream_bcs + outer_bcs
    my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)

    flux_top_cap = CylindricalFlux(field=H, surface=top_cap_Ni)
    flux_top_sidewall = CylindricalFlux(field=H, surface=top_sidewall_Ni)
    flux_liquid_surface = CylindricalFlux(field=H, surface=liquid_surface)
    flux_mid_membrane = CylindricalFlux(field=H, surface=mid_membrane_Ni)
    flux_bot_sidewall = CylindricalFlux(field=H, surface=bottom_sidewall_Ni)
    flux_bot_cap = CylindricalFlux(field=H, surface=bottom_cap_Ni)
    flux_glovebox = CylindricalFlux(field=H, surface=out_surf)

    my_model.exports = [
        flux_top_cap,
        flux_top_sidewall,
        flux_bot_sidewall,
        flux_liquid_surface,
        flux_mid_membrane,
        flux_bot_cap,
        flux_glovebox,
    ]

    flux_by_label = {
        "top_cap_Ni": flux_top_cap,
        "top_sidewall_Ni": flux_top_sidewall,
        "liquid_surface": flux_liquid_surface,
        "mid_membrane_Ni": flux_mid_membrane,
        "bottom_sidewall_Ni": flux_bot_sidewall,
        "bottom_cap_Ni": flux_bot_cap,
    }
    fluxes_dict = {
        "flux_by_label": flux_by_label,
        "six_labels": list(flux_by_label.keys()),
        "glovebox_flux": flux_glovebox,
        "up_labels": ["top_cap_Ni", "top_sidewall_Ni", "liquid_surface"],
        "down_labels": ["mid_membrane_Ni", "bottom_cap_Ni", "bottom_sidewall_Ni"],
    }
    return my_model, fluxes_dict


def _get_flux_value(flux_obj) -> float:
    try:
        data = getattr(flux_obj, "data", None)
        if data is not None:
            a = np.asarray(data, dtype=float)
            if a.size > 0:
                return float(a[-1])
        val = getattr(flux_obj, "value", None)
        if val is not None:
            return float(val)
    except Exception:
        pass
    return 0.0


def run_once(
    T_K, P_up, P_down, D_flibe, permeability_flibe, K_S_nickel, out_bc=None, y_ft=None
) -> float:
    my_model, fluxes_dict = make_model(
        temperature=T_K,
        D_flibe=D_flibe,
        permeability_flibe=permeability_flibe,
        K_S_nickel=K_S_nickel,
        P_up=P_up,
        P_down=P_down,
        out_bc=out_bc,
        y_ft=y_ft,
    )
    my_model.initialise()
    my_model.run()

    vals = {
        label: _get_flux_value(fluxes_dict["flux_by_label"][label])
        for label in fluxes_dict["six_labels"]
    }
    total_down = float(np.sum([vals[label] for label in fluxes_dict["down_labels"]]))
    _dispose_model(my_model)
    return total_down


# ── Experimental data helpers ─────────────────────────────────────────────────


@dataclass
class CalibPoint:
    case: str
    T_C: float
    T_K: float
    run: str
    P_up: float
    P_down: float
    P_gb: Optional[float]
    y_ft: float
    J_exp: float


def _collect_points(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    allowed_case_names: Optional[List[str]] = None,
) -> List[CalibPoint]:
    pts: List[CalibPoint] = []
    for case_name, cfg in cases.items():
        if allowed_case_names and case_name not in allowed_case_names:
            continue
        if not case_name.startswith("swap"):
            continue
        for Tc, row in cfg["table"].items():
            y5 = float(
                f"{float(Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])):.5f}"
            )
            for run_name, cond in row.get("runs", {}).items():
                pts.append(
                    CalibPoint(
                        case=case_name,
                        T_C=float(Tc),
                        T_K=float(T2K[Tc]),
                        run=str(run_name),
                        P_up=float(cond["P_up"]),
                        P_down=float(cond["P_down"]),
                        P_gb=float(cond["P_gb"]) if "P_gb" in cond else None,
                        y_ft=y5,
                        J_exp=float(cond["J_exp"]),
                    )
                )
    if not pts:
        raise RuntimeError("No calibration points found.")
    pts.sort(key=lambda p: (p.case, p.T_C, p.run))
    return pts


def get_exp_error(
    case_name: str, temp: float, run_name: str = "Run 1"
) -> Optional[float]:
    """Return 1-sigma flux uncertainty by dividing the stored k=2 value by 2."""
    case = swap_flux_err.get(case_name)
    if case is None:
        return None
    entry = case.get(float(temp))
    if entry is None:
        return None
    runs = entry.get("runs") if isinstance(entry, dict) else None
    raw = runs.get(run_name) if isinstance(runs, dict) else None
    try:
        val = float(raw) / 2.0
    except (TypeError, ValueError):
        return None
    return val if np.isfinite(val) and val > 0.0 else None


# ── Bisection inversion ───────────────────────────────────────────────────────


def _invert_point_child(p_dict, D_flibe, K_S_nickel, q):
    tiny = np.finfo(float).tiny

    class _P:
        __slots__ = (
            "case",
            "T_C",
            "T_K",
            "run",
            "P_up",
            "P_down",
            "P_gb",
            "y_ft",
            "J_exp",
        )

        def __init__(self, **kw):
            [setattr(self, k, kw[k]) for k in self.__slots__]

    p = _P(**p_dict)
    out_bc = (
        {"type": "sieverts", "pressure": p.P_gb}
        if p.P_gb is not None
        else {"type": "particle_flux_zero"}
    )

    def J_of_phi(phi_val: float) -> float:
        perm = htm.Permeability(pre_exp=float(phi_val), act_energy=0.0, law="henry")
        return max(
            float(
                run_once(
                    p.T_K, p.P_up, p.P_down, D_flibe, perm, K_S_nickel, out_bc, p.y_ft
                )
            ),
            tiny,
        )

    phi_lo, phi_hi = 1e10, 1e15
    tol_log, maxit = 3e-3, 18
    log_lo, log_hi = math.log10(phi_lo), math.log10(phi_hi)
    target = max(float(p.J_exp), tiny)

    J_lo, J_hi = J_of_phi(10.0**log_lo), J_of_phi(10.0**log_hi)
    if (J_lo - target) * (J_hi - target) > 0.0:
        if J_lo < target:
            log_hi += 1.0
            J_hi = J_of_phi(10.0**log_hi)
        else:
            log_lo -= 1.0
            J_lo = J_of_phi(10.0**log_lo)

    for _ in range(maxit):
        log_mid = 0.5 * (log_lo + log_hi)
        J_mid = J_of_phi(10.0**log_mid)
        if abs(log_hi - log_lo) < tol_log:
            q.put(10.0**log_mid)
            return
        if (J_lo - target) * (J_mid - target) <= 0.0:
            log_hi, J_hi = log_mid, J_mid
        else:
            log_lo, J_lo = log_mid, J_mid
    q.put(10.0 ** (0.5 * (log_lo + log_hi)))


def _phi_match_exp(p: CalibPoint, D_flibe, K_S_nickel) -> float:
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    p_dict = {s: getattr(p, s) for s in p.__dataclass_fields__}
    proc = ctx.Process(
        target=_invert_point_child,
        args=(p_dict, D_flibe, K_S_nickel, q),
        daemon=False,
    )
    proc.start()
    phi_T = q.get()
    proc.join()
    return float(phi_T)


# ── Inversion with uncertainty propagation ────────────────────────────────────


def _invert_points_with_sigma(
    pts: List[CalibPoint],
    D_flibe: htm.Diffusivity,
    K_S_nickel_by_case: Dict[str, htm.Solubility],
):
    """
    Run pointwise inversion for each CalibPoint, using the K_S_nickel that
    corresponds to each case's outer BC. Propagate flux uncertainty to
    ln(phi) uncertainty via a finite-difference derivative.

    Returns five dicts keyed by (case, run):
        invT_by_case, lnphi_by_case, sig_ln_by_case, meta_by_case, metrics_by_case
    """
    invT_by, lnphi_by, sig_by, meta_by, metrics_by = {}, {}, {}, {}, {}

    for case_name, run_name in sorted({(p.case, p.run) for p in pts}):
        K_S_nickel = K_S_nickel_by_case[case_name]
        invT_list, lnphi_list, sig_list, meta_rows, metrics_rows = [], [], [], [], []

        for p in [pp for pp in pts if pp.case == case_name and pp.run == run_name]:
            phi_T = _phi_match_exp(p, D_flibe, K_S_nickel)
            tiny = np.finfo(float).tiny

            out_bc_p = (
                {"type": "sieverts", "pressure": p.P_gb}
                if p.P_gb is not None
                else {"type": "particle_flux_zero"}
            )

            def J_of_phi(phi_val: float) -> float:
                perm = htm.Permeability(
                    pre_exp=float(phi_val), act_energy=0.0, law="henry"
                )
                return max(
                    float(
                        run_once(
                            p.T_K,
                            p.P_up,
                            p.P_down,
                            D_flibe,
                            perm,
                            K_S_nickel,
                            out_bc_p,
                            p.y_ft,
                        )
                    ),
                    tiny,
                )

            J_fit = J_of_phi(phi_T)
            J_exp = max(float(p.J_exp), tiny)
            err_abs = J_fit - J_exp
            err_rel = err_abs / J_exp

            sigma_J = get_exp_error(p.case, p.T_C, p.run)
            sigma_lnphi = None
            if sigma_J and np.isfinite(sigma_J) and sigma_J > 0.0:
                delta = max(0.02 * float(phi_T), 1e-16)
                try:
                    dJ_dphi = (
                        J_of_phi(phi_T + delta) - J_of_phi(max(phi_T - delta, tiny))
                    ) / (2.0 * delta)
                    val = (sigma_J / max(abs(dJ_dphi), tiny)) / max(float(phi_T), tiny)
                    sigma_lnphi = float(val) if np.isfinite(val) and val > 0.0 else None
                except Exception:
                    sigma_lnphi = None

            invT_list.append(1.0 / p.T_K)
            lnphi_list.append(math.log(max(phi_T, tiny)))
            sig_list.append(np.nan if sigma_lnphi is None else float(sigma_lnphi))
            meta_rows.append(p)
            metrics_rows.append(
                {"J_fit": J_fit, "err_abs": err_abs, "err_rel": err_rel}
            )

            if _RANK0:
                line = (
                    f"[{p.case} — {p.run}] T={p.T_C:.0f}C: "
                    f"J_exp={J_exp:.3e}, J_fit={J_fit:.3e}, rel_err={err_rel * 100:.2f}%"
                )
                log_file = OUTDIR / "logs" / "fit_summary.txt"
                log_file.parent.mkdir(parents=True, exist_ok=True)
                with log_file.open("a") as f:
                    f.write(line + "\n")

        key = (case_name, run_name)
        invT_by[key] = np.array(invT_list, float)
        lnphi_by[key] = np.array(lnphi_list, float)
        sig_by[key] = np.array(sig_list, float)
        meta_by[key] = meta_rows
        metrics_by[key] = metrics_rows

    return invT_by, lnphi_by, sig_by, meta_by, metrics_by


# ── Arrhenius fit ─────────────────────────────────────────────────────────────


def _fit_lnphi(invT, lnphi, sigma_ln=None):
    """Weighted least-squares fit of ln(phi) = a + b/T, returns (a, b, phi_0, E_eV, band_fn)."""
    x = np.asarray(invT, float)
    y = np.asarray(lnphi, float)
    w = (
        1.0 / np.asarray(sigma_ln, float) ** 2
        if sigma_ln is not None
        else np.ones_like(x)
    )
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)

    X = np.column_stack((np.ones_like(x), x))
    XtWX_inv = np.linalg.pinv(X.T @ np.diag(w) @ X)
    beta = XtWX_inv @ (X.T @ (w * y))
    a, b = float(beta[0]), float(beta[1])

    r = y - (a + b * x)
    s2 = float(np.sum(w * r * r)) / max(len(x) - 2, 1)
    cov = s2 * XtWX_inv

    def band(xq, z=1.96):
        xq = np.asarray(xq, float)
        var = cov[0, 0] + 2.0 * cov[0, 1] * xq + cov[1, 1] * xq**2
        std = np.sqrt(np.maximum(var, 0.0))
        return (a + b * xq) - z * std, (a + b * xq) + z * std

    return a, b, float(np.exp(a)), float(-b * kB_eV), band


# ── CSV outputs ───────────────────────────────────────────────────────────────


def _save_inverted_points_csv(rows: list) -> None:
    path = OUTDIR / "inverted_points.csv"
    fields = [
        "case",
        "run",
        "T_C",
        "T_K",
        "invT",
        "ln_phi",
        "sigma_lnphi",
        "phi",
        "sigma_phi",
        "J_exp",
        "sigma_J",
        "J_fit",
        "err_abs",
        "err_rel",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if _RANK0:
        print(f"[saved] {path}")


def _save_fitted_params_csv(fit_info: dict) -> None:
    path = OUTDIR / "fitted_params.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "run", "phi0", "E_eV", "R2"])
        w.writeheader()
        for (case_name, run_name), v in fit_info.items():
            w.writerow(
                {
                    "case": case_name,
                    "run": run_name,
                    "phi0": v["Phi0"],
                    "E_eV": v["E"],
                    "R2": v["R2"],
                }
            )
    if _RANK0:
        print(f"[saved] {path}")


# ── Main plotting / inversion routine ─────────────────────────────────────────


def make_dual_overlay_lnphi(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe: htm.Diffusivity,
    K_S_nickel_by_case: Dict[str, htm.Solubility],
    title_suffix: str = "SWAP configuration",
    save_csv: bool = True,
) -> dict:
    pts = _collect_points(cases, T2K, Y_FT_BY_TEMP_C, list(cases.keys()))
    invT_by, lnphi_by, sig_by, meta_by, metrics_by = _invert_points_with_sigma(
        pts, D_flibe, K_S_nickel_by_case
    )

    case_names = sorted({case for (case, _) in invT_by})
    palette = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1"])
    bc_colors = {name: palette[i % len(palette)] for i, name in enumerate(case_names)}
    runs_all = sorted({p.run for plist in meta_by.values() for p in plist})
    base_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    run_marker = {
        r: base_markers[i % len(base_markers)] for i, r in enumerate(runs_all)
    }

    MS, CAP, ELW, MECW = 5, 5, 0.9, 0.9
    fig, ax = plt.subplots(figsize=(8, 5.2))
    fig.subplots_adjust(top=0.80, bottom=0.25)

    for (case_name, run_name), invT in invT_by.items():
        color = bc_colors[case_name]
        lnphi = lnphi_by[(case_name, run_name)]
        sig = sig_by[(case_name, run_name)]
        x, y = 1000 * invT, np.exp(lnphi)
        ysig = np.where(np.isfinite(sig) & (sig > 0), y * sig, 0.0)

        ax.errorbar(
            x,
            y,
            yerr=ysig,
            fmt="none",
            ecolor=color,
            elinewidth=ELW,
            capsize=CAP,
            capthick=ELW,
            alpha=0.9,
            zorder=2.5,
        )
        ax.plot(
            x,
            y,
            linestyle="",
            marker=run_marker.get(run_name, "o"),
            ms=MS,
            mfc="white",
            mec=color,
            mew=MECW,
            label=f"{case_name} — {run_name}",
            zorder=4.0,
        )

    fit_info = {}
    for (case_name, run_name), invT in invT_by.items():
        key = (case_name, run_name)
        a, b, Phi0, E, band = _fit_lnphi(invT, lnphi_by[key], sigma_ln=sig_by[key])
        yhat = a + b * invT
        ss_res = np.sum((lnphi_by[key] - yhat) ** 2)
        ss_tot = np.sum((lnphi_by[key] - np.mean(lnphi_by[key])) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        fit_info[key] = dict(a=a, b=b, Phi0=Phi0, E=E, R2=R2, band=band)

        xx = np.linspace(invT.min(), invT.max(), 200)
        lo, hi = band(xx)
        c = bc_colors[case_name]
        ax.plot(
            1000 * xx,
            np.exp(a + b * xx),
            color=c,
            lw=2,
            label=f"{case_name} — {run_name} fit",
            zorder=3.5,
        )
        ax.plot(1000 * xx, np.exp(lo), color=c, lw=1.0, ls="--", alpha=0.8)
        ax.plot(1000 * xx, np.exp(hi), color=c, lw=1.0, ls="--", alpha=0.8)

    ax.set_xlabel("1000 / T [1/K]")
    ax.set_yscale("log")
    ax.set_ylabel(r"$\Phi_\mathrm{FLiBe}$  [H·m⁻¹·s⁻¹·Pa⁻⁰·⁵]")
    ax.grid(True, alpha=0.3)
    fig.suptitle(f"Pointwise inversion — {title_suffix}", y=0.98)

    bc_handles = [
        Line2D([0], [0], color=bc_colors[c], lw=2, label=f"{c} fit") for c in case_names
    ]
    run_handles = [
        Line2D(
            [0],
            [0],
            marker=run_marker[r],
            color="gray",
            linestyle="",
            mfc="white",
            ms=MS,
            label=r,
        )
        for r in runs_all
    ]
    ax.legend(
        handles=bc_handles + run_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=max(3, len(bc_handles) + len(run_handles)),
        frameon=True,
        framealpha=0.95,
        columnspacing=1.4,
        handlelength=2.2,
        borderaxespad=0.6,
    )

    _ensure_and_save(fig, OUTDIR / "dual_pointwise_lnphi_invT.png")

    if save_csv:
        rows = []
        for (case_name, run_name), invT in invT_by.items():
            metas = meta_by[(case_name, run_name)]
            lnphi = lnphi_by[(case_name, run_name)]
            sig = sig_by[(case_name, run_name)]
            mets = metrics_by[(case_name, run_name)]
            for i, p in enumerate(metas):
                s_ln = (
                    float(sig[i])
                    if np.isfinite(sig[i]) and sig[i] > 0
                    else float("nan")
                )
                phi = float(np.exp(lnphi[i]))
                s_phi = float(phi * s_ln) if np.isfinite(s_ln) else float("nan")
                sJ = get_exp_error(p.case, p.T_C, p.run)
                rows.append(
                    dict(
                        case=case_name,
                        run=p.run,
                        T_C=p.T_C,
                        T_K=p.T_K,
                        invT=1.0 / p.T_K,
                        ln_phi=float(lnphi[i]),
                        sigma_lnphi=s_ln,
                        phi=phi,
                        sigma_phi=s_phi,
                        J_exp=p.J_exp,
                        sigma_J=(0.0 if sJ is None else float(sJ)),
                        J_fit=mets[i]["J_fit"],
                        err_abs=mets[i]["err_abs"],
                        err_rel=mets[i]["err_rel"],
                    )
                )
        _save_inverted_points_csv(rows)
        _save_fitted_params_csv(
            {
                k: {"Phi0": v["Phi0"], "E": v["E"], "R2": v["R2"]}
                for k, v in fit_info.items()
            }
        )

    return fit_info


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)

    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

    T2K = {Tc: Tc + 273.15 for Tc in [500.0, 550.0, 600.0, 650.0, 700.0]}
    Y_FT_BY_TEMP_C = {
        500.0: 0.02914,
        550.0: 0.02919,
        600.0: 0.02925,
        650.0: 0.02930,
        700.0: 0.02936,
    }

    cases = {
        "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }

    # Load Ni solubility for each case from the dry-run Arrhenius fits.
    # Each case uses the BC-specific permeability (particle_flux_zero or sieverts).
    K_S_nickel_by_case = {
        case_name: _ni_solubility_for_bc(cfg["out_mode"])
        for case_name, cfg in cases.items()
    }

    fit_params = make_dual_overlay_lnphi(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        K_S_nickel_by_case=K_S_nickel_by_case,
        title_suffix="swap_infinite vs swap_transparent",
        save_csv=True,
    )

    if _RANK0:
        print("\nFitted parameters:")
        for (case_name, run_name), v in fit_params.items():
            print(
                f"  {case_name} — {run_name}: "
                f"Phi0={v['Phi0']:.3e}, E={v['E']:.4f} eV, R2={v['R2']:.4f}"
            )
