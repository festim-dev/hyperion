import festim as F
from mesh import generate_mesh, set_y_ft
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import h_transport_materials as htm
from typing import Literal, Tuple, Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import math
import matplotlib
import gc
from petsc4py import PETSc
import multiprocessing as mp

try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass

# ------------------------------ Globals ------------------------------
_DEFER_SHOW = True
kB_eV = 8.617333262e-5  # eV/K

# Mesh & model caches (per process)
_mesh_cache: dict[str, tuple] = {}

# Rank guard for logs (avoid duplicated “Info” lines with MPI>1)
_RANK0 = MPI.COMM_WORLD.rank == 0


# ------------------------------ Utilities ------------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _arrhenius_str(Phi0: float, E_eV: float, unit="H·m⁻¹·s⁻¹·Pa⁻¹") -> str:
    return (
        r"$\varphi(T)$ = "
        f"{Phi0:.4g} · exp("
        + r"$-\dfrac{"
        + f"{E_eV:.4g}"
        + r"\ \mathrm{eV}}{k_B T}$"
        + f")  [{unit}]"
    )


def _ensure_and_save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if _RANK0:
        print(f"[saved] {out_path}")
    plt.close(fig)


def _phi_at_T(T_K: float, perm) -> float:
    try:
        return float(perm.value(T_K).magnitude)
    except Exception:
        return float(perm.pre_exp) * math.exp(
            -float(perm.act_energy) / (kB_eV * float(T_K))
        )


def _dispose_model(m):
    if m is None:
        return
    try:
        # CylindricalFlux etc. can hang onto Forms/Vectors/Functions
        for e in getattr(m, "exports", []) or []:
            for attr in ("data", "field", "surface"):
                if hasattr(e, attr):
                    setattr(e, attr, None)
        # Drop strong refs on model
        m.exports = []
        m.interfaces = []
        m.subdomains = []
        m.boundary_conditions = []
        for attr in ("_forms", "_function_spaces", "_solvers", "_timers"):
            if hasattr(m, attr):
                setattr(m, attr, None)
        m.mesh = None
    except Exception:
        pass

    # Force Python + PETSc to actually free stuff now
    gc.collect()
    try:
        PETSc.garbage_cleanup()
    except Exception:
        pass
    try:
        MPI.COMM_WORLD.barrier()
    except Exception:
        pass


# ------------------------------ Mesh I/O ------------------------------
def _mesh_key_from_yft(y_ft: float) -> str:
    # Stabilize the filename key to 5 decimals (to avoid float precision issues)
    y5 = float(f"{float(y_ft):.5f}")
    return f"mesh_{y5:.5f}.msh"


def load_or_make_mesh(mesh_file: str, mesh_size: float, model_rank: int = 0):
    """
    Read a .msh once per process and reuse (prevents new communicators).
    """
    if not Path(mesh_file).exists():
        if _RANK0:
            print(f"[mesh] creating '{mesh_file}' (size={mesh_size:g})")
        generate_mesh(mesh_size=mesh_size, fname=mesh_file)

    if mesh_file in _mesh_cache:
        # optional debug:
        # if _RANK0: print(f"[mesh cache] HIT {mesh_file}")
        return _mesh_cache[mesh_file]

    _read = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, model_rank)
    mesh = _read.mesh
    cell_tags = _read.cell_tags
    facet_tags = _read.facet_tags
    _mesh_cache[mesh_file] = (mesh, cell_tags, facet_tags)
    return _mesh_cache[mesh_file]


# ------------------------------ Materials & Model ------------------------------
def make_materials(D_solid, D_liquid, K_solid, permeability_liquid):
    # solid
    D_0_solid = D_solid.pre_exp.magnitude
    E_D_solid = D_solid.act_energy.magnitude
    K_S_0_solid = K_solid.pre_exp.magnitude
    E_K_S_solid = K_solid.act_energy.magnitude
    # liquid
    D_0_liquid = D_liquid.pre_exp.magnitude
    E_D_liquid = D_liquid.act_energy.magnitude
    K_S_liquid = htm.Solubility(
        S_0=permeability_liquid.pre_exp / D_liquid.pre_exp,
        E_S=permeability_liquid.act_energy - D_liquid.act_energy,
        law=permeability_liquid.law,
    )
    K_S_0_liquid = K_S_liquid.pre_exp.magnitude
    E_K_S_liquid = K_S_liquid.act_energy.magnitude

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
    return mat_solid, mat_liquid


def make_model(
    D_flibe: htm.Diffusivity,
    D_nickel: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    K_S_nickel: htm.Solubility,
    temperature: float,
    P_up: float,
    case: Literal[
        "normal_infinite",
        "normal_transparent",
        "swap_infinite",
        "swap_transparent",
    ],
    mesh_size: float = 2e-4,
    penalty_term: float = 1e22,
    P_down: float = 5.0,
    out_bc: dict | None = None,
    y_ft: float | None = None,
) -> Tuple[F.HydrogenTransportProblemDiscontinuous, dict[str, list | CylindricalFlux]]:
    if y_ft is None:
        raise ValueError("y_ft must be provided")

    # stable file key by temperature → y_ft mapping
    y_ft_5 = float(f"{float(y_ft):.5f}")
    set_y_ft(y_ft_5)

    # record y_ft once
    Path("exports/yft_record").mkdir(parents=True, exist_ok=True)
    if _RANK0:
        with open("exports/yft_record/y_ft_values.txt", "a") as f:
            f.write(f"{y_ft_5:.5f}\n")

    mesh_file = _mesh_key_from_yft(y_ft_5)
    model_rank = 0
    mesh, cell_tags, facet_tags = load_or_make_mesh(
        mesh_file, mesh_size, model_rank=model_rank
    )

    mat_solid, mat_liquid = make_materials(
        D_solid=D_nickel,
        D_liquid=D_flibe,
        K_solid=K_S_nickel,
        permeability_liquid=permeability_flibe,
    )

    # BC-side parameters (Henry on liquid, Sieverts on Ni)
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )
    H_0_liq = K_S_liquid.pre_exp.magnitude
    E_H_liq = K_S_liquid.act_energy.magnitude
    K_S_0_Ni = K_S_nickel.pre_exp.magnitude
    E_S_Ni = K_S_nickel.act_energy.magnitude

    # subdomains
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

    my_model = F.HydrogenTransportProblemDiscontinuous()
    my_model.mesh = F.Mesh(mesh, coordinate_system="cylindrical")
    my_model.facet_meshtags = facet_tags
    my_model.volume_meshtags = cell_tags
    my_model.subdomains = [
        solid_volume,
        fluid_volume,
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

    my_model.method_interface = "penalty"
    interface = F.Interface(
        id=99, subdomains=[solid_volume, fluid_volume], penalty_term=penalty_term
    )
    my_model.interfaces = [interface]

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

    # case logic
    if case in ("normal_infinite", "normal_transparent"):
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [top_cap_Ni, top_sidewall_Ni]
        ] + [
            F.HenrysBC(
                subdomain=liquid_surface,
                species=H,
                pressure=P_down,
                H_0=H_0_liq,
                E_H=E_H_liq,
            )
        ]
    elif case in ("swap_infinite", "swap_transparent"):
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [top_cap_Ni, top_sidewall_Ni]
        ] + [
            F.HenrysBC(
                subdomain=liquid_surface,
                species=H,
                pressure=P_up,
                H_0=H_0_liq,
                E_H=E_H_liq,
            )
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
        ]
    else:
        raise ValueError(f"Unknown case: {case}")

    # optional out-surface (glovebox)
    if out_bc is None:
        out_bc = {"type": "none"}
    out_bcs = []
    t = out_bc.get("type", "none").lower()
    if t == "sieverts":
        out_bcs = [
            F.SievertsBC(
                subdomain=out_surf,
                species=H,
                pressure=float(out_bc.get("pressure", 0.0)),
                S_0=K_S_0_Ni,
                E_S=E_S_Ni,
            )
        ]
    elif t == "particle_flux_zero":
        out_bcs = [F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)]

    my_model.boundary_conditions = upstream_bcs + downstream_bcs + out_bcs
    my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)

    # Flux monitors
    flux_out_top_cap_Ni = CylindricalFlux(field=H, surface=top_cap_Ni)
    flux_out_top_sidewall_Ni = CylindricalFlux(field=H, surface=top_sidewall_Ni)
    flux_out_liquid_surface = CylindricalFlux(field=H, surface=liquid_surface)
    flux_out_mid_membrane_Ni = CylindricalFlux(field=H, surface=mid_membrane_Ni)
    flux_out_bottom_sidewall_Ni = CylindricalFlux(field=H, surface=bottom_sidewall_Ni)
    flux_out_bottom_cap_Ni = CylindricalFlux(field=H, surface=bottom_cap_Ni)
    glovebox_flux = CylindricalFlux(field=H, surface=out_surf)

    my_model.exports = [
        flux_out_top_cap_Ni,
        flux_out_top_sidewall_Ni,
        flux_out_bottom_sidewall_Ni,
        flux_out_liquid_surface,
        flux_out_mid_membrane_Ni,
        flux_out_bottom_cap_Ni,
        glovebox_flux,
    ]

    flux_by_label = {
        "top_cap_Ni": flux_out_top_cap_Ni,
        "top_sidewall_Ni": flux_out_top_sidewall_Ni,
        "liquid_surface": flux_out_liquid_surface,
        "mid_membrane_Ni": flux_out_mid_membrane_Ni,
        "bottom_sidewall_Ni": flux_out_bottom_sidewall_Ni,
        "bottom_cap_Ni": flux_out_bottom_cap_Ni,
    }
    six_labels = [
        "top_cap_Ni",
        "top_sidewall_Ni",
        "liquid_surface",
        "mid_membrane_Ni",
        "bottom_sidewall_Ni",
        "bottom_cap_Ni",
    ]
    up_labels = (
        ["mid_membrane_Ni", "bottom_cap_Ni", "bottom_sidewall_Ni"]
        if case in ("normal_infinite", "normal_transparent")
        else ["top_cap_Ni", "top_sidewall_Ni", "liquid_surface"]
    )
    down_labels = (
        ["top_cap_Ni", "top_sidewall_Ni", "liquid_surface"]
        if case in ("normal_infinite", "normal_transparent")
        else ["mid_membrane_Ni", "bottom_cap_Ni", "bottom_sidewall_Ni"]
    )
    fluxes_dict = {
        "flux_by_label": flux_by_label,
        "six_labels": six_labels,
        "glovebox_flux": glovebox_flux,
        "up_labels": up_labels,
        "down_labels": down_labels,
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


def _invert_point_child(p_dict, D_flibe, D_nickel, K_S_nickel, q):
    """Runs in a fresh process: do all J(φ) evaluations & bisection here."""

    tiny = np.finfo(float).tiny

    class P:  # small shim to avoid dataclass import in child
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

    p = P(**p_dict)
    out_bc = (
        {"type": "sieverts", "pressure": p.P_gb}
        if p.P_gb is not None
        else {"type": "particle_flux_zero"}
    )

    def J_of(phi_val: float) -> float:
        perm = htm.Permeability(pre_exp=float(phi_val), act_energy=0.0, law="henry")
        out = run_once(
            p.case,
            p.T_K,
            p.P_up,
            p.P_down,
            D_flibe,
            D_nickel,
            perm,
            K_S_nickel,
            out_bc,
            p.y_ft,
        )
        return max(float(out["total_out"]), tiny)

    # bracket in log space and bisection
    phi_lo, phi_hi = 1e10, 1e15
    tol_log, maxit = 3e-3, 18
    log_lo, log_hi = math.log10(phi_lo), math.log10(phi_hi)
    target = max(float(p.J_exp), tiny)

    J_lo, J_hi = J_of(10.0**log_lo), J_of(10.0**log_hi)
    if (J_lo - target) * (J_hi - target) > 0.0:
        if J_lo < target and J_hi < target:
            log_hi += 1.0
            J_hi = J_of(10.0**log_hi)
        elif J_lo > target and J_hi > target:
            log_lo -= 1.0
            J_lo = J_of(10.0**log_lo)

    for _ in range(maxit):
        log_mid = 0.5 * (log_lo + log_hi)
        J_mid = J_of(10.0**log_mid)
        if abs(log_hi - log_lo) < tol_log:
            q.put(10.0**log_mid)
            return
        if (J_lo - target) * (J_mid - target) <= 0.0:
            log_hi, J_hi = log_mid, J_mid
        else:
            log_lo, J_lo = log_mid, J_mid

    q.put(10.0 ** (0.5 * (log_lo + log_hi)))


def _run_once_in_subproc(args):
    (
        case,
        T_K,
        P_up,
        P_down,
        D_flibe,
        D_nickel,
        permeability_flibe,
        K_S_nickel,
        out_bc,
        y_ft,
    ) = args
    return run_once(
        case,
        T_K,
        P_up,
        P_down,
        D_flibe,
        D_nickel,
        permeability_flibe,
        K_S_nickel,
        out_bc,
        y_ft,
    )


def _child_run_once(args, q):
    """
    Runs in a fresh Python process (fresh MPI world), returns result via queue.
    """
    (
        case,
        T_K,
        P_up,
        P_down,
        D_flibe,
        D_nickel,
        permeability_flibe,
        K_S_nickel,
        out_bc,
        y_ft,
    ) = args
    try:
        out = run_once(
            case,
            T_K,
            P_up,
            P_down,
            D_flibe,
            D_nickel,
            permeability_flibe,
            K_S_nickel,
            out_bc,
            y_ft,
        )
        q.put(("ok", out))
    except Exception as e:
        q.put(("err", repr(e)))


def run_once_isolated(
    case,
    T_K,
    P_up,
    P_down,
    D_flibe,
    D_nickel,
    permeability_flibe,
    K_S_nickel,
    out_bc,
    y_ft,
):
    """
    One-off child process (no Pool) → no SIGTERM during teardown.
    """
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    p = ctx.Process(
        target=_child_run_once,
        args=(
            (
                case,
                T_K,
                P_up,
                P_down,
                D_flibe,
                D_nickel,
                permeability_flibe,
                K_S_nickel,
                out_bc,
                y_ft,
            ),
            q,
        ),
        daemon=False,
    )
    p.start()
    status, payload = q.get()
    p.join()
    if status == "err":
        raise RuntimeError(f"child solve failed: {payload}")
    return payload


def run_once_safe(
    case,
    T_K,
    P_up,
    P_down,
    D_flibe,
    D_nickel,
    permeability_flibe,
    K_S_nickel,
    out_bc,
    y_ft,
):
    """
    Use in-process for speed (swap_*), isolate normal_* in a fresh process.
    """
    if case.startswith("normal"):
        return run_once_isolated(
            case,
            T_K,
            P_up,
            P_down,
            D_flibe,
            D_nickel,
            permeability_flibe,
            K_S_nickel,
            out_bc,
            y_ft,
        )
    else:
        return run_once(
            case,
            T_K,
            P_up,
            P_down,
            D_flibe,
            D_nickel,
            permeability_flibe,
            K_S_nickel,
            out_bc,
            y_ft,
        )


def run_once(
    case: str,
    T_K: float,
    P_up: float,
    P_down: float,
    D_flibe,
    D_nickel,
    permeability_flibe,
    K_S_nickel,
    out_bc: dict | None = None,
    y_ft: float | None = None,
):
    """
    Solve steady-state once and return aggregated fluxes.
    IMPORTANT: Disposes model to avoid MPI communicator leaks.
    """
    my_model, fluxes_dict = make_model(
        temperature=T_K,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        permeability_flibe=permeability_flibe,
        K_S_nickel=K_S_nickel,
        P_up=P_up,
        case=case,
        P_down=P_down,
        out_bc=out_bc,
        y_ft=y_ft,
    )
    my_model.initialise()
    my_model.run()

    flux_objects = fluxes_dict["flux_by_label"]
    six_labels = fluxes_dict["six_labels"]
    vals_six = {label: _get_flux_value(flux_objects[label]) for label in six_labels}
    glovebox_val = float(_get_flux_value(fluxes_dict["glovebox_flux"]))
    up_labels = fluxes_dict["up_labels"]
    down_labels = fluxes_dict["down_labels"]
    total_up = float(np.sum([vals_six[label] for label in up_labels], dtype=float))
    total_down = float(np.sum([vals_six[label] for label in down_labels], dtype=float))

    # cleanup to free communicators
    _dispose_model(my_model)

    display_names = {
        "top_cap_Ni": "Top cap (Ni)",
        "top_sidewall_Ni": "Upper Ni sidewall",
        "liquid_surface": "FLiBe surface",
        "mid_membrane_Ni": "Middle membrane",
        "bottom_cap_Ni": "Bottom Ni (top face)",
        "bottom_sidewall_Ni": "Lower Ni sidewall",
    }
    per_surface = {
        "labels": six_labels,
        "values": [vals_six[label] for label in six_labels],
        "up_labels": up_labels,
        "down_labels": down_labels,
        "up_names": [display_names[label] for label in up_labels],
        "down_names": [display_names[label] for label in down_labels],
        "up_vals": [vals_six[label] for label in up_labels],
        "down_vals": [vals_six[label] for label in down_labels],
    }
    _dispose_model(my_model)
    return dict(
        total_in=total_up,
        glovebox=glovebox_val,
        total_out=total_down,
        balance=total_up + glovebox_val + total_down,
        per_surface=per_surface,
    )


# ------------------------------ Invert → Fit (guess-independent) ------------------------------
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


def _collect_points_for_cases(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    allowed_case_names: Optional[List[str]] = None,
) -> List[CalibPoint]:
    pts: List[CalibPoint] = []
    for case_name, cfg in cases.items():
        if allowed_case_names and case_name not in allowed_case_names:
            continue
        if not (case_name.startswith("normal") or case_name.startswith("swap")):
            continue

        table = cfg["table"]
        for Tc, row in table.items():
            Tk = T2K[Tc]
            # Stable y_ft retrieval (5 decimals)
            y_raw = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])
            y5 = float(f"{float(y_raw):.5f}")
            runs = row.get("runs", {})
            for run_name, cond in runs.items():
                pts.append(
                    CalibPoint(
                        case=case_name,
                        T_C=float(Tc),
                        T_K=float(Tk),
                        run=str(run_name),
                        P_up=float(cond["P_up"]),
                        P_down=float(cond["P_down"]),
                        P_gb=float(cond.get("P_gb")) if "P_gb" in cond else None,
                        y_ft=y5,
                        J_exp=float(cond["J_exp"]),
                    )
                )
    if not pts:
        raise RuntimeError("No calibration points found.")
    pts.sort(key=lambda p: (p.case, p.T_C, p.run))
    return pts


def _phi_match_exp_for_point(p, D_flibe, D_nickel, K_S_nickel) -> float:
    """Spawn one child to invert this point (normal/swap both OK)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
    # send only plain data to child (avoid dataclass/method closures)
    p_dict = {
        "case": p.case,
        "T_C": p.T_C,
        "T_K": p.T_K,
        "run": p.run,
        "P_up": p.P_up,
        "P_down": p.P_down,
        "P_gb": p.P_gb,
        "y_ft": p.y_ft,
        "J_exp": p.J_exp,
    }
    proc = ctx.Process(
        target=_invert_point_child,
        args=(p_dict, D_flibe, D_nickel, K_S_nickel, q),
        daemon=False,
    )
    proc.start()
    phi_T = q.get()
    proc.join()
    return float(phi_T)


def _parity_metrics(jm: np.ndarray, je: np.ndarray) -> tuple[float, float]:
    eps = np.finfo(float).tiny
    rmse_log = float(np.sqrt(np.mean((np.log10(jm + eps) - np.log10(je + eps)) ** 2)))
    max_rel = float(np.max(np.abs((jm - je) / (je + eps))))
    return rmse_log, max_rel


def calibrate_phi_schemeA(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    D_nickel,
    K_S_nickel,
    permeability_flibe_guess,  # kept for API compatibility; not used here
    outdir: Path = Path("exports") / "figs" / "calibration_A",
    also_show: bool = False,
    allowed_case_names: Optional[List[str]] = None,
    use_weighted: bool = True,  # weight by experimental error if available
):
    """
    Invert → Fit (guess-independent):
      1) For each point p, find φ_T so J_model(φ_T)=J_exp (bisection in log10 φ).
      2) Fit ln(φ_T) vs 1/T → (Φ0, E). Report FINAL R² on the ln-φ plot.
      3) Validate and plot parity + per-run bar charts.
    """
    case_suffix = None if (allowed_case_names is None) else allowed_case_names[0]
    outdir = outdir if case_suffix is None else (outdir / case_suffix)
    _ensure_dir(outdir)

    pts = _collect_points_for_cases(cases, T2K, Y_FT_BY_TEMP_C, allowed_case_names)

    # 1) per-T inversion
    ln_phi_list, invT_list, rows = [], [], []
    sigmas_lnphi = []  # propagated errors for ln(phi)
    for p in pts:
        phi_T = _phi_match_exp_for_point(p, D_flibe, D_nickel, K_S_nickel)

        # ---propagate σ_J -> σ_lnphi via finite-difference on J(phi) ---
        sigma_J = None
        try:
            sigma_J = get_exp_error(p.case, p.T_C, p.run)  # may be None/NaN
        except Exception:
            sigma_J = None

        tiny = np.finfo(float).tiny
        if (sigma_J is not None) and np.isfinite(sigma_J) and (sigma_J > 0.0):
            # Outflow BC for this point (to match inversion)
            out_bc_p = (
                {"type": "sieverts", "pressure": p.P_gb}
                if p.P_gb is not None
                else {"type": "particle_flux_zero"}
            )

            def J_of_phi(phi_val: float) -> float:
                perm_loc = htm.Permeability(
                    pre_exp=float(phi_val), act_energy=0.0, law="henry"
                )
                out_loc = run_once_safe(
                    p.case,
                    p.T_K,
                    p.P_up,
                    p.P_down,
                    D_flibe,
                    D_nickel,
                    perm_loc,
                    K_S_nickel,
                    out_bc_p,
                    p.y_ft,
                )
                return float(out_loc["total_out"])

            # centered derivative dJ/dphi at phi_T
            rel_step = 0.02  # 2% step;
            delta = max(rel_step * float(phi_T), 1e-16)
            try:
                Jp = J_of_phi(float(phi_T) + delta)
                Jm = J_of_phi(max(float(phi_T) - delta, tiny))
                dJ_dphi = (Jp - Jm) / (2.0 * delta)
                # σ_phi ≈ σ_J / |dJ/dphi| ;  σ_lnphi = σ_phi / phi_T
                sigma_lnphi = (sigma_J / max(abs(dJ_dphi), tiny)) / max(
                    float(phi_T), tiny
                )
                if (not np.isfinite(sigma_lnphi)) or (sigma_lnphi <= 0.0):
                    sigma_lnphi = None
            except Exception:
                sigma_lnphi = None
        else:
            sigma_lnphi = None

        sigmas_lnphi.append(sigma_lnphi)

        ln_phi_list.append(math.log(max(phi_T, tiny)))
        invT_list.append(1.0 / p.T_K)
        rows.append({"case": p.case, "T_C": p.T_C, "run": p.run, "phi_T": phi_T})

    invT = np.asarray(invT_list, dtype=float)
    ln_phi = np.asarray(ln_phi_list, dtype=float)

    # Optional weights using propagated σ for ln(phi)
    w = None
    if any((s is not None) for s in sigmas_lnphi):
        w = np.array(
            [
                0.0 if (s is None or (not np.isfinite(s)) or (s <= 0.0)) else 1.0 / s
                for s in sigmas_lnphi
            ],
            dtype=float,
        )
        if (not np.any(np.isfinite(w))) or np.all(w == 0.0):
            w = None

    # 2) fit ln φ = a + b (1/T)
    if w is None:
        b, a = np.polyfit(invT, ln_phi, deg=1)
    else:
        # numpy polyfit supports weights via w=
        b, a = np.polyfit(invT, ln_phi, deg=1, w=w)
    Phi0 = float(math.exp(a))
    E = float(-b * kB_eV)

    y_pred = b * invT + a
    ss_res = float(np.sum((ln_phi - y_pred) ** 2))
    ss_tot = float(np.sum((ln_phi - float(np.mean(ln_phi))) ** 2))
    R2_final = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    perm_fitted = htm.Permeability(pre_exp=Phi0, act_energy=E, law="henry")

    # 3) Validation & plots
    if _RANK0:
        print("\n[Scheme A] Validation with fitted (Phi0, E):")
        print(
            "Case | T[°C] | Run |   J_model_fit [H/s]   |    J_exp [H/s]     |  log10 err"
        )

    jm_fit_list, je_list, labels = [], [], []
    for p in pts:
        out_bc = (
            {"type": "sieverts", "pressure": p.P_gb}
            if p.P_gb is not None
            else {"type": "particle_flux_zero"}
        )
        out = run_once_safe(
            p.case,
            p.T_K,
            p.P_up,
            p.P_down,
            D_flibe,
            D_nickel,
            perm_fitted,
            K_S_nickel,
            out_bc,
            p.y_ft,
        )

        eps = np.finfo(float).tiny
        jm = max(float(out["total_out"]), eps)
        je = max(float(p.J_exp), eps)
        jm_fit_list.append(jm)
        je_list.append(je)
        labels.append(f"{p.case}@{int(p.T_C)}°C")
        if _RANK0:
            rlog = math.log10(jm) - math.log10(je)
            print(
                f"{p.case:<18} | {int(p.T_C):>5} | {p.run:<4} | {jm:>20.3e} | {je:>18.3e} | {rlog:>9.3f}"
            )

    # lnφ vs 1/T plot (FINAL fit + FINAL R²)
    expr_line = _arrhenius_str(Phi0, E)
    case_name_text = "" if case_suffix is None else f" ({case_suffix})"
    fig1, ax1 = plt.subplots(figsize=(6.6, 4.8))
    ax1.scatter(invT, ln_phi, label="Final corrected points", zorder=3)
    order = np.argsort(invT)
    ax1.plot(
        invT[order], (b * invT + a)[order], linestyle="--", label="Final linear fit"
    )
    ax1.set_xlabel("1 / T [1/K]")
    ax1.set_ylabel("ln φ  [ln(H·m⁻¹·s⁻¹·Pa⁻¹)]")
    fig1.suptitle(expr_line + f"   $R^2 = {R2_final:.4f}$", y=0.98, fontsize=10)
    ax1.set_title("Scheme A: ln(φ) vs 1/T" + case_name_text)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    _ensure_and_save(fig1, outdir / "schemeA_lnphi_vs_invT.png")

    # Parity
    jm_arr = np.array(jm_fit_list, dtype=float)
    je_arr = np.array(je_list, dtype=float)
    lo = float(min(jm_arr.min(), je_arr.min()))
    hi = float(max(jm_arr.max(), je_arr.max()))
    fig2, ax2 = plt.subplots(figsize=(6.0, 6.0))
    ax2.scatter(je_arr, jm_arr)
    ax2.plot([lo, hi], [lo, hi], linestyle="--")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Experimental flux  J_exp [H/s]")
    ax2.set_ylabel("Model flux (fitted) J_model [H/s]")
    ax2.set_title("Scheme A: Parity" + case_name_text)
    fig2.suptitle(expr_line, y=0.98, fontsize=10)
    for x, y, lab in zip(je_arr, jm_arr, labels):
        ax2.text(x, y, lab, fontsize=8)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    _ensure_and_save(fig2, outdir / "schemeA_parity.png")

    # per-case breakdowns if single case
    if case_suffix is not None:
        case_cfg = cases[case_suffix]
        plot_case_breakdowns_with_exp(
            case_name=case_suffix,
            case_cfg=case_cfg,
            T2K=T2K,
            Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
            D_flibe=D_flibe,
            permeability_flibe_fitted=perm_fitted,
            outdir_root=outdir.parent if outdir.name == case_suffix else outdir.parent,
        )

    if also_show and "agg" not in matplotlib.get_backend().lower():
        plt.show()

    return {
        "phi0": Phi0,
        "E": E,
        "R2": R2_final,
        "points": rows,
        "fig_dir": str(outdir),
        "permeability": perm_fitted,
    }


def calibrate_phi_all_cases(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    D_nickel,
    K_S_nickel,
    permeability_flibe_guess,
    outdir_root: Path = Path("exports") / "figs" / "calibration_A",
) -> Dict[str, dict]:
    results: Dict[str, dict] = {}
    for case_name in cases.keys():
        res = calibrate_phi_schemeA(
            cases=cases,
            T2K=T2K,
            Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
            D_flibe=D_flibe,
            D_nickel=D_nickel,
            K_S_nickel=K_S_nickel,
            permeability_flibe_guess=None,  # ignored in invert mode
            outdir=outdir_root,
            also_show=False,
            allowed_case_names=[case_name],
        )
        results[case_name] = res
    return results


def fig_saving(case_name: str, T_C: int, run_name: str) -> Path:
    return Path("exports") / "figs" / "calibration_A" / case_name / f"{T_C}C" / run_name


def save_breakdown(fig, outdir: Path, stem: str):
    _ensure_and_save(fig, outdir / f"{stem}.png")


def plot_case_breakdowns_with_exp(
    case_name: str,
    case_cfg: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    permeability_flibe_fitted,
    outdir_root: Path = Path("exports") / "figs" / "calibration_A",
):
    table = case_cfg.get("table", {})
    case_outdir = outdir_root / case_name
    _ensure_dir(case_outdir)

    for Tc in sorted(table.keys()):
        Tk = T2K[Tc]
        y_val = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])
        y_ft = float(f"{float(y_val):.5f}")

        runs = table[Tc].get("runs", {})
        for run_name, cond in runs.items():
            out_bc = (
                {"type": "sieverts", "pressure": float(cond.get("P_gb", 0.0))}
                if "P_gb" in cond
                else {"type": "particle_flux_zero"}
            )
            res = run_once_safe(
                case=case_name,
                T_K=Tk,
                P_up=float(cond["P_up"]),
                P_down=float(cond["P_down"]),
                D_flibe=D_flibe,
                D_nickel=D_nickel,
                K_S_nickel=K_S_nickel,
                permeability_flibe=permeability_flibe_fitted,
                out_bc=out_bc,
                y_ft=y_ft,
            )

            per = res["per_surface"]
            six_labels = per["labels"]
            six_values = [float(v) for v in per["values"]]
            values = [
                float(cond["J_exp"]),
                float(res["total_in"]),
                float(res["total_out"]),
                float(res["glovebox"]),
            ] + six_values
            labels = ["Flux exp", "Flux in", "Flux out", "Glovebox"] + list(six_labels)

            fig, ax = plt.subplots(figsize=(13, 6))
            x = np.arange(len(labels), dtype=float)
            bars = ax.bar(x, np.abs(values))
            for bar, v in zip(bars, values):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    (h if h > 0 else 0.0) * 1.02 + (1e-30 if h == 0 else 0),
                    f"{v:.2e}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Flux [H/s]")
            ax.set_title(
                f"{case_name} — {run_name} — {int(Tc)} °C  (y_ft={float(y_ft):.5f} m)"
            )
            info = (
                f"Case: {case_name}\nRun: {run_name}\nT = {int(Tc)} °C (T_K={Tk:.2f})\n"
                f"y_ft = {float(y_ft):.5f} m\nP_up = {float(cond['P_up']):.2e} Pa\n"
                f"P_down = {float(cond['P_down']):.2e} Pa\n"
                + (
                    f"P_glovebox = {float(cond['P_gb']):.2e} Pa"
                    if "P_gb" in cond
                    else "P_glovebox = (closed)"
                )
            )
            ax.text(
                0.99,
                0.98,
                info,
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0.5),
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.tight_layout()

            fd = fig_saving(case_name, int(Tc), run_name)
            save_breakdown(
                fig,
                fd,
                stem=f"{case_name}_{int(Tc)}C_{run_name}_breakdown_exp_vs_model",
            )


# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    # quieter logs from dolfinx
    set_log_level(LogLevel.WARNING)

    # ---- materials ----
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")
    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

    # Temperatures and K conversion
    temps_C_all = [500.0, 550.0, 600.0, 650.0, 700.0]
    T2K = {Tc: Tc + 273.15 for Tc in temps_C_all}

    # temperature-dependent FLiBe thickness [m]
    Y_FT_BY_TEMP_C = {
        500.0: 0.02914,
        550.0: 0.02919,
        600.0: 0.02925,
        650.0: 0.02930,
        700.0: 0.02936,
    }

    # ---- cases ----
    normal_infinite = {
        500.0: {
            "runs": {
                "Run 1": {"P_up": 1.11e5, "P_down": 4.55, "J_exp": 1.04e15},
                "Run 2": {"P_up": 1.05e5, "P_down": 4.84, "J_exp": 1.09e15},
            }
        },
        550.0: {
            "runs": {
                "Run 1": {"P_up": 1.10e5, "P_down": 7.10, "J_exp": 1.52e15},
                "Run 2": {"P_up": 1.05e5, "P_down": 7.80, "J_exp": 1.69e15},
            }
        },
        600.0: {
            "runs": {
                "Run 1": {"P_up": 1.05e5, "P_down": 9.36, "J_exp": 2.01e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 1.07e1, "J_exp": 2.38e15},
            }
        },
        650.0: {
            "runs": {
                "Run 1": {"P_up": 1.05e5, "P_down": 1.51e1, "J_exp": 3.30e15},
                "Run 2": {"P_up": 1.04e5, "P_down": 1.40e1, "J_exp": 3.00e15},
            }
        },
        700.0: {
            "runs": {
                "Run 1": {"P_up": 1.03e5, "P_down": 1.97e1, "J_exp": 4.34e15},
                "Run 2": {"P_up": 1.02e5, "P_down": 2.04e1, "J_exp": 4.38e15},
            }
        },
    }
    normal_transparent = {
        500.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.11e5,
                    "P_down": 4.55,
                    "P_gb": 5.0,
                    "J_exp": 1.04e15,
                },
                "Run 2": {
                    "P_up": 1.05e5,
                    "P_down": 4.84,
                    "P_gb": 3.0,
                    "J_exp": 1.09e15,
                },
            }
        },
        550.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.10e5,
                    "P_down": 7.10,
                    "P_gb": 5.0,
                    "J_exp": 1.52e15,
                },
                "Run 2": {
                    "P_up": 1.05e5,
                    "P_down": 7.08,
                    "P_gb": 5.0,
                    "J_exp": 1.69e15,
                },
            }
        },
        600.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.05e5,
                    "P_down": 9.36,
                    "P_gb": 5.0,
                    "J_exp": 2.01e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 1.07e1,
                    "P_gb": 5.0,
                    "J_exp": 2.38e15,
                },
            }
        },
        650.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.05e5,
                    "P_down": 1.51e1,
                    "P_gb": 5.0,
                    "J_exp": 3.30e15,
                },
                "Run 2": {
                    "P_up": 1.04e5,
                    "P_down": 1.40e1,
                    "P_gb": 5.0,
                    "J_exp": 3.00e15,
                },
            }
        },
        700.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.03e5,
                    "P_down": 1.97e1,
                    "P_gb": 7.0,
                    "J_exp": 4.34e15,
                },
                "Run 2": {
                    "P_up": 1.02e5,
                    "P_down": 2.04e1,
                    "P_gb": 7.0,
                    "J_exp": 4.38e15,
                },
            }
        },
    }
    swap_infinite = {
        500.0: {
            "runs": {
                "Run 1": {"P_up": 1.31e5, "P_down": 1.77e1, "J_exp": 3.89e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 1.99e1, "J_exp": 4.34e15},
            }
        },
        550.0: {
            "runs": {
                "Run 1": {"P_up": 1.31e5, "P_down": 3.21e1, "J_exp": 7.20e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 3.89e1, "J_exp": 8.58e15},
            }
        },
        600.0: {
            "runs": {
                "Run 1": {"P_up": 1.33e5, "P_down": 3.57e1, "J_exp": 7.64e15},
                "Run 2": {"P_up": 1.32e5, "P_down": 4.62e1, "J_exp": 1.01e16},
            }
        },
        650.0: {
            "runs": {"Run 2": {"P_up": 1.32e5, "P_down": 5.02e1, "J_exp": 1.10e16}}
        },
        700.0: {
            "runs": {
                "Run 1": {"P_up": 1.32e5, "P_down": 4.07e1, "J_exp": 9.04e15},
                "Run 2": {"P_up": 1.32e5, "P_down": 4.78e1, "J_exp": 1.04e16},
            }
        },
    }
    swap_transparent = {
        500.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 1.77e1,
                    "P_gb": 7.0,
                    "J_exp": 3.89e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 3.89e1,
                    "P_gb": 7.0,
                    "J_exp": 4.34e15,
                },
            }
        },
        550.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 3.21e1,
                    "P_gb": 1.0e1,
                    "J_exp": 7.20e15,
                },
                "Run 2": {
                    "P_up": 1.31e5,
                    "P_down": 3.89e1,
                    "P_gb": 1.0e1,
                    "J_exp": 8.58e15,
                },
            }
        },
        600.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.33e5,
                    "P_down": 3.57e1,
                    "P_gb": 1.2e1,
                    "J_exp": 7.64e15,
                },
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 4.62e1,
                    "P_gb": 1.2e1,
                    "J_exp": 1.01e16,
                },
            }
        },
        650.0: {
            "runs": {
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 5.02e1,
                    "P_gb": 1.5e1,
                    "J_exp": 1.10e16,
                },
            }
        },
        700.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.32e5,
                    "P_down": 4.07e1,
                    "P_gb": 2.2e1,
                    "J_exp": 9.04e15,
                },
                "Run 2": {
                    "P_up": 1.32e5,
                    "P_down": 4.78e1,
                    "P_gb": 2.2e1,
                    "J_exp": 1.04e16,
                },
            }
        },
    }
    exp_error_data = {
        "normal_infinite": {
            500.0: {"runs": {"Run 1": 2.72e13, "Run 2": 4.49e13}},
            550.0: {"runs": {"Run 1": 4.62e13, "Run 2": 5.50e13}},
            600.0: {"runs": {"Run 1": 5.04e13, "Run 2": 7.63e13}},
            650.0: {"runs": {"Run 1": 7.87e13, "Run 2": 6.96e13}},
            700.0: {"runs": {"Run 1": 9.65e13, "Run 2": 9.34e13}},
        },
        "normal_transparent": {
            500.0: {"runs": {"Run 1": 2.72e13, "Run 2": 4.49e13}},
            550.0: {"runs": {"Run 1": 4.62e13, "Run 2": 5.50e13}},
            600.0: {"runs": {"Run 1": 5.04e13, "Run 2": 7.63e13}},
            650.0: {"runs": {"Run 1": 7.87e13, "Run 2": 6.96e13}},
            700.0: {"runs": {"Run 1": 9.65e13, "Run 2": 9.34e13}},
        },
        "swap_infinite": {
            500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13}},
            550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
            600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14}},
            650.0: {"runs": {"Run 2": 2.26e14}},
            700.0: {"runs": {"Run 1": 1.99e14, "Run 2": 2.19e14}},
        },
        "swap_transparent": {
            500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13}},
            550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
            600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14}},
            650.0: {"runs": {"Run 2": 2.26e14}},
            700.0: {"runs": {"Run 1": 1.99e14, "Run 2": 2.19e14}},
        },
    }

    def get_exp_error(case_name: str, temp, run_name: str = "Run 1"):
        case = exp_error_data.get(case_name)
        if case is None:
            return None
        entry = case.get(float(temp))
        if entry is None:
            return None
        if isinstance(entry, (int, float)):
            val = float(entry)
            return val if np.isfinite(val) and val > 0.0 else None
        if isinstance(entry, dict):
            runs = entry.get("runs")
            if isinstance(runs, dict):
                val = runs.get(run_name)
            else:
                val = entry.get(run_name)
            try:
                val = float(val)
            except (TypeError, ValueError):
                return None
            return val if np.isfinite(val) and val > 0.0 else None
        return None

    # Choose the case(s) to calibrate
    cases = {
        # "normal_infinite": {"table": normal_infinite, "out_mode": "particle_flux_zero"},
        # "normal_transparent": {"table": normal_transparent, "out_mode": "sieverts"},
        "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }

    # Run calibration (invert → fit)
    results = calibrate_phi_all_cases(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        permeability_flibe_guess=None,
        outdir_root=Path("exports") / "figs" / "calibration_A",
    )
    if _RANK0:
        print("Per-case fitted parameters:")
        for k, v in results.items():
            print(k, {kk: v[kk] for kk in ("phi0", "E", "R2", "fig_dir")})

    if (not _DEFER_SHOW) and ("agg" not in matplotlib.get_backend().lower()):
        plt.show()
