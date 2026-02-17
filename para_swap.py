import festim as F
from mesh import generate_mesh, set_y_ft
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI
import h_transport_materials as htm
from typing import Tuple, Optional, List, Dict
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
import math
import matplotlib
import gc
from petsc4py import PETSc
import multiprocessing as mp
import matplotlib as mpl
from matplotlib.lines import Line2D


try:
    mp.set_start_method("spawn")
except RuntimeError:
    pass  # already set elsewhere

# ------------------------------ Globals ------------------------------
_DEFER_SHOW = True
kB_eV = 8.617333262e-5  # eV/K
_mesh_cache: dict[str, tuple] = {}
_RANK0 = MPI.COMM_WORLD.rank == 0


# ------------------------------ Utilities ------------------------------
def _ensure_and_save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    if _RANK0:
        print(f"[saved] {out_path}")
    plt.close(fig)


# ------------------------------ Model disposal ------------------------------
def _dispose_model(m):
    if m is None:
        return
    try:
        for e in getattr(m, "exports", []) or []:
            for attr in ("data", "field", "surface"):
                if hasattr(e, attr):
                    setattr(e, attr, None)
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
    y5 = float(f"{float(y_ft):.5f}")
    return f"mesh_{y5:.5f}.msh"


def load_or_make_mesh(mesh_file: str, mesh_size: float, model_rank: int = 0):
    if not Path(mesh_file).exists():
        if _RANK0:
            print(f"[mesh] creating '{mesh_file}' (size={mesh_size:g})")
        generate_mesh(mesh_size=mesh_size, fname=mesh_file)

    if mesh_file in _mesh_cache:
        return _mesh_cache[mesh_file]

    _read = gmshio.read_from_msh(mesh_file, MPI.COMM_WORLD, model_rank)
    mesh = _read.mesh
    cell_tags = _read.cell_tags
    facet_tags = _read.facet_tags
    _mesh_cache[mesh_file] = (mesh, cell_tags, facet_tags)
    return _mesh_cache[mesh_file]


# ------------------------------ Materials & Model ------------------------------
def make_materials(D_solid, D_liquid, K_solid, permeability_liquid):
    D_0_solid = D_solid.pre_exp.magnitude
    E_D_solid = D_solid.act_energy.magnitude
    K_S_0_solid = K_solid.pre_exp.magnitude
    E_K_S_solid = K_solid.act_energy.magnitude

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
    mesh_size: float = 2e-4,
    penalty_term: float = 1e22,
    P_down: float = 5.0,
    out_bc: dict | None = None,
    y_ft: float | None = None,
) -> Tuple[F.HydrogenTransportProblemDiscontinuous, dict[str, list | CylindricalFlux]]:
    if y_ft is None:
        raise ValueError("y_ft must be provided")

    y_ft_5 = float(f"{float(y_ft):.5f}")
    set_y_ft(y_ft_5)

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

    H_0_liq = mat_liquid.K_S_0
    E_H_liq = mat_liquid.E_K_S
    K_S_0_Ni = mat_solid.K_S_0
    E_S_Ni = mat_solid.E_K_S

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

    downstream_bcs = [
        F.SievertsBC(subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni)
        for s in [mid_membrane_Ni, bottom_cap_Ni, bottom_sidewall_Ni]
    ]
    upstream_bcs = [
        F.SievertsBC(subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni)
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
    down_labels = ["mid_membrane_Ni", "bottom_cap_Ni", "bottom_sidewall_Ni"]
    up_labels = ["top_cap_Ni", "top_sidewall_Ni", "liquid_surface"]
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


def run_once(
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
    my_model, fluxes_dict = make_model(
        temperature=T_K,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        permeability_flibe=permeability_flibe,
        K_S_nickel=K_S_nickel,
        P_up=P_up,
        P_down=P_down,
        out_bc=out_bc,
        y_ft=y_ft,
    )
    my_model.initialise()
    my_model.run()

    flux_objects = fluxes_dict["flux_by_label"]
    six_labels = fluxes_dict["six_labels"]
    vals_six = {label: _get_flux_value(flux_objects[label]) for label in six_labels}

    down_labels = fluxes_dict["down_labels"]
    total_down = float(np.sum([vals_six[label] for label in down_labels], dtype=float))
    _dispose_model(my_model)

    return total_down


# ------------------------------ Invert → Fit ------------------------------
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
        if not (case_name.startswith("swap")):
            continue
        table = cfg["table"]
        for Tc, row in table.items():
            Tk = T2K[Tc]
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


# --- EXP error lookup ---
exp_error_data = {
    "swap_infinite": {
        500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13, "Run 3": 4.90e13}},
        550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
        600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14, "Run 3": 1.01e14}},
        650.0: {"runs": {"Run 2": 2.26e14}},
        700.0: {"runs": {"Run 1": 1.99e14, "Run 2": 2.19e14, "Run 3": 1.52e14}},
    },
    "swap_transparent": {
        500.0: {"runs": {"Run 1": 8.81e13, "Run 2": 9.63e13, "Run 3": 4.90e13}},
        550.0: {"runs": {"Run 1": 1.50e14, "Run 2": 1.77e14}},
        600.0: {"runs": {"Run 1": 1.79e14, "Run 2": 2.09e14, "Run 3": 1.01e14}},
        650.0: {"runs": {"Run 2": 2.26e14}},
        700.0: {"runs": {"Run 1": 1.99e14, "Run 2": 2.19e14, "Run 3": 1.52e14}},
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
        val = runs.get(run_name) if isinstance(runs, dict) else entry.get(run_name)
        try:
            val = float(val)
        except (TypeError, ValueError):
            return None
        return val if np.isfinite(val) and val > 0.0 else None
    return None


# ------------------------------ Core: inversion with uncertainty ------------------------------
# ----- child processes for safe solves and inversion -----
def _invert_point_child(p_dict, D_flibe, D_nickel, K_S_nickel, q):
    tiny = np.finfo(float).tiny

    class P:
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

    def J_of_phi(phi_val: float) -> float:
        perm = htm.Permeability(pre_exp=float(phi_val), act_energy=0.0, law="henry")
        out = run_once(
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
        return max(float(out), tiny)

    phi_lo, phi_hi = 1e10, 1e15
    tol_log, maxit = 3e-3, 18
    log_lo, log_hi = math.log10(phi_lo), math.log10(phi_hi)
    target = max(float(p.J_exp), tiny)

    J_lo, J_hi = J_of_phi(10.0**log_lo), J_of_phi(10.0**log_hi)
    if (J_lo - target) * (J_hi - target) > 0.0:
        if J_lo < target and J_hi < target:
            log_hi += 1.0
            J_hi = J_of_phi(10.0**log_hi)
        elif J_lo > target and J_hi > target:
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


def _phi_match_exp_for_point(p, D_flibe, D_nickel, K_S_nickel) -> float:
    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()
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


def _invert_points_with_sigma(pts, D_flibe, D_nickel, K_S_nickel):
    """
    Perform pointwise inversion for each experimental point and
    group results by (case, run).

    Returns:
        invT_by_case:      dict[(case, run)] -> np.array of 1/T
        lnphi_by_case:     dict[(case, run)] -> np.array of ln(phi)
        sig_ln_by_case:    dict[(case, run)] -> np.array of sigma_lnphi
        meta_by_case:      dict[(case, run)] -> list[CalibPoint]
        metrics_by_case:   dict[(case, run)] -> list[dict(J_fit, err_abs, err_rel)]
    """
    invT_by_case, lnphi_by_case, sig_ln_by_case, meta_by_case, metrics_by_case = (
        {},
        {},
        {},
        {},
        {},
    )

    # Loop over each (case, run) pair
    case_run_pairs = sorted(
        {(p.case, p.run) for p in pts},
        key=lambda cr: (cr[0], cr[1]),
    )

    for case_name, run_name in case_run_pairs:
        invT_list, lnphi_list, sig_list, meta_rows, metrics_rows = [], [], [], [], []

        # Select all points for this (case, run)
        for p in [pp for pp in pts if pp.case == case_name and pp.run == run_name]:
            phi_T = _phi_match_exp_for_point(p, D_flibe, D_nickel, K_S_nickel)
            tiny = np.finfo(float).tiny

            def J_of_phi(phi_val: float) -> float:
                perm_loc = htm.Permeability(
                    pre_exp=float(phi_val), act_energy=0.0, law="henry"
                )
                out_bc_p = (
                    {"type": "sieverts", "pressure": p.P_gb}
                    if p.P_gb is not None
                    else {"type": "particle_flux_zero"}
                )
                out = run_once(
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
                return max(float(out), tiny)

            # modeled flux (at solved φ) and errors vs experiment
            J_fit = J_of_phi(phi_T)
            J_exp = max(float(p.J_exp), tiny)
            err_abs = J_fit - J_exp
            err_rel = err_abs / J_exp

            # propagate σ_J -> σ_lnφ
            sigma_J = get_exp_error(p.case, p.T_C, p.run)
            sigma_lnphi = None
            if sigma_J and np.isfinite(sigma_J) and sigma_J > 0.0:
                rel_step = 0.02
                delta = max(rel_step * float(phi_T), 1e-16)
                try:
                    Jp = J_of_phi(float(phi_T) + delta)
                    Jm = J_of_phi(max(float(phi_T) - delta, tiny))
                    dJ_dphi = (Jp - Jm) / (2.0 * delta)
                    sigma_lnphi = (sigma_J / max(abs(dJ_dphi), tiny)) / max(
                        float(phi_T), tiny
                    )
                    if (not np.isfinite(sigma_lnphi)) or (sigma_lnphi <= 0.0):
                        sigma_lnphi = None
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
                # console + log summary per (case, run)
                line = (
                    f"[{p.case} — {p.run}] T={p.T_C:.0f}°C: "
                    f"J_exp={J_exp:.3e}, J_fit={J_fit:.3e}, rel_err={err_rel * 100:.2f}%"
                )
                log_dir = Path("exports") / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / "fit_summary.txt"
                with log_file.open("a") as f:
                    f.write(line + "\n")

        key = (case_name, run_name)
        invT_by_case[key] = np.array(invT_list, float)
        lnphi_by_case[key] = np.array(lnphi_list, float)
        sig_ln_by_case[key] = np.array(sig_list, float)
        meta_by_case[key] = meta_rows
        metrics_by_case[key] = metrics_rows

    return (
        invT_by_case,
        lnphi_by_case,
        sig_ln_by_case,
        meta_by_case,
        metrics_by_case,
    )


def _fit_lnphi(invT, lnphi, sigma_ln=None):
    """
    Weighted least squares fit: lnφ = a + b * (1/T).
    Returns (a, b, Phi0, E, band) where band(x) -> (ylo, yhi) is a
    95% CI on the fitted mean ŷ(x) using WLS covariance.
    """
    x = np.asarray(invT, float)
    y = np.asarray(lnphi, float)

    # weights: default to 1; if sigma provided, use 1/sigma^2
    if sigma_ln is not None:
        s = np.asarray(sigma_ln, float)
        w = np.where(np.isfinite(s) & (s > 0.0), 1.0 / (s * s), 1.0)
    else:
        w = np.ones_like(x)

    # design matrix and weight matrix
    X = np.column_stack((np.ones_like(x), x))  # [1, x]
    W = np.diag(w)

    # beta = (X^T W X)^(-1) X^T W y
    XtW = X.T @ W
    XtWX = XtW @ X
    XtWy = XtW @ y
    XtWX_inv = np.linalg.pinv(XtWX)  # robust inverse
    beta = XtWX_inv @ XtWy
    a, b = float(beta[0]), float(beta[1])

    # Arrhenius parameters
    Phi0 = float(np.exp(a))
    E = float(-b * kB_eV)

    # residuals and weighted RSS
    r = y - (a + b * x)
    n, p = X.shape
    dof = max(n - p, 1)
    RSS_w = float(np.sum(w * r * r))
    s2 = RSS_w / dof  # variance estimate

    # covariance of coefficients (WLS)
    cov_beta = s2 * XtWX_inv

    def band(xq, z=1.96):
        xq = np.asarray(xq, float)
        yhat = a + b * xq
        v0 = np.ones_like(xq)
        v1 = xq
        # var(yhat) = [v]^T cov_beta [v], with v = [1, x]
        var = (
            cov_beta[0, 0] * v0 * v0
            + 2.0 * cov_beta[0, 1] * v0 * v1
            + cov_beta[1, 1] * v1 * v1
        )
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return yhat - z * std, yhat + z * std

    return a, b, Phi0, E, band


def _save_inverted_points_csv(outdir, rows):
    """
    rows: list of dicts with keys:
      case, run, T_C, T_K, invT, ln_phi, sigma_lnphi, phi, sigma_phi,
      J_exp, sigma_J, J_fit, err_abs, err_rel
    """
    outdir.mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = outdir / "inverted_points.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if _RANK0:
        print(f"[saved] {csv_path}")


def _save_fitted_params_csv(outdir, fit_info):
    """
    fit_info:
        either dict[case] -> dict(Phi0, E, R2)
        or     dict[(case, run)] -> dict(Phi0, E, R2)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = outdir / "fitted_params.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "run", "phi0", "E_eV", "R2"])
        w.writeheader()
        for key, v in fit_info.items():
            if isinstance(key, tuple):
                case_name, run_name = key
            else:
                case_name, run_name = key, ""  # backward compatibility
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
        print(f"[saved] {csv_path}")


def make_dual_overlay_lnphi(
    cases,
    T2K,
    Y_FT_BY_TEMP_C,
    D_flibe,
    D_nickel,
    K_S_nickel,
    outdir: Path,
    title_suffix="SWAP configuration",
    save_csv: bool = True,
):
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- collect & invert
    pts = _collect_points_for_cases(
        cases, T2K, Y_FT_BY_TEMP_C, allowed_case_names=list(cases.keys())
    )
    invT_by, lnphi_by, sig_by, meta_by, metrics_by = _invert_points_with_sigma(
        pts, D_flibe, D_nickel, K_S_nickel
    )

    # ---- style maps
    # list of unique case names
    case_names = sorted({case for (case, run) in invT_by.keys()})

    palette = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["C0", "C1", "C2"])
    bc_colors = {
        "swap_infinite": palette[0],
        "swap_transparent": palette[1],
    }

    # all run names
    runs_all = sorted({p.run for plist in meta_by.values() for p in plist})
    base_markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    run_marker = {
        r: base_markers[i % len(base_markers)] for i, r in enumerate(runs_all)
    }

    # ==== plotting section  ====
    fig, ax = plt.subplots(figsize=(8, 5.2))
    # Leave extra headroom for an outside legend + suptitle
    fig.subplots_adjust(top=0.80, bottom=0.25)

    # --- Plot points with error bars and run-dependent markers
    MS = 5
    CAP = MS
    ELW = 0.9
    MECW = 0.9

    # Each key here is (case_name, run_name)
    for (case_name, run_name), invT in invT_by.items():
        color = bc_colors.get(case_name, "C0")
        lnphi = lnphi_by[(case_name, run_name)]
        sig = sig_by[(case_name, run_name)]
        metas = meta_by[(case_name, run_name)]

        x = 1000 * invT  # scale to 1000 / K for better readability
        y = np.exp(lnphi)
        ysig = np.where(np.isfinite(sig) & (sig > 0), y * sig, 0.0)

        # vertical error bars
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
        # markers
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

    # --- Fit lines + dashed CI: now per (case, run)
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
        xx_plot = 1000 * xx  # scale to 1000 / K for better readability
        yy = fit_info[key]["a"] + fit_info[key]["b"] * xx
        yy_plot = np.exp(yy)
        lo, hi = fit_info[key]["band"](xx)
        plot_lo = np.exp(lo)
        plot_hi = np.exp(hi)
        c = bc_colors.get(case_name, "C0")

        ax.plot(
            xx_plot,
            yy_plot,
            color=c,
            lw=2,
            label=f"{case_name} — {run_name} fit",
            zorder=3.5,
        )
        ax.plot(xx_plot, plot_lo, color=c, lw=1.0, ls="--", alpha=0.8)
        ax.plot(xx_plot, plot_hi, color=c, lw=1.0, ls="--", alpha=0.8)

    # --- Axes, legends, and labels
    ax.set_xlabel("1000 / T [1/K]")
    ax.set_yscale("log")
    ax.set_ylabel("φ  [H·m⁻¹·s⁻¹·Pa⁻¹]")
    ax.grid(True, alpha=0.3)

    # Title placed as figure-level suptitle to avoid overlap with legend
    fig.suptitle(f"Pointwise inversion across BCs — {title_suffix}", y=0.98)

    # Legend handles:
    #  - one line style per case (color)
    #  - one marker style per run
    bc_handles = [
        Line2D(
            [0],
            [0],
            color=bc_colors.get(case, "C0"),
            lw=2,
            label=f"{case} fits",
        )
        for case in case_names
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
    handles = bc_handles + run_handles

    # Legend outside, above the axes (under the suptitle)
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=max(3, len(handles)),
        frameon=True,
        framealpha=0.95,
        columnspacing=1.4,
        handlelength=2.2,
        borderaxespad=0.6,
    )

    _ensure_and_save(fig, outdir / "dual_pointwise_lnphi_invT.png")

    # ---- CSV exports
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
        _save_inverted_points_csv(outdir, rows)
        _save_fitted_params_csv(
            outdir,
            {
                k: {"Phi0": v["Phi0"], "E": v["E"], "R2": v["R2"]}
                for k, v in fit_info.items()
            },
        )

    return fit_info


# ------------------------------ Main ------------------------------
if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)

    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")
    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]
    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

    temps_C_all = [500.0, 550.0, 600.0, 650.0, 700.0]
    T2K = {Tc: Tc + 273.15 for Tc in temps_C_all}
    Y_FT_BY_TEMP_C = {
        500.0: 0.02914,
        550.0: 0.02919,
        600.0: 0.02925,
        650.0: 0.02930,
        700.0: 0.02936,
    }

    # ---- SWAP input tables
    # Note: run 3 is for D2, so we won't use it for fitting H data, but we keep it here for completeness and potential future use.
    swap_infinite = {
        500.0: {
            "runs": {
                "Run 1": {"P_up": 1.31e5, "P_down": 1.77e1, "J_exp": 3.89e15},
                "Run 2": {"P_up": 1.31e5, "P_down": 1.99e1, "J_exp": 4.34e15},
                "Run 3": {"P_up": 1.31e5, "P_down": 8.66, "J_exp": 1.91e15},
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
                "Run 3": {"P_up": 1.33e5, "P_down": 2.10e1, "J_exp": 4.50e15},
            }
        },
        650.0: {
            "runs": {"Run 2": {"P_up": 1.32e5, "P_down": 5.02e1, "J_exp": 1.10e16}}
        },
        700.0: {
            "runs": {
                "Run 1": {"P_up": 1.32e5, "P_down": 4.07e1, "J_exp": 9.04e15},
                "Run 2": {"P_up": 1.32e5, "P_down": 4.78e1, "J_exp": 1.04e16},
                "Run 3": {"P_up": 1.31e5, "P_down": 3.230e1, "J_exp": 7.12e15},
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
                    "P_down": 1.99e1,
                    "P_gb": 7.0,
                    "J_exp": 4.34e15,
                },
                "Run 3": {
                    "P_up": 1.31e5,
                    "P_down": 8.66,
                    "P_gb": 7.0,
                    "J_exp": 1.91e15,
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
                "Run 3": {
                    "P_up": 1.33e5,
                    "P_down": 2.10e1,
                    "P_gb": 1.2e1,
                    "J_exp": 4.50e15,
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
                "Run 3": {
                    "P_up": 1.31e5,
                    "P_down": 3.230e1,
                    "P_gb": 2.2e1,
                    "J_exp": 7.12e15,
                },
            }
        },
    }

    cases = {
        "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }
    outdir = Path("exports") / "figs" / "calibration_A" / "SWAP_bundle"
    fit_params = make_dual_overlay_lnphi(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=htm.Diffusivity(D_0=2.5e-7, E_D=0.24),
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        outdir=outdir,
        title_suffix="SWAP configuration",
        save_csv=True,
    )

    if _RANK0:
        print("Fitted (Phi0, E, R2) per BC:")
        for k, v in fit_params.items():
            print(f"  {k}: Phi0={v['Phi0']:.3e}, E={v['E']:.4f} eV, R2={v['R2']:.4f}")

    if (not _DEFER_SHOW) and ("agg" not in matplotlib.get_backend().lower()):
        plt.show()
