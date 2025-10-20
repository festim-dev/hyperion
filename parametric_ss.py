import festim as F
from mesh import generate_mesh, set_y_ft
from dolfinx.log import set_log_level, LogLevel
from cylindrical_flux import CylindricalFlux
from dolfinx.io import gmshio
from mpi4py import MPI
import h_transport_materials as htm
from typing import Literal, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


from dataclasses import dataclass
from typing import List, Dict
import math
import matplotlib
from typing import Optional


_DEFER_SHOW = True 


def make_materials(D_solid, D_liquid, K_solid, permeability_liquid):
    # material parameters for solid
    D_0_solid = D_solid.pre_exp.magnitude  # m^2/s
    E_D_solid = D_solid.act_energy.magnitude  # ev/particle
    K_S_0_solid = K_solid.pre_exp.magnitude  # particle m^-3 Pa^-0.5
    E_K_S_solid = K_solid.act_energy.magnitude  # ev/particle

    # material parameters for liquid
    D_0_liquid = D_liquid.pre_exp.magnitude  # m^2/s
    E_D_liquid = D_liquid.act_energy.magnitude  # ev/particle

    K_S_liquid = htm.Solubility(
        S_0=permeability_liquid.pre_exp / D_liquid.pre_exp,
        E_S=permeability_liquid.act_energy - D_liquid.act_energy,
        law=permeability_liquid.law,
    )

    K_S_0_liquid = K_S_liquid.pre_exp.magnitude  # particle m^-3 Pa^-1
    E_K_S_liquid = K_S_liquid.act_energy.magnitude  # ev/particle

    # Define materials
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

_mesh_cache = {}

def load_or_make_mesh(mesh_file, mesh_size, model_rank=0):
    """
    Load an existing mesh or create it once and reuse.
    This prevents new MPI communicators from being created every run.
    """
    # If mesh file does not exist yet, generate it
    if not Path(mesh_file).exists():
        generate_mesh(mesh_size=mesh_size, fname=mesh_file)

    # If the mesh is already cached in memory, reuse it
    if mesh_file in _mesh_cache:
        return _mesh_cache[mesh_file]

    # Otherwise, read from disk and store in cache
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        mesh_file, MPI.COMM_WORLD, model_rank
    )
    _mesh_cache[mesh_file] = (mesh, cell_tags, facet_tags)
    return _mesh_cache[mesh_file]


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
    if y_ft is not None:
        set_y_ft(y_ft)
    else:
        raise ValueError("y_ft must be provided")

    # --- Record each y_ft value to a separate log file to examine correctness ---
    Path("exports/yft_record").mkdir(parents=True, exist_ok=True)
    with open("exports/yft_record/y_ft_values.txt", "a") as f:
        f.write(f"{y_ft:.5f}\n")

    # --- Generate mesh file only if missing
    mesh_file = f"mesh_{y_ft:.5f}.msh"
    generate_mesh(mesh_size=mesh_size, fname=mesh_file)

    # --- Load and reuse mesh from cache (avoids repeated communicator creation)
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

    # recompute liquid solubility (Henry) parameters for boundary conditions
    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,
    )
    H_0_liq = K_S_liquid.pre_exp.magnitude
    E_H_liq = K_S_liquid.act_energy.magnitude
    K_S_0_Ni = K_S_nickel.pre_exp.magnitude
    E_S_Ni = K_S_nickel.act_energy.magnitude

    # --- subdomains ---
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

    # --- choose which surfaces are upstream vs downstream for each case ---
    # Rule wanted:
    #   - normal_* : liquid belongs to downstream
    #   - swap_*   : liquid belongs to upstream

    if case in ("normal_infinite", "normal_transparent"):
        # BCs: liquid is downstream -> Henry at P_down
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
        # BCs: liquid is upstream -> Henry at P_up
        upstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_up, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [top_cap_Ni, top_sidewall_Ni]
        ] + [
            F.HenrysBC(
                subdomain=liquid_surface, species=H, pressure=P_up, H_0=H_0_liq, E_H=E_H_liq
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

    # optional out-surface BC (glovebox). Supported:
    #   {"type": "none"}
    #   {"type": "sieverts", "pressure": <Pa>}
    #   {"type": "particle_flux_zero"}
    out_bcs = []
    if out_bc is None:
        out_bc = {"type": "none"}
    keywords = out_bc.get("type", "none").lower()
    if keywords == "sieverts":
        out_bcs = [
            F.SievertsBC(
                subdomain=out_surf,
                species=H,
                pressure=float(out_bc.get("pressure", 0.0)),
                S_0=K_S_0_Ni,
                E_S=E_S_Ni,
            )
        ]
    elif keywords == "particle_flux_zero":
        out_bcs = [F.ParticleFluxBC(subdomain=out_surf, species=H, value=0.0)]

    my_model.boundary_conditions = upstream_bcs + downstream_bcs + out_bcs
    my_model.settings = F.Settings(atol=1e12, rtol=1e-13, transient=False)

    # -------- flux monitors (register each surface explicitly) --------
    # Six monitored faces: three upstream, three downstream
    # A = liquid-side surfaces, B = solid-side surfaces
    flux_out_top_cap_Ni = CylindricalFlux(field=H, surface=top_cap_Ni)  # A
    flux_out_top_sidewall_Ni = CylindricalFlux(field=H, surface=top_sidewall_Ni)  # A
    flux_out_liquid_surface = CylindricalFlux(field=H, surface=liquid_surface)  # A
    flux_out_mid_membrane_Ni = CylindricalFlux(field=H, surface=mid_membrane_Ni)  # B
    flux_out_bottom_sidewall_Ni = CylindricalFlux(field=H, surface=bottom_sidewall_Ni)  # B
    flux_out_bottom_cap_Ni = CylindricalFlux(field=H, surface=bottom_cap_Ni)  # B

    # glovebox outlet (external surface)
    glovebox_flux = CylindricalFlux(field=H, surface=out_surf)

    # register fluxes into model exports
    my_model.exports = [
        flux_out_top_cap_Ni,
        flux_out_top_sidewall_Ni,
        flux_out_bottom_sidewall_Ni,
        flux_out_liquid_surface,
        flux_out_mid_membrane_Ni,
        flux_out_bottom_cap_Ni,
        glovebox_flux,
    ]

    # explicit label-to-object mapping for easy reference
    flux_by_label = {
        "top_cap_Ni": flux_out_top_cap_Ni,
        "top_sidewall_Ni": flux_out_top_sidewall_Ni,
        "liquid_surface": flux_out_liquid_surface,
        "mid_membrane_Ni": flux_out_mid_membrane_Ni,
        "bottom_sidewall_Ni": flux_out_bottom_sidewall_Ni,
        "bottom_cap_Ni": flux_out_bottom_cap_Ni,
    }

    # list of labels for plotting and output order
    six_labels = [
        "top_cap_Ni",
        "top_sidewall_Ni",
        "liquid_surface",
        "mid_membrane_Ni",
        "bottom_sidewall_Ni",
        "bottom_cap_Ni",
    ]

    # upstream/downstream definitions depend on case type
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

    # package all flux information into dictionary
    fluxes_dict = {
        "flux_by_label": flux_by_label,  # label -> CylindricalFlux
        "six_labels": six_labels,  # list of monitored faces
        "glovebox_flux": glovebox_flux,  # external flux
        "up_labels": up_labels,
        "down_labels": down_labels,
    }

    return my_model, fluxes_dict


def _get_flux_value(flux_obj) -> float:
    # Prefer sampled data; fall back to .value; otherwise 0.0
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
    Solve steady-state once and return all relevant flux numbers for plotting.
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
    my_model.run()  # steady state

    # per-face values (only the six faces)
    flux_objects = fluxes_dict["flux_by_label"]  # label -> CylindricalFlux
    six_labels = fluxes_dict["six_labels"]  # fixed order of the six faces
    vals_six = {label: _get_flux_value(flux_objects[label]) for label in six_labels}

    # glovebox (out surface)
    glovebox_val = float(_get_flux_value(fluxes_dict["glovebox_flux"]))

    # totals: simple sum of their own values
    up_labels = fluxes_dict["up_labels"]
    down_labels = fluxes_dict["down_labels"]

    # sum over the upstream group
    total_up = float(np.sum([vals_six[label] for label in up_labels], dtype=float))
    # sum over the downstream group
    total_down = float(np.sum([vals_six[label] for label in down_labels], dtype=float))

    display_names = {
        "top_cap_Ni": "Top cap (Ni)",
        "top_sidewall_Ni": "Upper Ni sidewall",
        "liquid_surface": "FLiBe surface",
        "mid_membrane_Ni": "Middle membrane",
        "bottom_cap_Ni": "Bottom Ni (top face)",
        "bottom_sidewall_Ni": "Lower Ni sidewall",
    }

    # per-surface fluxes
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

    return dict(
        total_in=total_up,  
        glovebox=glovebox_val,  
        total_out=total_down,  
        balance=total_up + glovebox_val + total_down,
        per_surface=per_surface,
    )

# ================== Scheme A: single-pass scaling + Arrhenius smoothing ==================
kB_eV = 8.617333262e-5  # eV/K


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


def _ensure_fig_dir(dirpath: Path) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)


def _phi_guess_at_T(permeability_flibe, T_K: float) -> float:
    """Evaluate guess FLiBe permeability at temperature T_K."""
    try:
        return float(permeability_flibe.value(T_K).magnitude)
    except Exception:
        return float(permeability_flibe.pre_exp) * math.exp(
            -float(permeability_flibe.act_energy) / (kB_eV * float(T_K))
        )


def _collect_points_for_cases(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    allowed_case_names: Optional[List[str]] = None,
) -> List[CalibPoint]:
    """
    Build a list of CalibPoint objects from the `cases` configuration.
    """
    pts: List[CalibPoint] = []
    for case_name, cfg in cases.items():
        if allowed_case_names is not None and case_name not in allowed_case_names:
            continue
        is_normal = case_name.startswith("normal")
        is_swap = case_name.startswith("swap")
        if not (is_normal or is_swap):
            continue
        table = cfg["table"]
        for Tc, row in table.items():
            Tk = T2K[Tc]
            yft = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])
            runs = row.get("runs", {})
            if "Run 1" not in runs:
                continue
            cond = runs["Run 1"]
            pts.append(
                CalibPoint(
                    case=case_name,
                    T_C=float(Tc),
                    T_K=float(Tk),
                    run="Run 1",
                    P_up=float(cond["P_up"]),
                    P_down=float(cond["P_down"]),
                    P_gb=float(cond.get("P_gb")) if "P_gb" in cond else None,
                    y_ft=float(yft),
                    J_exp=float(cond["J_exp"]),
                )
            )
    if not pts:
        raise RuntimeError("No calibration points found.")
    pts.sort(key=lambda p: (p.case, p.T_C))
    return pts

def _ensure_fig_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _ensure_and_save(fig, path: Path, dpi=220):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def _arrhenius_str(Phi0: float, E_eV: float, unit="H·m⁻¹·s⁻¹·Pa⁻¹") -> str:
    """
    Produce a neat Arrhenius law string for annotation.
    φ(T) = Φ0 · exp(-E/(k_B T))
    """
    return (r"$\varphi(T)$ = "
            f"{Phi0:.4g} · exp(" +
            r"$-\dfrac{" + f"{E_eV:.4g}" + r"\ \mathrm{eV}}{k_B T}$" +
            f")  [{unit}]")

# ---------- figure folders + saving---------
def _ensure_and_save(fig, out_path: Path) -> None:
    # ensure parent exists, save, print path for debugging, and close
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"[saved] {out_path}")
    plt.close(fig)


def save_overview(fig, case: str, name: str = "overview"):
    out = Path("exports") / "figs" / case / f"{name}.png"
    _ensure_and_save(fig, out)


def fig_saving(case_name: str, T_C: int, run_name: str) -> Path:
    return Path("exports") / "figs" / "calibration_A" / case_name / f"{T_C}C" / run_name

def save_breakdown(fig, outdir: Path, stem: str):
    """Save breakdown figure with a stable naming convention."""
    _ensure_and_save(fig, outdir / f"{stem}.png")

# ==== Calibration + plotting==================================================

def calibrate_phi_schemeA(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    D_nickel,
    K_S_nickel,
    permeability_flibe_guess,
    # plotting/output control
    outdir: Path = Path("exports") / "figs" / "calibration_A",
    also_show: bool = False,
    allowed_case_names: Optional[List[str]] = None,
):
    """
    Scheme A (per-case when allowed_case_names is given):
      1) Run once with guess permeability (permeability_flibe_guess).
      2) Scale at each point: s = J_exp / J_model; set φ_new(T) = s * φ_guess(T).
      3) Fit ln φ_new vs 1/T (Arrhenius) -> (Φ0, E).
      4) Re-run with fitted (Φ0, E); produce a parity plot.
      5) Write the explicit Arrhenius expression onto the ln-φ plot and parity plot.
      6) Persist results under a case-scoped folder, tied to the case name.
      7) Auto-generate per-temperature bar charts (exp vs model in/out + glovebox + six surfaces) using the fitted φ.
    """
    # Derive case suffix and output folder
    case_suffix = None if (allowed_case_names is None) else allowed_case_names[0]
    outdir = outdir if case_suffix is None else (outdir / case_suffix)
    _ensure_fig_dir(outdir)

    # Collect Run 1 points from all/specified cases
    pts = _collect_points_for_cases(cases, T2K, Y_FT_BY_TEMP_C, allowed_case_names)

    ln_phi_new_list, invT_list, rows = [], [], []

    # 1–2) Single-pass scaling using the guess permeability
    for p in pts:
        out_bc = {"type": "sieverts", "pressure": p.P_gb} if p.P_gb is not None else {"type": "particle_flux_zero"}
        out = run_once(
            case=p.case,
            T_K=p.T_K,
            P_up=p.P_up,
            P_down=p.P_down,
            D_flibe=D_flibe,
            D_nickel=D_nickel,
            permeability_flibe=permeability_flibe_guess,
            K_S_nickel=K_S_nickel,
            out_bc=out_bc,
            y_ft=p.y_ft,
        )
        eps = np.finfo(float).tiny
        J_model = max(float(out["total_out"]), eps)
        J_exp = max(float(p.J_exp), eps)

        s = J_exp / J_model
        phi_guess_T = _phi_guess_at_T(permeability_flibe_guess, p.T_K)
        phi_new_T = max(s * phi_guess_T, eps)

        ln_phi_new_list.append(math.log(phi_new_T))
        invT_list.append(1.0 / p.T_K)
        rows.append(
            {
                "case": p.case,
                "T_C": p.T_C,
                "run": p.run,
                "J_guess": J_model,
                "J_exp": J_exp,
                "scale_s": s,
                "phi_guess_T": phi_guess_T,
                "phi_new_T": phi_new_T,
            }
        )

    ln_phi_new = np.array(ln_phi_new_list, dtype=float)
    invT = np.array(invT_list, dtype=float)

    # 3) Arrhenius fit in ln-space:  ln φ = a + b * (1/T)
    b, a = np.polyfit(invT, ln_phi_new, deg=1)
    Phi0_hat = float(math.exp(a))
    E_hat = float(-b * kB_eV)  # keep unit convention

    # simple R^2 in ln-space
    y_pred = b * invT + a
    ss_res = float(np.sum((ln_phi_new - y_pred) ** 2))
    ss_tot = float(np.sum((ln_phi_new - float(np.mean(ln_phi_new))) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n[Scheme A] Fitted Arrhenius parameters from scaled points:")
    print(f"  Phi0 = {Phi0_hat:.4e}")
    print(f"  E    = {E_hat:.4f} eV")
    print(f"  R^2  = {R2:.4f} (fit in ln(phi) vs 1/T)")


    # 4) Validation with fitted permeability + parity plot
    perm_fitted = htm.Permeability(pre_exp=Phi0_hat, act_energy=E_hat, law="henry")

    print("\n[Scheme A] Validation with fitted (Phi0, E):")
    print("Case | T[°C] | Run |   J_model_fit [H/s]   |    J_exp [H/s]     |  log10 err")
    jm_fit_list, je_list, labels = [], [], []

    for p in pts:
        out_bc = {"type": "sieverts", "pressure": p.P_gb} if p.P_gb is not None else {"type": "particle_flux_zero"}
        out = run_once(
            case=p.case,
            T_K=p.T_K,
            P_up=p.P_up,
            P_down=p.P_down,
            D_flibe=D_flibe,
            D_nickel=D_nickel,
            permeability_flibe=perm_fitted,
            K_S_nickel=K_S_nickel,
            out_bc=out_bc,
            y_ft=p.y_ft,
        )
        eps = np.finfo(float).tiny
        jm = max(float(out["total_out"]), eps)
        je = max(float(p.J_exp), eps)
        jm_fit_list.append(jm)
        je_list.append(je)
        labels.append(f"{p.case}@{int(p.T_C)}°C")
        rlog = math.log10(jm) - math.log10(je)
        print(f"{p.case:<18} | {int(p.T_C):>5} | {p.run:<4} | {jm:>20.3e} | {je:>18.3e} | {rlog:>9.3f}")

    # --------- Plots (save; only show at the very end) ---------
    _ensure_fig_dir(outdir)
    title_suffix = " (all)" if case_suffix is None else f" ({case_suffix})"

    # ln(phi_new) vs 1/T with explicit formula annotation
    fig1, ax1 = plt.subplots(figsize=(6.6, 4.8))
    ax1.scatter(invT, ln_phi_new, label="Scaled points (ln φ_new)", zorder=3)
    order = np.argsort(invT)
    ax1.plot(invT[order], y_pred[order], linestyle="--", label="Linear fit")
    ax1.set_xlabel("1 / T  [1/K]")
    ax1.set_ylabel("ln φ  [ln(H·m⁻¹·s⁻¹·Pa⁻¹)]")
    expr_line = _arrhenius_str(Phi0_hat, E_hat)
    fig1.suptitle(expr_line + f"   $R^2={R2:.4f}$", y=0.98, fontsize=10)
    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    ax1.set_title("Scheme A: ln(φ_new) vs 1/T" + title_suffix)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    _ensure_and_save(fig1, outdir / "schemeA_lnphi_vs_invT.png")

    # Parity plot (experimental vs model with fitted permeability)
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
    ax2.set_title("Scheme A: Parity" + title_suffix)
    fig2.suptitle(expr_line, y=0.98, fontsize=10)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    for x, y, lab in zip(je_arr, jm_arr, labels):
        ax2.text(x, y, lab, fontsize=8)
    fig2.tight_layout()
    _ensure_and_save(fig2, outdir / "schemeA_parity.png")

    # 7) Auto-generate bar charts for case using the fitted permeability
    if case_suffix is not None:
        case_cfg = cases[case_suffix]
        plot_case_breakdowns_with_exp(
            case_name=case_suffix,
            case_cfg=case_cfg,
            T2K=T2K,
            Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
            D_flibe=D_flibe,
            permeability_flibe_fitted=perm_fitted,
            outdir_root=outdir.parent if outdir.name == case_suffix else outdir.parent,  # keep same root
        )

    if also_show and "agg" not in matplotlib.get_backend().lower():
        plt.show()

    return {
        "phi0": Phi0_hat,
        "E": E_hat,
        "R2": R2,
        "points": rows,
        "fig_dir": str(outdir),
        "permeability": perm_fitted,  # fitted curve for downstream use
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
    """
    Calibrate (Φ0, E) for each case separately (Run 1 only).
    Outputs are written to: outdir_root/<case_name>/...
    Returns {case_name: result_dict}, where result_dict contains the fitted permeability.
    """
    results: Dict[str, dict] = {}
    for case_name in cases.keys():
        res = calibrate_phi_schemeA(
            cases=cases,
            T2K=T2K,
            Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
            D_flibe=D_flibe,
            D_nickel=D_nickel,
            K_S_nickel=K_S_nickel,
            permeability_flibe_guess=permeability_flibe_guess,
            outdir=outdir_root,
            also_show=False,
            allowed_case_names=[case_name],  # per-case fit
        )
        results[case_name] = res
    return results


def plot_case_breakdowns_with_exp(
    case_name: str,
    case_cfg: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    permeability_flibe_fitted,
    outdir_root: Path = Path("exports") / "figs" / "calibration_A",
):
    """
    For a single case, loop over its temperatures and runs; re-run with the
    case-specific fitted permeability and produce a bar chart comparing:
      - Flux exp (J_exp)
      - Flux in  (total_in)
      - Flux out (total_out)
      - Glovebox flux (glovebox)
      - Per-surface fluxes (six faces, in per_surface['labels'] order)

    Each bar is annotated with its numeric value. The figure also displays
    the current temperature, y_ft, and the boundary-condition parameters used.

    Keep numeric labels above every bar.
    """
    table = case_cfg.get("table", {})
    case_outdir = outdir_root / case_name
    _ensure_fig_dir(case_outdir)

    for Tc in sorted(table.keys()):
        Tk = T2K[Tc]
        y_ft_val = Y_FT_BY_TEMP_C.get(Tc, list(Y_FT_BY_TEMP_C.values())[-1])

        runs = table[Tc].get("runs", {})
        for run_name, cond in runs.items():
            out_bc = (
                {"type": "sieverts", "pressure": float(cond.get("P_gb", 0.0))}
                if "P_gb" in cond
                else {"type": "particle_flux_zero"}
            )

            res = run_once(
                case=case_name,
                T_K=Tk,
                P_up=float(cond["P_up"]),
                P_down=float(cond["P_down"]),
                D_flibe=D_flibe,
                D_nickel=D_nickel,
                K_S_nickel=K_S_nickel,
                permeability_flibe=permeability_flibe_fitted,
                out_bc=out_bc,
                y_ft=float(y_ft_val),
            )

            # Pack bar labels/values: [exp, in, out] + six surfaces
            per = res["per_surface"]
            six_labels = per["labels"]
            six_values = [float(v) for v in per["values"]]
            values = [float(cond["J_exp"]), float(res["total_in"]), float(res["total_out"]), float(res["glovebox"]),] + six_values
            labels = ["Flux exp", "Flux in", "Flux out", "Glovebox"] + list(six_labels)

            fig, ax = plt.subplots(figsize=(13, 6))
            x = np.arange(len(labels), dtype=float)
            bars = ax.bar(x, np.abs(values))

            # annotate numerics on top of bars
            for bar, v in zip(bars, values):
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    (h if h > 0 else 0.0) * 1.02 + (1e-30 if h == 0 else 0),  # keep text above even if tiny
                    f"{v:.2e}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.set_ylabel("Flux [H/s]")
            ax.set_title(f"{case_name} — {run_name} — {int(Tc)} °C  (y_ft={float(y_ft_val):.5f} m)")

            # explicit conditions box
            info = (
                f"Case: {case_name}\nRun: {run_name}\n"
                f"T = {int(Tc)} °C (T_K={Tk:.2f})\n"
                f"y_ft = {float(y_ft_val):.5f} m\n"
                f"P_up = {float(cond['P_up']):.2e} Pa\n"
                f"P_down = {float(cond['P_down']):.2e} Pa\n"
                f"P_glovebox = {float(cond['P_gb']):.2e} Pa"
                if "P_gb" in cond
                else f"Case: {case_name}\nRun: {run_name}\n"
                     f"T = {int(Tc)} °C (T_K={Tk:.2f})\n"
                     f"y_ft = {float(y_ft_val):.5f} m\n"
                     f"P_up = {float(cond['P_up']):.2e} Pa\n"
                     f"P_down = {float(cond['P_down']):.2e} Pa\n"
                     f"P_glovebox = (closed)"
            )
            ax.text(
                0.99, 0.98, info,
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.85, lw=0.5),
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            fig.tight_layout()

            fd = fig_saving(case_name, int(Tc), run_name)
            save_breakdown(fig, fd, stem=f"{case_name}_{int(Tc)}C_{run_name}_breakdown_exp_vs_model")


def plot_flux_breakdown(
    case_name: str,
    case_cfg: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    permeability_flibe_fitted,
    outdir_root: Path = Path("exports") / "figs" / "calibration_A",
):
    """Compatibility wrapper: old name -> new implementation."""
    return plot_case_breakdowns_with_exp(
        case_name=case_name,
        case_cfg=case_cfg,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        permeability_flibe_fitted=permeability_flibe_fitted,
        outdir_root=outdir_root,
    )


# ----------------------- main -----------------------
if __name__ == "__main__":
    set_log_level(LogLevel.WARNING)

    # materials
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")
    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)
    permeability_flibe = htm.Permeability(pre_exp=2.0e13, act_energy=0.44, law="henry")

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

    # ---- experiment/BC tables for the 4 cases ----
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
            "runs": {"Run 1": {"P_up": 1.31e5, "P_down": 1.77e1, "J_exp": 3.89e15}}
        },
        550.0: {
            "runs": {"Run 1": {"P_up": 1.31e5, "P_down": 3.21e1, "J_exp": 7.20e15}}
        },
        600.0: {
            "runs": {"Run 1": {"P_up": 1.33e5, "P_down": 3.57e1, "J_exp": 7.64e15}}
        },
        700.0: {
            "runs": {"Run 1": {"P_up": 1.32e5, "P_down": 4.07e1, "J_exp": 9.04e15}}
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
                }
            }
        },
        550.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.31e5,
                    "P_down": 3.21e1,
                    "P_gb": 1.00e1,
                    "J_exp": 7.20e15,
                }
            }
        },
        600.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.33e5,
                    "P_down": 3.57e1,
                    "P_gb": 1.20e1,
                    "J_exp": 7.64e15,
                }
            }
        },
        700.0: {
            "runs": {
                "Run 1": {
                    "P_up": 1.32e5,
                    "P_down": 4.07e1,
                    "P_gb": 2.20e1,
                    "J_exp": 9.04e15,
                }
            }
        },
    }

   # Although the code is designed to process all four cases at once, possibly due to MPI memory issues (not fully confirmed), it is safer to run only one case at a time.
    cases = {
        # "normal_infinite": {"table": normal_infinite, "out_mode": "particle_flux_zero"},
        # "normal_transparent": {"table": normal_transparent, "out_mode": "sieverts"},
        # "swap_infinite": {"table": swap_infinite, "out_mode": "particle_flux_zero"},
        "swap_transparent": {"table": swap_transparent, "out_mode": "sieverts"},
    }

    def phi_at_T(T_K: float) -> float:
        try:
            return float(permeability_flibe.value(T_K).magnitude)
        except Exception:
            kB_eV = 8.617333262e-5
            return float(
                permeability_flibe.pre_exp
                * np.exp(-permeability_flibe.act_energy / (kB_eV * T_K))
            )

    # ==================== PER-CASE CALIBRATION (each case its own Φ) ====================
    result_by_case = calibrate_phi_all_cases(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        permeability_flibe_guess=permeability_flibe,
        outdir_root=Path("exports") / "figs" / "calibration_A",
    )
    print("Per-case fitted parameters:")
    for k, v in result_by_case.items():
        print(k, {kk: v[kk] for kk in ("phi0", "E", "R2", "fig_dir")})

    # ==================== RUN ALL CASES & COLLECT ====================
    all_results = {}  # case -> Tc -> run -> {model, exp, conds}
    case_summaries = {}  # arrays for overview plots
    for case_name, cfg in cases.items():
        perm_use = result_by_case.get(case_name, {}).get("permeability", permeability_flibe)
        table = cfg["table"]
        out_mode = cfg["out_mode"]

        run_names = sorted({r for Tc in table for r in table[Tc]["runs"].keys()})
        totals_by_run = {r: [] for r in run_names}
        exp_by_run = {r: [] for r in run_names}
        xticks = []

        print(f"\n=== {case_name} ===")
        header = "Temp(°C) | Run   | Model [H/s] | Exp [H/s] | Rel. err (%)"
        print(header)
        print("-" * len(header))

        all_results[case_name] = {}

        for Tc in sorted(table.keys()):
            Tk = T2K[Tc]
            xticks.append(Tc)
            all_results[case_name][Tc] = {}

            for r in run_names:
                if r not in table[Tc]["runs"]:
                    totals_by_run[r].append(np.nan)
                    exp_by_run[r].append(np.nan)
                    continue

                row = table[Tc]["runs"][r]
                P_up = row["P_up"]
                P_down = row["P_down"]
                if out_mode == "sieverts":
                    out_bc = {
                        "type": "sieverts",
                        "pressure": float(row.get("P_gb", 0.0)),
                    }
                else:
                    out_bc = {"type": "particle_flux_zero"}

                y_ft_val = Y_FT_BY_TEMP_C.get(Tc, Y_FT_BY_TEMP_C[700.0])
                res = run_once(
                    case=case_name,
                    T_K=Tk,
                    P_up=P_up,
                    P_down=P_down,
                    D_flibe=D_flibe,
                    D_nickel=D_nickel,
                    permeability_flibe=perm_use,
                    K_S_nickel=K_S_nickel,
                    out_bc=out_bc,
                    y_ft=y_ft_val,
                )

                all_results[case_name][Tc][r] = {
                    "model": res,
                    "exp": row["J_exp"],
                    "conds": {
                        "case": case_name,
                        "run": r,
                        "T_C": Tc,
                        "P_up": P_up,
                        "P_down": P_down,
                        "P_gb": row.get("P_gb", None),
                        "y_ft": y_ft_val,
                        "phi": phi_at_T(Tk),
                    },
                }

                model_total = res["total_out"]
                exp_total = row["J_exp"]
                rel_err = (
                    100.0 * (model_total - exp_total) / exp_total
                    if exp_total
                    else np.nan
                )
                print(
                    f"{int(Tc):>8} | {r:<5} | {model_total:>12.3e} | {exp_total:>9.3e} | {rel_err:>11.2f}"
                )

                totals_by_run[r].append(model_total)
                exp_by_run[r].append(exp_total)

        case_summaries[case_name] = dict(
            run_names=run_names,
            totals_by_run=totals_by_run,
            exp_by_run=exp_by_run,
            xticks=xticks,
        )

    # --- helper: Ni (Sieverts side) permeability at T ---
    def pi_ni_at_T(T_K: float) -> float:
        """
        Simple Ni permeability Φ(T). Default: Φ = D * K_S  (Sieverts side).
        """
        kB_eV = 8.617333262e-5
        try:
            D_val = float(D_nickel.value(T_K).magnitude)
        except Exception:
            D_val = float(D_nickel.pre_exp) * np.exp(
                -float(D_nickel.act_energy) / (kB_eV * T_K)
            )
        try:
            Ks_val = float(K_S_nickel.value(T_K).magnitude)
        except Exception:
            Ks_val = float(K_S_nickel.pre_exp) * np.exp(
                -float(K_S_nickel.act_energy) / (kB_eV * T_K)
            )
        return D_val * Ks_val

    # ==================== OVERVIEW: per-case model vs exp ====================
    for case_name, summary in case_summaries.items():
        run_names = summary["run_names"]
        totals_by_run = summary["totals_by_run"]
        exp_by_run = summary["exp_by_run"]
        xticks = summary["xticks"]

        x = np.arange(len(xticks), dtype=float)
        group_w = 0.8
        n_series = max(2, len(run_names))
        w = group_w / (2 * n_series)

        figA, axA = plt.subplots(figsize=(13, 6))
        shifts = np.linspace(-group_w / 2 + w, group_w / 2 - w, len(run_names))

        for s, r in zip(shifts, run_names):
            bars_exp = axA.bar(
                x + s - w / 2,
                exp_by_run[r],
                width=w,
                label=f"Exp {r}",
                edgecolor="black",
                fill=False,
            )
            bars_mod = axA.bar(
                x + s + w / 2,
                totals_by_run[r],
                width=w,
                label=f"Model {r}",
            )
            def _annot(container, values):
                for rect, val in zip(container, values):
                    if np.isnan(val):
                        continue
                    axA.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_height(),
                        f"{val:.2e}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            _annot(bars_exp, exp_by_run[r])
            _annot(bars_mod, totals_by_run[r])

        xticklabels = []
        for T in xticks:
            Tk = T2K[T]
            phi_flibe = phi_at_T(Tk)
            pi_ni = pi_ni_at_T(Tk)
            yft = Y_FT_BY_TEMP_C.get(T, np.nan)
            xticklabels.append(
                f"{int(T)}°C\nΦ_FLiBe={phi_flibe:.2e}\nΦ_Ni={pi_ni:.2e}\n" f"y_ft={yft:.5f} m"
            )

        axA.set_xticks(x)
        axA.set_xticklabels(xticklabels)
        axA.set_ylabel("Total downstream flux [H/s]")
        axA.set_title(f"{case_name}: model vs experiment")
        axA.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axA.legend(ncol=max(2, len(run_names)))
        figA.tight_layout()
        save_overview(figA, case_name, name="overview")  # prints path and closes


    # finally, pop all windows *once* (if not headless/agg)
    if (not _DEFER_SHOW) and ("agg" not in matplotlib.get_backend().lower()):
        plt.show()
