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


# ---------- figure folders + saving---------
def fig_dir_for(case: str, Tc_C: int, run_label: str) -> Path:
    safe_run = run_label.replace(" ", "")
    d = Path("exports") / "figs" / case / f"{Tc_C}C" / safe_run
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_overview(fig, case: str, name: str = "overview"):
    out = Path("exports") / "figs" / case
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=220, bbox_inches="tight")


def save_breakdown(fig, fig_dir: Path, stem: str):
    fig.savefig(fig_dir / f"{stem}.png", dpi=220, bbox_inches="tight")


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

    generate_mesh(mesh_size=mesh_size)
    model_rank = 0
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        "mesh.msh", MPI.COMM_WORLD, model_rank
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
    top_Ni_bottom = F.SurfaceSubdomain(id=5)
    Ds_Ni_left = F.SurfaceSubdomain(id=6)
    Up_Ni_left = F.SurfaceSubdomain(id=7)
    Liquid_top = F.SurfaceSubdomain(id=8)
    mem_Ni_bottom = F.SurfaceSubdomain(id=9)
    bottom_Ni_top = F.SurfaceSubdomain(id=10)
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
        top_Ni_bottom,
        Ds_Ni_left,
        Up_Ni_left,
        Liquid_top,
        mem_Ni_bottom,
        bottom_Ni_top,
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
        top_Ni_bottom: solid_volume,
        Ds_Ni_left: solid_volume,
        Up_Ni_left: solid_volume,
        Liquid_top: fluid_volume,
        mem_Ni_bottom: solid_volume,
        bottom_Ni_top: solid_volume,
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
            for s in [mem_Ni_bottom, bottom_Ni_top, Up_Ni_left]
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [top_Ni_bottom, Ds_Ni_left]
        ] + [
            F.HenrysBC(
                subdomain=Liquid_top,
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
            for s in [top_Ni_bottom, Ds_Ni_left]
        ] + [
            F.HenrysBC(
                subdomain=Liquid_top, species=H, pressure=P_up, H_0=H_0_liq, E_H=E_H_liq
            )
        ]
        downstream_bcs = [
            F.SievertsBC(
                subdomain=s, species=H, pressure=P_down, S_0=K_S_0_Ni, E_S=E_S_Ni
            )
            for s in [mem_Ni_bottom, bottom_Ni_top, Up_Ni_left]
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

    # -------- flux monitors (register each surface explicitly) --------
    # Six monitored faces: three upstream, three downstream
    # A = liquid-side surfaces, B = solid-side surfaces
    flux_out_top_Ni_bottom = CylindricalFlux(field=H, surface=top_Ni_bottom)  # A
    flux_out_Ds_Ni_left = CylindricalFlux(field=H, surface=Ds_Ni_left)  # A
    flux_out_Up_Ni_left = CylindricalFlux(field=H, surface=Up_Ni_left)  # B
    flux_out_Liquid_top = CylindricalFlux(field=H, surface=Liquid_top)  # A
    flux_out_mem_Ni_bottom = CylindricalFlux(field=H, surface=mem_Ni_bottom)  # B
    flux_out_bottom_Ni_top = CylindricalFlux(field=H, surface=bottom_Ni_top)  # B

    # glovebox outlet (external surface)
    glovebox_flux = CylindricalFlux(field=H, surface=out_surf)

    # register fluxes into model exports
    my_model.exports = [
        flux_out_top_Ni_bottom,
        flux_out_Ds_Ni_left,
        flux_out_Up_Ni_left,
        flux_out_Liquid_top,
        flux_out_mem_Ni_bottom,
        flux_out_bottom_Ni_top,
        glovebox_flux,
    ]

    # explicit label-to-object mapping for easy reference
    flux_by_label = {
        "top_Ni_bottom": flux_out_top_Ni_bottom,
        "Ds_Ni_left": flux_out_Ds_Ni_left,
        "Up_Ni_left": flux_out_Up_Ni_left,
        "Liquid_top": flux_out_Liquid_top,
        "mem_Ni_bottom": flux_out_mem_Ni_bottom,
        "bottom_Ni_top": flux_out_bottom_Ni_top,
    }

    # list of labels for plotting and output order
    six_labels = [
        "top_Ni_bottom",
        "Ds_Ni_left",
        "Up_Ni_left",
        "Liquid_top",
        "mem_Ni_bottom",
        "bottom_Ni_top",
    ]

    # upstream/downstream definitions depend on case type
    up_labels = (
        ["mem_Ni_bottom", "bottom_Ni_top", "Up_Ni_left"]
        if case in ("normal_infinite", "normal_transparent")
        else ["top_Ni_bottom", "Ds_Ni_left", "Liquid_top"]
    )
    down_labels = (
        ["top_Ni_bottom", "Ds_Ni_left", "Liquid_top"]
        if case in ("normal_infinite", "normal_transparent")
        else ["mem_Ni_bottom", "bottom_Ni_top", "Up_Ni_left"]
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
    my_model, flux = make_model(
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

    # flux_value = _get_flux_value(flux["flux_by_label"]["top_Ni_bottom"])
    # print("Flux at top_Ni_bottom =", flux_value)

    # flux_value = _get_flux_value(flux["flux_by_label"]["top_Ni_bottom"])
    # print("top_Ni_bottom flux =", flux_value)

    # per-face values (only the six faces)
    flux_objects = flux["flux_by_label"]  # label -> CylindricalFlux
    six_labels = flux["six_labels"]  # fixed order of the six faces
    vals_six = {lbl: _get_flux_value(flux_objects[lbl]) for lbl in six_labels}
    # print("  Fluxes at six faces:")
    # print("   ", ", ".join([f"{lbl}: {vals_six[lbl]:.3e}" for lbl in six_labels]))

    # glovebox (out surface)
    glovebox_val = float(_get_flux_value(flux["glovebox_flux"]))

    # totals: simple sum of their own values (no duplication)
    up_labels = flux["up_labels"]
    down_labels = flux["down_labels"]

    total_up = float(np.sum([vals_six[lbl] for lbl in up_labels], dtype=float))
    total_down = float(np.sum([vals_six[lbl] for lbl in down_labels], dtype=float))

    # per-surface payload (keep backward-compatible keys)
    per_surface = {
        "labels": six_labels,
        "values": [vals_six[lbl] for lbl in six_labels],
        "up_labels": up_labels,
        "down_labels": down_labels,
        "up_names": up_labels,
        "down_names": down_labels,
        "up_vals": [vals_six[lbl] for lbl in up_labels],
        "down_vals": [vals_six[lbl] for lbl in down_labels],
    }

    return dict(
        total_in=total_up,  # sum over the upstream group
        glovebox=glovebox_val,  # out_surf only
        total_out=total_down,  # sum over the downstream group
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


def _phi_base_at_T(permeability_flibe, T_K: float) -> float:
    """Evaluate base FLiBe permeability at temperature T_K."""
    try:
        return float(permeability_flibe.value(T_K).magnitude)
    except Exception:
        return float(permeability_flibe.pre_exp) * math.exp(
            -float(permeability_flibe.act_energy) / (kB_eV * float(T_K))
        )


def _collect_normal_run1_and_swap_points(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
) -> List[CalibPoint]:
    """
    Build dataset using:
      - normal_* cases, Run 1 only
      - swap_*   cases, Run 1 only
    Each (case, temperature) contributes one point.
    """
    pts: List[CalibPoint] = []
    for case_name, cfg in cases.items():
        is_normal = case_name.startswith("normal")
        is_swap = case_name.startswith("swap")
        if not (is_normal or is_swap):
            continue  # ignore other families if any

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
        raise RuntimeError("No points found for normal_* Run 1 and swap_* Run 1.")
    # Sort by (case, T) for reproducible logs/plots
    pts.sort(key=lambda p: (p.case, p.T_C))
    return pts


def calibrate_phi_schemeA(
    cases: Dict,
    T2K: Dict[float, float],
    Y_FT_BY_TEMP_C: Dict[float, float],
    D_flibe,
    D_nickel,
    K_S_nickel,
    permeability_flibe_base,
    # plotting/output control
    outdir: Path = Path("exports") / "figs" / "calibration_A",
    also_show: bool = False,
):
    """
    Scheme A:
      1) Run the model once with a base permeability (permeability_flibe_base).
      2) For each point, compute s = J_exp / J_model.
      3) Define phi_new(T) = s * phi_base(T).
      4) Fit ln(phi_new) = a + b * (1/T) via linear regression (Arrhenius).
         Then Phi0 = exp(a), E = -b * kB.
      5) Re-run with fitted (Phi0, E) for reporting and a parity plot.

    Returns:
      dict(phi0=..., E=..., r2=..., table=[...])
    """
    _ensure_fig_dir(outdir)

    # 0) Collect dataset (normal Run 1 + swap Run 1)
    pts = _collect_normal_run1_and_swap_points(cases, T2K, Y_FT_BY_TEMP_C)

    # 1–2) Single-pass scaling per (case, T)
    # For each point: run model with base permeability, compute s and phi_new(T)
    ln_phi_new_list = []
    invT_list = []
    rows = []  # for logging/report

    for p in pts:
        # Build boundary condition for this point
        out_bc = (
            {"type": "sieverts", "pressure": p.P_gb}
            if p.P_gb is not None
            else {"type": "particle_flux_zero"}
        )

        # Evaluate model flux using the base permeability
        out = run_once(
            case=p.case,
            T_K=p.T_K,
            P_up=p.P_up,
            P_down=p.P_down,
            D_flibe=D_flibe,
            D_nickel=D_nickel,
            permeability_flibe=permeability_flibe_base,
            K_S_nickel=K_S_nickel,
            out_bc=out_bc,
            y_ft=p.y_ft,
        )
        eps = np.finfo(float).tiny
        J_model = max(float(out["total_out"]), eps)
        J_exp = max(float(p.J_exp), eps)

        s = J_exp / J_model
        phi_base_T = _phi_base_at_T(permeability_flibe_base, p.T_K)
        phi_new_T = max(s * phi_base_T, eps)

        ln_phi_new_list.append(math.log(phi_new_T))
        invT_list.append(1.0 / p.T_K)

        rows.append(
            {
                "case": p.case,
                "T_C": p.T_C,
                "run": p.run,
                "J_model_base": J_model,
                "J_exp": J_exp,
                "scale_s": s,
                "phi_base_T": phi_base_T,
                "phi_new_T": phi_new_T,
            }
        )

    ln_phi_new = np.array(ln_phi_new_list, dtype=float)
    invT = np.array(invT_list, dtype=float)

    # 3–4) Arrhenius linear regression: ln(phi) = a + b*(1/T)
    # => Phi0 = exp(a), E = -b * kB
    # Use ordinary least squares via numpy.polyfit
    b, a = np.polyfit(
        invT, ln_phi_new, deg=1
    )  # returns [slope, intercept] for y ≈ b*x + a
    Phi0_hat = float(math.exp(a))
    E_hat = float(-b * kB_eV)

    # Optional goodness-of-fit (R^2) in ln-space
    y_pred = b * invT + a
    ss_res = float(np.sum((ln_phi_new - y_pred) ** 2))
    ss_tot = float(np.sum((ln_phi_new - float(np.mean(ln_phi_new))) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print("\n[Scheme A] Fitted Arrhenius parameters from scaled points:")
    print(f"  Phi0 = {Phi0_hat:.3e}\n")
    print(f"  E    = {E_hat:.4f} eV\n")
    print(f"  R^2  = {R2:.4f} (fit in ln(phi) vs 1/T)\n")

    # 5) Re-run with fitted permeability for reporting and parity plot
    perm_fitted = htm.Permeability(pre_exp=Phi0_hat, act_energy=E_hat, law="henry")

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
        print(
            f"{p.case:<18} | {int(p.T_C):>5} | {p.run:<4} | {jm:>20.3e} | {je:>18.3e} | {rlog:>9.3f}"
        )

    # ---------- Plots (saved to disk, optional show) ----------
    # 1) ln(phi_new) vs 1/T with fitted line
    _ensure_fig_dir(outdir)
    fig1, ax1 = plt.subplots(figsize=(6.2, 4.6))
    ax1.scatter(invT, ln_phi_new, label="Scaled points (ln φ_new)", zorder=3)
    x_sorted_idx = np.argsort(invT)
    x_line = invT[x_sorted_idx]
    y_line = y_pred[x_sorted_idx]
    ax1.plot(x_line, y_line, label="Linear fit", linestyle="--")
    ax1.set_xlabel("1 / T  [1/K]")
    ax1.set_ylabel("ln φ  [ln(H·m⁻¹·s⁻¹·Pa⁻¹)]")
    ax1.set_title("Scheme A: ln(φ_new) vs 1/T (normal Run1 + swap Run1)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(outdir / "schemeA_lnphi_vs_invT.png", dpi=300, bbox_inches="tight")

    # 2) Parity plot: J_model(fitted) vs J_exp
    jm_arr = np.array(jm_fit_list, dtype=float)
    je_arr = np.array(je_list, dtype=float)
    lo = float(min(jm_arr.min(), je_arr.min()))
    hi = float(max(jm_arr.max(), je_arr.max()))
    fig2, ax2 = plt.subplots(figsize=(5.8, 5.8))
    ax2.scatter(je_arr, jm_arr)
    ax2.plot([lo, hi], [lo, hi], linestyle="--")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Experimental flux  J_exp [H/s]")
    ax2.set_ylabel("Model flux (fitted) J_model [H/s]")
    ax2.set_title("Scheme A: Parity (normal Run1 + swap Run1)")
    # annotate points
    for x, y, lab in zip(je_arr, jm_arr, labels):
        ax2.text(x, y, lab, fontsize=8)
    fig2.tight_layout()
    fig2.savefig(outdir / "schemeA_parity.png", dpi=300, bbox_inches="tight")

    # Optional show (will do nothing on headless Agg)
    if also_show and "agg" not in matplotlib.get_backend().lower():
        plt.show()

    return {
        "phi0": Phi0_hat,
        "E": E_hat,
        "R2": R2,
        "points": rows,  # list of dicts for per-point diagnostics
        "fig_dir": str(outdir),
    }

    # ================== End Scheme A block ==================


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
    # temps_C_all = [500.0]
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

    cases = {
        # "normal_infinite": {"table": normal_infinite, "out_mode": "particle_flux_zero"},
        "normal_transparent": {"table": normal_transparent, "out_mode": "sieverts"},
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

    # ==================== RUN ALL CASES & COLLECT ====================
    all_results = {}  # case -> Tc -> run -> {model, exp, conds}
    case_summaries = {}  # arrays for overview plots

    for case_name, cfg in cases.items():
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
                    permeability_flibe=permeability_flibe,
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

        # D_Ni(T)
        try:
            D_val = float(D_nickel.value(T_K).magnitude)
        except Exception:
            D_val = float(D_nickel.pre_exp) * np.exp(
                -float(D_nickel.act_energy) / (kB_eV * T_K)
            )

        # K_S_Ni(T)
        try:
            Ks_val = float(K_S_nickel.value(T_K).magnitude)
        except Exception:
            Ks_val = float(K_S_nickel.pre_exp) * np.exp(
                -float(K_S_nickel.act_energy) / (kB_eV * T_K)
            )

        return D_val * Ks_val

    # print("\n=== Overview plots ===")
    # print(f"{pi_ni_at_T(773.15):.2e}")
    # print(f"{pi_ni_at_T(823.15):.2e}")
    # print(f"{pi_ni_at_T(873.15):.2e}")
    # print(f"{pi_ni_at_T(923.15):.2e}")
    # print(f"{pi_ni_at_T(973.15):.2e}")

    # ==================== OVERVIEW: 4 CASES FIRST ====================
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

        # draw bars + annotate each bar with its numeric value
        for s, r in zip(shifts, run_names):
            # experimental bars (hollow)
            bars_exp = axA.bar(
                x + s - w / 2,
                exp_by_run[r],
                width=w,
                label=f"Exp {r}",
                edgecolor="black",
                fill=False,
            )
            # model bars (filled)
            bars_mod = axA.bar(
                x + s + w / 2,
                totals_by_run[r],
                width=w,
                label=f"Model {r}",
            )

            # annotate values on each bar
            def _annotate(container, values):
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
                        rotation=0,
                    )

            _annotate(bars_exp, exp_by_run[r])
            _annotate(bars_mod, totals_by_run[r])

        # x tick labels with FLiBe permeability Φ and Ni permeability Π_Ni
        xticklabels = []
        for T in xticks:
            Tk = T2K[T]
            phi_flibe = phi_at_T(Tk)
            pi_ni = pi_ni_at_T(Tk)
            yft = Y_FT_BY_TEMP_C.get(T, np.nan)
            xticklabels.append(
                f"{int(T)}°C\nΦ_FLiBe={phi_flibe:.2e}\nΦ_Ni={pi_ni:.2e}\n"
                f"y_ft={yft:.5f} m"
            )

        axA.set_xticks(x)
        axA.set_xticklabels(xticklabels)
        axA.set_ylabel("Total downstream flux [H/s]")
        axA.set_title(f"{case_name}: model vs experiment")
        axA.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axA.legend(ncol=max(2, len(run_names)))
        figA.tight_layout()
        save_overview(figA, case_name, name="overview")
        plt.show()  # popup

    # Use your current base permeability as the starting curve
    result_A = calibrate_phi_schemeA(
        cases=cases,
        T2K=T2K,
        Y_FT_BY_TEMP_C=Y_FT_BY_TEMP_C,
        D_flibe=D_flibe,
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        permeability_flibe_base=permeability_flibe,
        outdir=Path("exports") / "figs" / "calibration_A",
        also_show=False,  # set True if you have a GUI backend
    )
    print("Scheme A fitted parameters:", result_A)

    # print("\n=== Permeability sensitivity test ===")
    # # Sensitivity test: vary permeability pre-exponential by factors of 0.5, 1.0, 2.0
    # # Expect linear scaling of output flux J
    # factors = [0.5, 1.0, 2.0]
    # Js = []
    # for k in factors:
    #     perm = htm.Permeability(
    #         pre_exp=permeability_flibe.pre_exp * k,
    #         act_energy=permeability_flibe.act_energy,
    #         law="henry",
    #     )
    #     out = run_once(
    #         case="normal_transparent",  # pick any one case/temp to test
    #         T_K=823.15,  # e.g. 550°C
    #         P_up=1.10e5,
    #         P_down=7.10,
    #         D_flibe=D_flibe,
    #         D_nickel=D_nickel,
    #         permeability_flibe=perm,
    #         K_S_nickel=K_S_nickel,
    #         out_bc={"type": "sieverts", "pressure": 5.0},
    #         y_ft=0.02919,
    #     )
    #     Js.append(out["total_out"])

    # print("J ratios:", [Js[i] / Js[1] for i in range(len(factors))])
    # Expect roughly [0.5, 1.0, 2.0]

    # If you want to immediately use the fitted curve in subsequent runs:
    # permeability_flibe = htm.Permeability(pre_exp=result_A["phi0"], act_energy=result_A["E"], law="henry")

    # # ==================== THEN: PER-TEMPERATURE BREAKDOWNS ====================
    # for case_name, per_case in all_results.items():
    #     for Tc in sorted(per_case.keys()):
    #         for r, pack in per_case[Tc].items():
    #             res = pack["model"]
    #             per = res["per_surface"]
    #             up_names = per["up_names"]
    #             down_names = per["down_names"]
    #             up_vals = per["up_vals"]
    #             down_vals = per["down_vals"]

    #             labels = (
    #                 ["Flux in (total)", "Flux to glovebox", "Flux out (total)"]
    #                 + [f"in:{n}" for n in up_names]
    #                 + [f"out:{n}" for n in down_names]
    #             )
    #             values = (
    #                 [res["total_in"], res["glovebox"], res["total_out"]]
    #                 + up_vals
    #                 + down_vals
    #             )

    #             figB, axB = plt.subplots(figsize=(12, 5.5))
    #             xB = np.arange(len(labels), dtype=float)
    #             bars = axB.bar(xB, np.abs(values))
    #             for bar, v in zip(bars, values):
    #                 axB.text(
    #                     bar.get_x() + bar.get_width() / 2.0,
    #                     bar.get_height() * 1.02 if bar.get_height() > 0 else 1.0,
    #                     f"{v:.2e}",
    #                     ha="center",
    #                     va="bottom",
    #                     fontsize=9,
    #                 )

    #             axB.set_xticks(xB)
    #             axB.set_xticklabels(labels, rotation=30, ha="right")
    #             axB.set_ylabel("Flux [H/s]")

    #             conds = pack["conds"]
    #             title = (
    #                 f"{case_name} — {r} — {int(Tc)} °C  "
    #                 f"(Φ={conds['phi']:.2e}, y_ft={conds['y_ft']:.5f} m)"
    #             )
    #             axB.set_title(title)

    #             # explicit experimental conditions on the figure
    #             txt = (
    #                 f"Case: {conds['case']}\nRun: {conds['run']}\n"
    #                 f"T = {int(conds['T_C'])} °C\n"
    #                 f"P_up = {conds['P_up']:.2e} Pa\n"
    #                 f"P_down = {conds['P_down']:.2e} Pa\n"
    #                 f"P_glovebox = {conds['P_gb']:.2e} Pa"
    #                 if conds["P_gb"] is not None
    #                 else f"Case: {conds['case']}\nRun: {conds['run']}\n"
    #                 f"T = {int(conds['T_C'])} °C\n"
    #                 f"P_up = {conds['P_up']:.2e} Pa\n"
    #                 f"P_down = {conds['P_down']:.2e} Pa\n"
    #                 f"P_glovebox = (closed)"
    #             )
    #             axB.text(
    #                 0.99,
    #                 0.98,
    #                 txt,
    #                 transform=axB.transAxes,
    #                 ha="right",
    #                 va="top",
    #                 bbox=dict(boxstyle="round", fc="white", alpha=0.8, lw=0.5),
    #             )
    #             axB.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    #             figB.tight_layout()

    #             fd = fig_dir_for(case_name, int(Tc), r)
    #             save_breakdown(figB, fd, stem=f"{case_name}_{int(Tc)}C_{r}_breakdown")
    #             plt.show()  # popup for each run/temperature
