import festim as F
import numpy as np
import h_transport_materials as htm
from pathlib import Path
import csv
from typing import Dict, Optional
import matplotlib.pyplot as plt
import math
import pandas as pd


# ------------------------------ Temperature -> FLiBe thickness ------------------------------
L_FLIBE_BY_TEMP_C: Dict[float, float] = {
    # 500.0: 0.005139858,
    # 550.0: 0.005194021,
    # 600.0: 0.005249337,
    650.0: 0.005305845,
    700.0: 0.005363582,
}

J_exp_BY_TEMP_C: Dict[float, float] = {
    500.0: 4.34e15,
    550.0: 8.58e15,
    600.0: 1.01e16,
    650.0: 1.10e16,
    700.0: 1.04e16,
}


def read_experimental_points(excel_path: Path, T_C: float):
    sheet = f"{int(T_C)}C"
    df = pd.read_excel(excel_path, sheet_name=sheet)

    t_exp = df["time_s"].to_numpy(dtype=float)
    J_exp = df["flux_Hps"].to_numpy(dtype=float)

    return t_exp, J_exp


def flibe_thickness_from_T_C(T_C: float) -> float:
    temps = np.array(sorted(L_FLIBE_BY_TEMP_C.keys()), dtype=float)
    Ls = np.array([L_FLIBE_BY_TEMP_C[T] for T in temps], dtype=float)
    return float(np.interp(T_C, temps, Ls))


# ------------------------------ Materials ------------------------------
def make_materials_ni_flibe(
    D_nickel: htm.Diffusivity,
    K_S_nickel: htm.Solubility,
    D_flibe: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
):
    """
    Ni: Sievert (solid)
    FLiBe: Henry (liquid), solubility reconstructed from permeability & diffusivity (P = D*K)
    """
    mat_ni = F.Material(
        D_0=D_nickel.pre_exp.magnitude,
        E_D=D_nickel.act_energy.magnitude,
        K_S_0=K_S_nickel.pre_exp.magnitude,
        E_K_S=K_S_nickel.act_energy.magnitude,
        solubility_law="sievert",
    )

    K_S_liquid = htm.Solubility(
        S_0=permeability_flibe.pre_exp / D_flibe.pre_exp,
        E_S=permeability_flibe.act_energy - D_flibe.act_energy,
        law=permeability_flibe.law,  # "henry"
    )
    mat_flibe = F.Material(
        D_0=D_flibe.pre_exp.magnitude,
        E_D=D_flibe.act_energy.magnitude,
        K_S_0=K_S_liquid.pre_exp.magnitude,
        E_K_S=K_S_liquid.act_energy.magnitude,
        solubility_law="henry",
    )
    return mat_ni, mat_flibe, K_S_liquid


def beta(t):
    """
    Bubble coverage factor.
    """
    t_val = float(t)
    beta_inf = 0.6
    tau = 30.0
    # return beta_inf + (1.0 - beta_inf) * math.exp(-t_val / tau)
    return 1.0


# ------------------------------ Build model ------------------------------
def build_collapsed_1d(
    T_C: float,
    P_up: float,
    L_Ni: float,
    radius: float,
    D_nickel: htm.Diffusivity,
    K_S_nickel: htm.Solubility,
    D_flibe: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    # downstream (left) boundary option:
    left_mode: str = "fixed_c0",  # "fixed_c0" or "sievert_Pdown"
    P_down: Optional[float] = None,  # only used if left_mode="sievert_Pdown"
    penalty_term: float = 1e20,
    n_ni: int = 80,
    n_flibe: int = 80,
    final_time: float = 2.5e4,
):
    """
    Geometry:
      [0, L_Ni]                 = Ni (LEFT)
      [L_Ni, L_Ni + L_FLiBe(T)] = FLiBe (RIGHT)

    BCs (right -> left permeation):
      - RIGHT boundary (x=end, FLiBe): HenrysBC at P_up  (upstream)
      - LEFT boundary  (x=0, Ni): either
            fixed_c0         -> FixedConcentrationBC(value=0.0)  (sink)
            sievert_Pdown    -> SievertsBC at P_down             (finite downstream pressure)

    Flux monitors:
      SurfaceFlux returns flux density [H/m^2/s].
      Convert to total [H/s] via A = pi r^2.
    """
    T_K = T_C + 273.15
    L_flibe = flibe_thickness_from_T_C(T_C)

    mat_ni, mat_flibe, K_S_liquid = make_materials_ni_flibe(
        D_nickel=D_nickel,
        K_S_nickel=K_S_nickel,
        D_flibe=D_flibe,
        permeability_flibe=permeability_flibe,
    )

    x0 = 0.0
    x1 = L_Ni
    x2 = L_Ni + L_flibe

    # volumes
    vol_ni = F.VolumeSubdomain1D(id=1, borders=[x0, x1], material=mat_ni)
    vol_flibe = F.VolumeSubdomain1D(id=2, borders=[x1, x2], material=mat_flibe)

    # boundaries
    left_bc = F.SurfaceSubdomain1D(id=1, x=x0)  # Ni side
    right_bc = F.SurfaceSubdomain1D(id=2, x=x2)  # FLiBe side (upstream)

    model = F.HydrogenTransportProblemDiscontinuous()
    model.subdomains = [vol_ni, vol_flibe, left_bc, right_bc]
    model.surface_to_volume = {left_bc: vol_ni, right_bc: vol_flibe}
    model.interfaces = [
        F.Interface(id=3, subdomains=[vol_ni, vol_flibe], penalty_term=penalty_term)
    ]

    # mesh
    v_ni = np.linspace(x0, x1, n_ni)
    v_flibe = np.linspace(x1, x2, n_flibe)[1:]
    model.mesh = F.Mesh1D(np.concatenate([v_ni, v_flibe]))

    # species
    H = F.Species("H", subdomains=model.volume_subdomains)
    model.species = [H]

    # temperature
    model.temperature = T_K

    # BCs: RIGHT is upstream on FLiBe
    # bc_right_upstream = F.HenrysBC(
    #     subdomain=right_bc,
    #     species=H,
    #     pressure=P_up,
    #     H_0=K_S_liquid.pre_exp.magnitude,
    #     E_H=K_S_liquid.act_energy.magnitude,
    # )
    kB_eV = 8.617333262e-5

    K_H_T = K_S_liquid.pre_exp.magnitude * math.exp(
        -K_S_liquid.act_energy.magnitude / (kB_eV * T_K)
    )

    bc_right_upstream = F.FixedConcentrationBC(
        subdomain=right_bc,
        species=H,
        value=lambda t: beta(t) * K_H_T * P_up,
    )

    # LEFT is downstream on Ni
    if left_mode == "fixed_c0":
        bc_left_downstream = F.FixedConcentrationBC(
            subdomain=left_bc, species=H, value=0.0
        )
    elif left_mode == "sievert_Pdown":
        if P_down is None:
            raise ValueError("left_mode='sievert_Pdown' requires P_down.")
        bc_left_downstream = F.SievertsBC(
            subdomain=left_bc,
            species=H,
            pressure=P_down,
            S_0=K_S_nickel.pre_exp.magnitude,
            E_S=K_S_nickel.act_energy.magnitude,
        )
    else:
        raise ValueError("left_mode must be 'fixed_c0' or 'sievert_Pdown'.")

    model.boundary_conditions = [bc_left_downstream, bc_right_upstream]

    # transient settings
    dt = F.Stepsize(
        initial_value=1, growth_factor=1.1, cutback_factor=0.9, target_nb_iterations=4
    )
    model.settings = F.Settings(
        atol=1e12, rtol=1e-13, transient=True, stepsize=dt, final_time=final_time
    )

    # exports: flux densities + profiles
    j_left = F.SurfaceFlux(field=H, surface=left_bc)  # downstream side
    j_right = F.SurfaceFlux(field=H, surface=right_bc)  # upstream side
    prof_ni = F.Profile1DExport(field=H, subdomain=vol_ni)
    prof_flibe = F.Profile1DExport(field=H, subdomain=vol_flibe)
    model.exports = [j_left, j_right, prof_ni, prof_flibe]
    area = np.pi * radius**2
    return model, (j_left, j_right), (prof_ni, prof_flibe), area, (T_K, L_flibe)


# ------------------------------ Run one temperature  ------------------------------
def run_one_T(
    T_C: float,
    P_up: float,
    L_Ni: float,
    radius: float,
    D_nickel: htm.Diffusivity,
    K_S_nickel: htm.Solubility,
    D_flibe: htm.Diffusivity,
    permeability_flibe: htm.Permeability,
    left_mode: str = "fixed_c0",
    P_down: Optional[float] = None,
    out_dir: Path = Path("exports/bubble_1d"),
):
    model, (j_left, j_right), (prof_ni, prof_flibe), area, (T_K, L_flibe) = (
        build_collapsed_1d(
            T_C=T_C,
            P_up=P_up,
            L_Ni=L_Ni,
            radius=radius,
            D_nickel=D_nickel,
            K_S_nickel=K_S_nickel,
            D_flibe=D_flibe,
            permeability_flibe=permeability_flibe,
            left_mode=left_mode,
            P_down=P_down,
        )
    )

    model.initialise()
    model.run()

    t = np.asarray(j_left.t, dtype=float)

    # total fluxes [H/s]
    J_downstream_left = np.asarray(j_left.data, dtype=float) * area
    J_upstream_right = np.asarray(j_right.data, dtype=float) * area

    J_exp = J_exp_BY_TEMP_C[T_C]
    s_T = J_exp / J_downstream_left[-1]

    J_downstream_left *= s_T

    J_model = np.asarray(j_left.data, dtype=float) * area

    # ------------------ beta extraction from experimental points ------------------
    excel_path = Path("experimental_data.xlsx")

    if T_C in [650.0, 700.0]:
        t_exp, J_exp = read_experimental_points(excel_path, T_C)

        # interpolate model onto experimental times
        J_model_at_exp = np.interp(t_exp, t, J_model)

        # beta(t)
        beta_exp = J_exp / (s_T * J_model_at_exp)

        # save beta CSV
        beta_dir = out_dir / "beta"
        beta_dir.mkdir(parents=True, exist_ok=True)

        beta_csv = beta_dir / f"T{int(T_C)}C_beta.csv"
        with beta_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_s", "beta"])
            for ti, bi in zip(t_exp, beta_exp):
                w.writerow([ti, bi])

        # plot beta(t)
        plt.figure(figsize=(6, 4))
        plt.scatter(t_exp, beta_exp, s=18)
        plt.axhline(1.0, color="k", linestyle="--", linewidth=1)
        plt.xlabel("Time [s]")
        plt.ylabel("β(t)")
        plt.title(f"Extracted β(t) at {int(T_C)} °C")
        plt.grid(True)
        plt.tight_layout()

        beta_fig = beta_dir / f"T{int(T_C)}C_beta.png"
        plt.savefig(beta_fig, dpi=300)
        plt.close()

    # ---- plot downstream flux vs time ----
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(t, J_downstream_left)
    plt.xlabel("Time [s]")
    plt.ylabel("Downstream flux [H/s]")
    plt.title(f"Downstream flux vs time (T = {T_C:.0f} °C)")
    plt.grid(True)
    plt.tight_layout()

    fig_path = fig_dir / f"T{int(T_C)}C_downstream_flux.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # save time series
    out_dir.mkdir(parents=True, exist_ok=True)
    flux_csv = out_dir / f"T{int(T_C)}C_flux.csv"
    with flux_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["time [s]", "Flux_upstream_right [H/s]", "Flux_downstream_left [H/s]"]
        )
        for i in range(len(t)):
            w.writerow([t[i], J_upstream_right[i], J_downstream_left[i]])

    # optional profiles (last step)
    prof_dir = out_dir / "profiles"
    prof_dir.mkdir(parents=True, exist_ok=True)
    if prof_ni.data:
        np.savetxt(
            prof_dir / f"T{int(T_C)}C_ni.csv", np.array(prof_ni.data[-1]), delimiter=","
        )
    if prof_flibe.data:
        np.savetxt(
            prof_dir / f"T{int(T_C)}C_flibe.csv",
            np.array(prof_flibe.data[-1]),
            delimiter=",",
        )

    return {
        "T_C": T_C,
        "T_K": T_K,
        "L_flibe": L_flibe,
        "P_up_right": P_up,
        "left_mode": left_mode,
        "P_down_left": P_down if P_down is not None else "",
        "J_out_left_final": float(J_downstream_left[-1]),
        "flux_csv": str(flux_csv),
    }


# ------------------------------ Main  ------------------------------
if __name__ == "__main__":
    # Nickel data
    diffusivities_nickel = htm.diffusivities.filter(material="nickel").filter(
        isotope="h"
    )
    solubilities_nickel = htm.solubilities.filter(material="nickel").filter(isotope="h")
    D_nickel = diffusivities_nickel[1]
    K_S_nickel = solubilities_nickel[1]

    # FLiBe diffusivity
    D_flibe = htm.Diffusivity(D_0=2.5e-7, E_D=0.24)

    # geometry constants
    L_Ni = 0.002032  # [m]
    radius = 3.07 * 0.0254 / 2.0  # [m]

    # choose a permeability
    permeability_flibe = htm.Permeability(
        pre_exp=41587400565660.95, act_energy=0.4655730255084721, law="henry"
    )

    # pressures (upstream is on RIGHT)
    P_up = 1.32e5

    results = []
    for T_C in sorted(L_FLIBE_BY_TEMP_C.keys()):
        results.append(
            run_one_T(
                T_C=T_C,
                P_up=P_up,
                L_Ni=L_Ni,
                radius=radius,
                D_nickel=D_nickel,
                K_S_nickel=K_S_nickel,
                D_flibe=D_flibe,
                permeability_flibe=permeability_flibe,
                left_mode="fixed_c0",
                # left_mode="sievert_Pdown",
                # P_down=5.02e1,
            )
        )

    # master summary
    out_path = Path("exports/bubble_1d/master_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "T_C",
        "T_K",
        "L_flibe",
        "P_up_right",
        "left_mode",
        "P_down_left",
        "J_out_left_final",
        "flux_csv",
    ]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})

    print(f"Saved: {out_path}")
