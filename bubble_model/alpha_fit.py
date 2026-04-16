#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# ===================== DATA =====================
# T_K = np.array([773.15, 823.15, 873.15, 923.15, 973.15], dtype=float)
# alpha_ideal = np.array([0.1868, 0.177959, 0.159689, 0.221991, 0.222223], dtype=float)
# alpha_no = np.array([0.310431, 0.375522, 0.443258, 0.678974, 0.782391], dtype=float)

T_K = np.array([873.15, 923.15, 973.15], dtype=float)
alpha_ideal = np.array([0.9190368908973282, 0.688, 0.451], dtype=float)
alpha_no = np.array([0.8591861633155928, 0.660, 0.446], dtype=float)

# T_K = np.array([923.15, 973.15], dtype=float)
# alpha_ideal = np.array([0.688, 0.451], dtype=float)
# alpha_no = np.array([0.660, 0.446], dtype=float)

datasets = {
    "Ideal coating": alpha_ideal,
    "No coating": alpha_no,
}

outdir = Path("exports") / "alpha_fit_models_overlay_v2"
outdir.mkdir(parents=True, exist_ok=True)

T_fit = np.linspace(T_K.min(), T_K.max(), 500)


# ===================== MODELS =====================
def linear(T, a, b):
    return a + b * T


def power_law(T, A, n):
    return A * T**n


def exp_T(T, a, b, c):
    return a + b * np.exp(c * T)


def arrhenius_offset(T, a, b, E):
    R = 8.314462618  # J/mol/K
    return a + b * np.exp(-E / (R * T))


def init_power_law(T, y):
    # log(y)=log(A)+n log(T)
    n0, logA0 = np.polyfit(np.log(T), np.log(y), 1)
    return float(np.exp(logA0)), float(n0)


EV_PER_JMOL = 1.0 / 96485.33212
models = {
    "linear": {
        "func": linear,
        "p0": lambda T, y: (float(np.mean(y)), 0.0),
        "bounds": (-np.inf, np.inf),
        "title": r"Linear model:  $\alpha = a + bT$",
        "param_text": lambda p: (
            rf"$a={p[0]:.6g}$" + "\n" + rf"$b={p[1]:.3e}\ \mathrm{{K^{{-1}}}}$"
        ),
        "fit_label": "Linear fit",
        "fname": "model_linear.png",
    },
    "power_law": {
        "func": power_law,
        "p0": lambda T, y: init_power_law(T, y),
        "bounds": (-np.inf, np.inf),
        "title": r"Power-law model:  $\alpha = A\,T^n$",
        "param_text": lambda p: (rf"$A={p[0]:.3e}$" + "\n" + rf"$n={p[1]:.3f}$"),
        "fit_label": "Power-law fit",
        "fname": "model_power_law.png",
    },
    "exp_T": {
        "func": exp_T,
        "p0": lambda T, y: (
            float(np.min(y)),
            float(max(np.max(y) - np.min(y), 1e-6)),
            1e-3,
        ),
        "bounds": ([-np.inf, -np.inf, -1.0], [np.inf, np.inf, 1.0]),
        "title": r"Exponential model:  $\alpha = a + b\,e^{cT}$",
        "param_text": lambda p: (
            rf"$a={p[0]:.6g}$"
            + "\n"
            + rf"$b={p[1]:.3e}$"
            + "\n"
            + rf"$c={p[2]:.3e}\ \mathrm{{K^{{-1}}}}$"
        ),
        "fit_label": "Exp(T) fit",
        "fname": "model_exp_T.png",
    },
    "arrhenius": {
        "func": arrhenius_offset,
        "p0": lambda T, y: (
            float(np.min(y)),
            float(max(np.max(y) - np.min(y), 1e-6)),
            2e4,
        ),
        "bounds": ([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 5e5]),
        "title": r"Arrhenius-offset model:  $\alpha = a + b\,e^{-E/(RT)}$",
        "param_text": lambda p: (
            rf"$a={p[0]:.6g}$"
            + "\n"
            + rf"$b={p[1]:.3e}$"
            + "\n"
            + rf"$E={p[2] * EV_PER_JMOL:.3f}\ \mathrm{{eV}}$"
        ),
        "fit_label": "Arrhenius-offset fit",
        "fname": "model_arrhenius.png",
    },
}

# ===================== FIT + PLOT (ONE FIGURE PER MODEL) =====================
for model_name, m in models.items():
    func = m["func"]

    # Fit both datasets using this model
    fit_params = {}
    for ds_name, y in datasets.items():
        p0 = m["p0"](T_K, y)
        popt, _ = curve_fit(func, T_K, y, p0=p0, bounds=m["bounds"], maxfev=50000)
        fit_params[ds_name] = popt

    # ---- Plot overlay ----
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    # reserve right margin for parameter boxes
    fig.subplots_adjust(right=0.72)

    for ds_name, y in datasets.items():
        ax.scatter(T_K, y, s=70, marker="o", label=f"{ds_name} data", zorder=3)
        ax.plot(
            T_fit,
            func(T_fit, *fit_params[ds_name]),
            linestyle="--",
            linewidth=2,
            label=f"{ds_name} {m['fit_label']}",
        )

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(m["title"])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")

    # Put parameter boxes OUTSIDE the axes (to the right)
    # Use figure coordinates so it always stays outside plot area.
    # Two vertically stacked boxes.
    text_ideal = "Ideal coating\n" + m["param_text"](fit_params["Ideal coating"])
    text_no = "No coating\n" + m["param_text"](fit_params["No coating"])

    # x=0.74 is just to the right of the axes because we used right=0.72
    fig.text(
        0.74,
        0.78,
        text_no,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95),
    )
    fig.text(
        0.74,
        0.42,
        text_ideal,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.95),
    )

    outpath = outdir / m["fname"]
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outpath}")

print("\nDone. One figure per model; parameter boxes are outside the plot region.")
