# plot_swap_1D_2D_Exp_panel_journal.py
# 2x2 journal-style figure:
# (a) swap_infinite – H (Run 2)   (b) swap_infinite – D (Run 3)
# (c) swap_transparent – H (Run 2) (d) swap_transparent – D (Run 3)
#
# Styling:
# - log y
# - black/gray-safe (no reliance on color)
# - 1D: solid line + square marker
# - 2D: dashed line + triangle marker
# - Exp: circle markers only (no line)
# Saved to: plots/swap_1D_2D_Exp_panel.pdf (+ .png)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent

# --------- Inputs (adjust if your filenames differ) ----------
CSV_1D = ROOT / "exports" / "results_1d" / "all_results_1d.csv"
CSV_2D = ROOT / "exports" / "2Dcomparison" / "sim_vs_exp.csv"

# --------- Output ----------
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUTDIR / "swap_1D_2D_Exp_panel.pdf"
OUT_PNG = OUTDIR / "swap_1D_2D_Exp_panel.png"

# --------- Scaling (set to 2.0 if that dataset is half-geometry) ----------
SCALE_1D = 1.0
SCALE_2D = 1.0  # if your sim_vs_exp.csv already has J_sim*2, keep 1.0

CASES = ["swap_infinite", "swap_transparent"]
RUNS = ["Run 2", "Run 3"]  # Run 2 = H, Run 3 = D

RUN_SHORT = {"Run 2": "H", "Run 3": "D"}
CASE_SHORT = {"swap_infinite": "Ideal coating", "swap_transparent": "No coating"}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def _pick_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


def prepare_1d(df: pd.DataFrame) -> pd.DataFrame:
    case_col = _pick_col(df, ["case"])
    run_col = _pick_col(df, ["run"])
    tc_col = _pick_col(df, ["T_C", "Tc", "temp_C", "Temperature_C"])
    jsim_col = _pick_col(df, ["J_sim", "Jsim", "J_out", "J_out_sim", "J_out"])

    out = df[[case_col, run_col, tc_col, jsim_col]].copy()
    out = out.rename(
        columns={case_col: "case", run_col: "run", tc_col: "T_C", jsim_col: "J_1D"}
    )

    out["case"] = out["case"].astype(str).str.strip()
    out["run"] = out["run"].astype(str).str.strip()
    out["T_C"] = pd.to_numeric(out["T_C"], errors="coerce")
    out["J_1D"] = pd.to_numeric(out["J_1D"], errors="coerce") * SCALE_1D
    return out


def prepare_2d(df: pd.DataFrame) -> pd.DataFrame:
    case_col = _pick_col(df, ["case"])
    run_col = _pick_col(df, ["run"])
    tc_col = _pick_col(df, ["T_C", "Tc", "temp_C", "Temperature_C"])
    jsim_col = _pick_col(df, ["J_sim", "Jsim", "J_out", "J_out_sim", "J_out"])
    jexp_col = _pick_col(df, ["J_exp", "Jexp", "J_out_exp"])

    out = df[[case_col, run_col, tc_col, jsim_col, jexp_col]].copy()
    out = out.rename(
        columns={
            case_col: "case",
            run_col: "run",
            tc_col: "T_C",
            jsim_col: "J_2D",
            jexp_col: "J_exp",
        }
    )

    out["case"] = out["case"].astype(str).str.strip()
    out["run"] = out["run"].astype(str).str.strip()
    out["T_C"] = pd.to_numeric(out["T_C"], errors="coerce")
    out["J_2D"] = pd.to_numeric(out["J_2D"], errors="coerce") * SCALE_2D
    out["J_exp"] = pd.to_numeric(out["J_exp"], errors="coerce")
    return out


def journal_axes_style(ax):
    ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.6)

    # nice log ticks (major at 10^n, minor at 2..9)
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # journal-like spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def plot_panel(ax, d, case_name, run_name, panel_label):
    sub = d[(d["case"] == case_name) & (d["run"] == run_name)].sort_values("T_C")
    if sub.empty:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(
            f"{panel_label}  {CASE_SHORT.get(case_name, case_name)} – {RUN_SHORT.get(run_name, run_name)}"
        )
        journal_axes_style(ax)
        return

    T = sub["T_C"].to_numpy(float)

    # ---- Black/white-safe styles ----
    # 1D: solid + square
    ax.plot(
        T,
        sub["J_1D"].to_numpy(float),
        linestyle="-",
        marker="s",
        linewidth=2.0,
        markersize=6.5,
        markerfacecolor="white",
        markeredgewidth=1.5,
        color="black",
        label="1D model",
    )

    # 2D: dashed + triangle
    ax.plot(
        T,
        sub["J_2D"].to_numpy(float),
        linestyle="--",
        marker="^",
        linewidth=2.0,
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
        color="black",
        label="2D model",
    )

    # Exp: markers only (no line)
    ax.plot(
        T,
        sub["J_exp"].to_numpy(float),
        linestyle="None",
        marker="o",
        markersize=7.0,
        markerfacecolor="white",
        markeredgewidth=1.5,
        color="black",
        label="Experiment",
    )

    ax.set_title(
        f"{panel_label}  {CASE_SHORT.get(case_name, case_name)} – {RUN_SHORT.get(run_name, run_name)}"
    )
    journal_axes_style(ax)


def main():
    # font sizes tuned for journals
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
        }
    )

    df1 = prepare_1d(_read_csv(CSV_1D))
    df2 = prepare_2d(_read_csv(CSV_2D))

    df1 = df1[df1["case"].isin(CASES) & df1["run"].isin(RUNS)]
    df2 = df2[df2["case"].isin(CASES) & df2["run"].isin(RUNS)]

    merged = pd.merge(df1, df2, on=["case", "run", "T_C"], how="inner")
    if merged.empty:
        raise RuntimeError(
            "Merged dataset is empty. Check that 1D and 2D share case/run/T_C entries."
        )

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)

    panels = [
        ("swap_infinite", "Run 2", "(a)"),
        ("swap_infinite", "Run 3", "(b)"),
        ("swap_transparent", "Run 2", "(c)"),
        ("swap_transparent", "Run 3", "(d)"),
    ]

    for ax, (case_name, run_name, lab) in zip(axes.flat, panels):
        plot_panel(ax, merged, case_name, run_name, lab)

    # shared labels
    fig.supxlabel("Temperature [°C]")
    fig.supylabel(r"Flux out [$\mathrm{H\,s^{-1}}$]")

    # one legend (from first axis), journal placement
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", OUT_PDF)
    print("Saved:", OUT_PNG)


if __name__ == "__main__":
    main()
