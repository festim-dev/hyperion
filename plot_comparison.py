import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

import morethemes as mt

mt.set_theme("lumen")

ROOT = Path(__file__).resolve().parent

# ── Inputs ────────────────────────────────────────────────────────────────────
# All CSV outputs from run_sim_vs_exp.py, run_sim_vs_exp_1d.py live in results/

RESULTS_DIR = ROOT / "results"

CSV_1D = RESULTS_DIR / "all_results_1d.csv"
CSV_2D = RESULTS_DIR / "jsim_jexp.csv"
XLSX_EXP_600 = RESULTS_DIR / "exp_600.xlsx"

# ── Outputs ───────────────────────────────────────────────────────────────────
OUTDIR = RESULTS_DIR
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_H_PDF = OUTDIR / "1D_2D_comparison_H.pdf"
OUT_D_PDF = OUTDIR / "1D_2D_comparison_D.pdf"
OUT_BC_PDF = OUTDIR / "BC_comparison.pdf"
OUT_EXP_600_PDF = OUTDIR / "H_experimental.pdf"

# ── Plot configuration ────────────────────────────────────────────────────────

SCALE_1D = 1.0
SCALE_2D = 1.0

CASES = ["swap_infinite", "swap_transparent"]
RUNS = ["Run 2", "Run 3"]

RUN_COLOR = {
    "Run 2": "red",
    "Run 3": "black",
}

RUN_LABEL = {
    "Run 2": "H",
    "Run 3": "D",
}

RUN_UNIT = {
    "Run 2": r"H s$^{-1}$",
    "Run 3": r"D s$^{-1}$",
}

CASE_LABEL = {
    "swap_infinite": "Ideal coating",
    "swap_transparent": "Uncoated",
}

CASE_MARKER = {
    "swap_infinite": "s",
    "swap_transparent": "^",
}

BASE_RC = {
    "font.size": 15,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 12,
    "axes.linewidth": 1.0,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
}

FIGSIZE_TWO_PANEL = (7.2, 8.2)
FIGSIZE_EXP = (7.2, 4.1)

TEMP_TICKS_K = [773, 823, 873, 923, 973]
TEMP_LIMS_K = (768, 978)

MODEL_1D_STYLE = dict(
    linestyle="--", linewidth=2.4, marker="+", markersize=9, markeredgewidth=1.4
)
MODEL_2D_STYLE = dict(
    linestyle="-", linewidth=2.4, marker="x", markersize=9, markeredgewidth=1.4
)
EXP_STYLE = dict(
    linestyle="None",
    marker="o",
    markersize=8,
    markeredgewidth=1.4,
    elinewidth=1.6,
    capsize=5,
    capthick=1.6,
)
BC_MODEL_STYLE = dict(
    linestyle="None", markerfacecolor="white", markeredgewidth=1.4, zorder=3
)

# ── Style helpers ─────────────────────────────────────────────────────────────


def apply_global_style():
    plt.rcParams.update(BASE_RC)


def style_axes(ax):
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.12, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)


def set_temperature_axis(ax, show_labels=True):
    ax.set_xticks(TEMP_TICKS_K)
    ax.set_xlim(*TEMP_LIMS_K)
    if not show_labels:
        ax.tick_params(axis="x", labelbottom=False)


def set_flux_axis(ax, run_name, for_bc=False):
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(BASE_RC["ytick.labelsize"])


def save_fig(fig, path: Path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")


def run_ylabel(run_name: str) -> str:
    return rf"Downstream flux ({RUN_UNIT.get(run_name, 'atoms s$^{{-1}}$')})"


# ── Data loading ──────────────────────────────────────────────────────────────


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
    jsim_col = _pick_col(df, ["J_sim", "Jsim", "J_out", "J_out_sim"])

    out = df[[case_col, run_col, tc_col, jsim_col]].copy()
    out = out.rename(
        columns={case_col: "case", run_col: "run", tc_col: "T_C", jsim_col: "J_1D"}
    )
    out["case"] = out["case"].astype(str).str.strip()
    out["run"] = out["run"].astype(str).str.strip()
    out["T_C"] = pd.to_numeric(out["T_C"], errors="coerce")
    out["J_1D"] = pd.to_numeric(out["J_1D"], errors="coerce") * SCALE_1D
    return out.dropna(subset=["case", "run", "T_C", "J_1D"])


def prepare_2d(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads the long-format jsim_jexp.csv (columns: run, case, type, config,
    T_C, series, value, error) and returns a wide-format DataFrame with
    columns [case, run, T_C, J_2D, J_exp, J_exp_error].
    """
    case_col = _pick_col(df, ["case"])
    run_col = _pick_col(df, ["run"])
    tc_col = _pick_col(df, ["T_C", "Tc", "temp_C", "Temperature_C"])
    series_col = _pick_col(df, ["series"])
    value_col = _pick_col(df, ["value"])
    error_col = _pick_col(df, ["error"])

    out = df[[case_col, run_col, tc_col, series_col, value_col, error_col]].copy()
    out = out.rename(
        columns={
            case_col: "case",
            run_col: "run",
            tc_col: "T_C",
            series_col: "series",
            value_col: "value",
            error_col: "error",
        }
    )
    for col in ("case", "run", "series"):
        out[col] = out[col].astype(str).str.strip()
    out["T_C"] = pd.to_numeric(out["T_C"], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["error"] = pd.to_numeric(out["error"], errors="coerce")
    out = out.dropna(subset=["case", "run", "T_C", "series", "value"])

    sim = out[out["series"] == "J_sim"][["case", "run", "T_C", "value"]].rename(
        columns={"value": "J_2D"}
    )
    exp = out[out["series"] == "J_exp"][
        ["case", "run", "T_C", "value", "error"]
    ].rename(columns={"value": "J_exp", "error": "J_exp_error"})

    merged = pd.merge(sim, exp, on=["case", "run", "T_C"], how="inner")
    merged["J_2D"] = pd.to_numeric(merged["J_2D"], errors="coerce") * SCALE_2D
    merged["J_exp"] = pd.to_numeric(merged["J_exp"], errors="coerce")
    merged["J_exp_error"] = pd.to_numeric(merged["J_exp_error"], errors="coerce")
    return merged


# ── Reusable plot primitives ──────────────────────────────────────────────────


def plot_experiment(ax, x, y, yerr, color, label=None, filled=False, zorder=5):
    ax.errorbar(
        x,
        y,
        yerr=yerr,
        color=color,
        ecolor=color,
        markeredgecolor=color,
        markerfacecolor=(color if filled else "white"),
        label=label,
        zorder=zorder,
        **EXP_STYLE,
    )


def plot_model_line(ax, x, y, color, style, label=None):
    ax.plot(x, y, color=color, markeredgecolor=color, label=label, **style)


def bc_case_offsets(n_cases, spread=8.0):
    return np.linspace(-spread, spread, n_cases) if n_cases > 1 else np.array([0.0])


def bc_legend_handles():
    h_handle = plt.errorbar(
        [],
        [],
        yerr=[[1], [1]],
        color="red",
        ecolor="red",
        markeredgecolor="red",
        markerfacecolor="white",
        label=r"Experiment (H)",
        **EXP_STYLE,
    )
    d_handle = plt.errorbar(
        [],
        [],
        yerr=[[1], [1]],
        color="black",
        ecolor="black",
        markeredgecolor="black",
        markerfacecolor="white",
        label=r"Experiment (D)",
        **EXP_STYLE,
    )
    return [
        Line2D(
            [0],
            [0],
            marker="s",
            color="0.3",
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=1.8,
            markersize=9,
            label=CASE_LABEL[CASES[0]],
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="0.3",
            linestyle="None",
            markerfacecolor="white",
            markeredgewidth=1.8,
            markersize=10,
            label=CASE_LABEL[CASES[1]],
        ),
        h_handle,
        d_handle,
    ]


# ── Figure 1 & 2: 1D vs 2D vs experiment ─────────────────────────────────────


def _plot_comparison_panel(ax, sub, run_name, case_name, panel_label, add_legend=False):
    color = RUN_COLOR[run_name]
    T = sub["T_C"].to_numpy(float) + 273.15

    plot_model_line(
        ax,
        T,
        sub["J_1D"].to_numpy(float),
        color,
        MODEL_1D_STYLE,
        label="1D FESTIM" if add_legend else None,
    )
    plot_model_line(
        ax,
        T,
        sub["J_2D"].to_numpy(float),
        color,
        MODEL_2D_STYLE,
        label="2D FESTIM" if add_legend else None,
    )
    plot_experiment(
        ax,
        T,
        sub["J_exp"].to_numpy(float),
        sub["J_exp_error"].to_numpy(float),
        color=color,
        label="Experiment" if add_legend else None,
    )

    set_temperature_axis(ax)
    set_flux_axis(ax, run_name)
    ax.set_title(f"{panel_label} {CASE_LABEL[case_name]}")
    style_axes(ax)


def make_two_panel_plot(run_name: str, out_pdf: Path):
    df1 = prepare_1d(_read_csv(CSV_1D))
    df2 = prepare_2d(_read_csv(CSV_2D))

    df1 = df1[df1["case"].isin(CASES) & (df1["run"] == run_name)]
    df2 = df2[df2["case"].isin(CASES) & (df2["run"] == run_name)]

    merged = pd.merge(df1, df2, on=["case", "run", "T_C"], how="inner")
    if merged.empty:
        raise RuntimeError(f"No merged data for run={run_name}.")

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TWO_PANEL, sharex=True, sharey=True)

    for i, (ax, case_name, label) in enumerate(zip(axes, CASES, ["(a)", "(b)"])):
        sub = merged[
            (merged["case"] == case_name) & (merged["run"] == run_name)
        ].sort_values("T_C")
        if sub.empty:
            ax.text(
                0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center"
            )
            style_axes(ax)
            continue
        _plot_comparison_panel(ax, sub, run_name, case_name, label, add_legend=(i == 0))

    fig.supxlabel("Temperature [K]")
    fig.supylabel(run_ylabel(run_name))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
        frameon=True,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, out_pdf)


# ── Figure 3: BC comparison ───────────────────────────────────────────────────


def _plot_bc_panel(ax, run_name: str, df_run: pd.DataFrame):
    df_run = df_run.sort_values(["T_C", "case"]).copy()
    temps_K = np.array(sorted(df_run["T_C"].dropna().unique()), dtype=float) + 273.15
    x = temps_K.copy()

    cases_present = [c for c in CASES if c in df_run["case"].unique()]
    offsets = bc_case_offsets(len(cases_present))

    for offset, case_name in zip(offsets, cases_present):
        sub = df_run[df_run["case"] == case_name].sort_values("T_C")
        ax.plot(
            x + offset,
            sub["J_2D"].to_numpy(float),
            marker=CASE_MARKER.get(case_name, "o"),
            color=RUN_COLOR[run_name],
            **BC_MODEL_STYLE,
        )

    exp_sub = (
        df_run.groupby("T_C", as_index=False)
        .agg({"J_exp": "first", "J_exp_error": "first"})
        .sort_values("T_C")
    )
    plot_experiment(
        ax,
        x,
        exp_sub["J_exp"].to_numpy(float),
        exp_sub["J_exp_error"].to_numpy(float),
        color=RUN_COLOR[run_name],
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(T)}" for T in temps_K])
    set_flux_axis(ax, run_name, for_bc=True)
    style_axes(ax)


def make_bc_marker_panel(out_pdf: Path):
    df2 = prepare_2d(_read_csv(CSV_2D))
    df2 = df2[df2["case"].isin(CASES) & df2["run"].isin(RUNS)].copy()

    if df2.empty:
        raise RuntimeError("No 2D/experiment data found for Run 2 / Run 3.")

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_TWO_PANEL, sharex=True)

    for ax, run_name, label in zip(axes, RUNS, ["(a) H", "(b) D"]):
        _plot_bc_panel(ax, run_name, df2[df2["run"] == run_name].copy())
        ax.text(
            0.5,
            1.01,
            label,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=15,
        )

    fig.supxlabel("Temperature [K]")
    fig.supylabel(r"Downstream flux [atom s$^{-1}$]")

    handles = bc_legend_handles()
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.95),
        frameon=True,
        columnspacing=2.0,
        handletextpad=0.8,
        borderpad=0.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    save_fig(fig, out_pdf)


# ── Figure 4: experimental transient at 600 C ─────────────────────────────────


def make_exp_600_plot():
    if not XLSX_EXP_600.exists():
        print(f"[skip] {XLSX_EXP_600} not found")
        return

    df = pd.read_excel(XLSX_EXP_600)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Time (h)": "time_h", "Flux (H/s)": "flux"})

    fig, ax = plt.subplots(figsize=FIGSIZE_EXP)
    ax.plot(
        df["time_h"],
        df["flux"],
        color=RUN_COLOR["Run 2"],
        linestyle="-",
        linewidth=1.5,
        marker="o",
        markersize=4,
        markerfacecolor=RUN_COLOR["Run 2"],
        markeredgecolor=RUN_COLOR["Run 2"],
        label="Experiment, 873 K",
    )

    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"Flux [$\mathrm{H\,s^{-1}}$]")
    style_axes(ax)
    fig.tight_layout()
    save_fig(fig, OUT_EXP_600_PDF)


# ── Main ──────────────────────────────────────────────────────────────────────

apply_global_style()
make_two_panel_plot("Run 2", OUT_H_PDF)
make_two_panel_plot("Run 3", OUT_D_PDF)
make_bc_marker_panel(OUT_BC_PDF)
make_exp_600_plot()
