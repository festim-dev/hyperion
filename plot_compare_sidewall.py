import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import morethemes as mt

mt.set_theme("lumen")

# ── Paths ─────────────────────────────────────────────────────────────────────

CSV_FILE = Path("results") / "percentage_metrics.csv"
OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
    }
)

RUN_COLOR = {
    "Run 2": "red",
    "Run 3": "black",
}

RUN_LABEL = {
    "Run 2": "H",
    "Run 3": "D",
}

CASE_LABEL = {
    "swap_infinite": "Ideal coating",
    "swap_transparent": "Uncoated",
}

CASE_MARKER = {
    "swap_infinite": "s",
    "swap_transparent": "^",
}

# ── Data ──────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_FILE)
df["T_C"] = pd.to_numeric(df["T_C"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["T_K"] = df["T_C"] + 273.15

df = df[df["case"].isin(["swap_infinite", "swap_transparent"])].copy()
df = df[df["run"].isin(["Run 2", "Run 3"])].copy()
df = df.dropna(subset=["T_C", "value"])

if df.empty:
    raise RuntimeError(
        "No valid swap Run 2 / Run 3 data found in percentage_metrics.csv"
    )

# ── Generic plotter ───────────────────────────────────────────────────────────


def make_plot(metric_key: str, ylabel: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6.2))

    for run in ["Run 2", "Run 3"]:
        for case in ["swap_infinite", "swap_transparent"]:
            sub = df[
                (df["run"] == run)
                & (df["case"] == case)
                & (df["metric_key"] == metric_key)
            ].sort_values("T_C")
            if sub.empty:
                continue
            ax.plot(
                sub["T_K"],
                sub["value"],
                color=RUN_COLOR[run],
                marker=CASE_MARKER[case],
                linestyle="-",
                linewidth=1.8,
                markersize=10,
                markerfacecolor="none",
                label=f"{RUN_LABEL[run]} - {CASE_LABEL[case]}",
            )

    ax.set_xlabel("Temperature [K]")
    ax.set_xlim(768, 978)
    ax.set_xticks([773, 823, 873, 923, 973])
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {outfile}")


# ── Figures ───────────────────────────────────────────────────────────────────

make_plot(
    metric_key="pct_sidewall_leak",
    ylabel="Sidewall loss / upstream flux [%]",
    outfile=OUTDIR / "sidewall_loss.pdf",
)

make_plot(
    metric_key="pct_sidewall_comp",
    ylabel="Sidewall contribution / downstream flux [%]",
    outfile=OUTDIR / "sidewall_contribution.pdf",
)
