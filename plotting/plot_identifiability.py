"""Figures for the (D, K_H) identifiability study.

Reads the CSVs written by identifiability.py and identifiability_transient.py.

    identifiability_degeneracy.pdf/.png
        (a) K_H compensates D exactly     (b) Phi is invariant
        (c) the modelled flux is constant along D * K_H = const

    identifiability_transient.pdf/.png
        (a) the steady flux does not resolve D   (b) the time lag does

Run from the repository root, after both study scripts:
    python plot_identifiability.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

OUTDIR = Path(__file__).resolve().parents[1] / "results"

# Categorical slots 1-2 from the validated palette; the two BC envelopes are the
# only categorical dimension in these figures, so identity never relies on more
# than two hues and both series are directly labelled.
C_IDEAL = "#2a78d6"      # slot 1, blue   -- swap_infinite   (ideal coating)
C_UNCOAT = "#eb6834"     # slot 2, orange -- swap_transparent (uncoated)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8a85"

CASE_STYLE = {
    "swap_infinite": (C_IDEAL, "ideal coating"),
    "swap_transparent": (C_UNCOAT, "uncoated"),
}

# Plotting vocabulary taken from the paper's own figure scripts
# (plot_comparison.py, plot_perm_fits.py) so these figures read as part of the
# same set: the outer-wall boundary condition is carried by the marker shape,
# colour is left for the remaining dimension, and markers are open with a
# coloured edge.  The two colours are the paper's own literature palette.
CASE_MARKER = {"swap_infinite": "s", "swap_transparent": "^"}
CASE_LABEL = {"swap_infinite": "Ideal coating", "swap_transparent": "Uncoated"}
PAPER_COLORS = ["#4C72B0", "#DD8452", "#55A868"]

# Diverging ramp: blue <-> red with the neutral gray midpoint of the reference
# palette.  Used for a signed quantity (flux above/below the measurement).
DIVERGING = LinearSegmentedColormap.from_list(
    "skill_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a3a2", "#e34948", "#8f2020"],
)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "grid.color": "#b8bcc4",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.8,
    "axes.linewidth": 1.0,
    "axes.grid": True,
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


def _style_axes(ax) -> None:
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)


def _load_normalised() -> pd.DataFrame:
    """Every inversion, normalised by its own nominal-D result."""
    inv = pd.read_csv(OUTDIR / "identifiability_inversion.csv")
    inv = inv[inv["ok"].astype(str).str.lower().isin({"true", "1"})].copy()
    ref = (inv[inv["D_mult"] == 1.0]
           .set_index(["case", "run", "T_C"])[["K_H", "phi"]]
           .rename(columns={"K_H": "K_H_ref", "phi": "phi_ref"}))
    inv = inv.join(ref, on=["case", "run", "T_C"])
    inv["K_H_rel"] = inv["K_H"] / inv["K_H_ref"]
    inv["phi_rel"] = inv["phi"] / inv["phi_ref"]
    return inv


# Sequential blue ramp for temperature, an ordered variable.  Steps from the
# reference palette, spaced for separation on a light ground.
T_RAMP = {500.0: "#86b6ef", 550.0: "#3987e5", 600.0: "#256abf",
          650.0: "#184f95", 700.0: "#0d366b"}
RUN_MARKER = {"Run 1": "o", "Run 2": "^"}


def figure_recovered(case: str, suffix: str,
                     T_list: tuple = (500.0, 700.0)) -> None:
    """The recovered parameters themselves, in absolute units, unaveraged.

    One boundary condition per figure, at the two ends of the measured
    temperature range, with both runs shown.  Every marker is an independent
    inversion carrying its own jittered search bracket, so the five points along
    a series had to converge to the same Phi rather than being forced there by a
    shared search path.
    """
    inv = pd.read_csv(OUTDIR / "identifiability_inversion.csv")
    inv = inv[inv["ok"].astype(str).str.lower().isin({"true", "1"})]
    inv = inv[(inv["case"] == case) & (inv["T_C"].isin(T_list))]
    inv = inv.sort_values(["T_C", "run", "D_mult"])
    label = CASE_STYLE[case][1]

    # Two steps of the sequential ramp: temperature is ordered, not categorical.
    t_color = {min(T_list): "#86b6ef", max(T_list): "#0d366b"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))

    x = np.logspace(np.log10(inv["D_mult"].min()) - 0.12,
                    np.log10(inv["D_mult"].max()) + 0.12, 100)
    for (T_C, run), g in inv.groupby(["T_C", "run"]):
        color = t_color[T_C]
        g = g.sort_values("D_mult")
        ref = g.loc[np.isclose(g["D_mult"], 1.0)]
        # Guide lines through each series' own reference point, drawn behind the
        # markers: K_H = Phi_ref / D in (a), Phi = Phi_ref in (b).  A departure
        # from the degeneracy would show as markers leaving their own line.
        axes[0].plot(x, float(ref["K_H"].iloc[0]) / x, color=MUTED, lw=1.0,
                     zorder=1)
        axes[1].axhline(float(ref["phi"].iloc[0]), color=MUTED, lw=1.0, zorder=1)
        for ax, col in ((axes[0], "K_H"), (axes[1], "phi")):
            ax.plot(g["D_mult"], g[col], RUN_MARKER[run], ms=6.5, ls="none",
                    mfc="white", mec=color, mew=1.5, zorder=3)

    for ax in axes:
        _style_axes(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(r"prescribed $D\,/\,D_{\mathrm{ref}}$")

    axes[0].set_ylabel(r"recovered $K_{\mathrm{H}}$"
                       "\n" r"[H m$^{-3}$ Pa$^{-1}$]")
    axes[0].set_title("(a)  Henry coefficient", loc="left")
    axes[1].set_ylabel(r"recovered $\Phi = D\,K_{\mathrm{H}}$"
                       "\n" r"[H m$^{-1}$ s$^{-1}$ Pa$^{-1}$]")
    axes[1].set_title("(b)  Permeability", loc="left")

    # Give (b) the same number of decades as (a), so "flat" is judged on the same
    # scale on which (a) falls by two decades -- otherwise the contrast is an
    # artefact of one axis being stretched.
    a_lo, a_hi = axes[0].get_ylim()
    span = np.log10(a_hi / a_lo)
    b_lo, b_hi = axes[1].get_ylim()
    mid = 10 ** (0.5 * (np.log10(b_lo) + np.log10(b_hi)))
    axes[1].set_ylim(mid / 10 ** (span / 2), mid * 10 ** (span / 2))

    t_handles = [Line2D([0], [0], color=t_color[T], lw=0, marker="s", ms=6.5,
                        mfc=t_color[T], mec=t_color[T],
                        label=f"{int(T)} °C ({T + 273.15:.0f} K)")
                 for T in sorted(T_list)]
    r_handles = [Line2D([0], [0], color=MUTED, lw=0, marker=RUN_MARKER[r],
                        ms=6.5, mfc="white", mec=MUTED, mew=1.5, label=r)
                 for r in sorted(set(inv["run"]))]
    axes[0].legend(handles=t_handles, loc="lower left", frameon=False,
                   fontsize=7.5, handletextpad=.5, borderaxespad=.4)
    axes[1].legend(handles=r_handles, loc="lower left", frameon=False,
                   fontsize=7.5, handletextpad=.5, borderaxespad=.4)

    fig.text(0.0, 1.0, f"{label} outer wall", ha="left", va="top",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = f"identifiability_recovered{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}")
    print(f"[saved] {OUTDIR / stem}.pdf (+ .png)  [{label}, "
          f"{'/'.join(str(int(t)) for t in sorted(T_list))} C]")
    plt.close(fig)


def figure_recovered_bc(T_list: tuple = (500.0, 700.0), run: str = "Run 2") -> None:
    """SWAP data, one run, with the outer-wall boundary condition as the contrast.

    Encoding follows the paper's own figure scripts: the boundary condition is the
    marker shape (square = ideal coating, triangle = uncoated) and colour carries
    the remaining dimension, here temperature, in the paper's literature palette.
    Markers are open with a coloured edge, as in plot_perm_fits.py.
    """
    inv = pd.read_csv(OUTDIR / "identifiability_inversion.csv")
    inv = inv[inv["ok"].astype(str).str.lower().isin({"true", "1"})]
    inv = inv[(inv["T_C"].isin(T_list)) & (inv["run"] == run)]
    inv = inv.sort_values(["case", "T_C", "D_mult"])

    t_color = {T: PAPER_COLORS[i] for i, T in enumerate(sorted(T_list))}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))

    x = np.logspace(np.log10(inv["D_mult"].min()) - 0.12,
                    np.log10(inv["D_mult"].max()) + 0.12, 100)
    for (case, T_C), g in inv.groupby(["case", "T_C"]):
        color, marker = t_color[T_C], CASE_MARKER[case]
        g = g.sort_values("D_mult")
        ref = g.loc[np.isclose(g["D_mult"], 1.0)]
        axes[0].plot(x, float(ref["K_H"].iloc[0]) / x, color=color, lw=0.9,
                     alpha=.28, zorder=1)
        axes[1].axhline(float(ref["phi"].iloc[0]), color=color, lw=0.9,
                        alpha=.28, zorder=1)
        for ax, col in ((axes[0], "K_H"), (axes[1], "phi")):
            ax.plot(g["D_mult"], g[col], marker, ms=8, ls="none",
                    mfc="white", mec=color, mew=1.6, zorder=3)

    for ax in axes:
        _style_axes(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel(r"prescribed $D\,/\,D_{\mathrm{ref}}$")

    axes[0].set_ylabel(r"$K_{\mathrm{H}}$  [H m$^{-3}$ Pa$^{-1}$]")
    axes[0].set_title(r"(a)  Recovered $K_{\mathrm{H}}$")
    axes[1].set_ylabel(r"$\Phi$  [H m$^{-1}$ s$^{-1}$ Pa$^{-1}$]")
    axes[1].set_title(r"(b)  Inferred $\Phi = D\,K_{\mathrm{H}}$")

    lo, hi = inv["phi"].min(), inv["phi"].max()
    axes[1].set_ylim(lo / 10 ** 0.5, hi * 10 ** 0.5)
    axes[1].yaxis.set_minor_formatter(mticker.NullFormatter())

    bc_handles = [Line2D([0], [0], color="0.35", lw=0, marker=CASE_MARKER[c],
                         ms=8, mfc="white", mec="0.35", mew=1.6, label=CASE_LABEL[c])
                  for c in CASE_STYLE]
    t_handles = [Line2D([0], [0], color=t_color[T], lw=0, marker="o", ms=8,
                        mfc="white", mec=t_color[T], mew=1.6,
                        label=f"{int(T)} °C ({T + 273.15:.0f} K)")
                 for T in sorted(T_list)]
    # One boxed legend above both panels, following the convention of the
    # simulation-vs-experiment figures in plot_comparison.py.
    handles = bc_handles + t_handles
    fig.legend(handles, [h.get_label() for h in handles], loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=4, frameon=True,
               columnspacing=2.0, handletextpad=0.8, borderpad=0.5)

    # Quiet labels for the behaviour each panel shows; no arrows, no boxes.
    axes[0].text(0.97, 0.95, r"$K_{\mathrm{H}} \propto D^{-1}$", ha="right",
                 va="top", transform=axes[0].transAxes, color=INK2, fontsize=10)
    axes[1].text(0.97, 0.95, r"$\Phi$ = const.", ha="right", va="top",
                 transform=axes[1].transAxes, color=INK2, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"identifiability_recovered_bc.{ext}")
    print(f"[saved] {OUTDIR}/identifiability_recovered_bc.pdf (+ .png)  [{run}]")
    plt.close(fig)


def figure_degeneracy(case: str | None = None, suffix: str = "") -> None:
    """Two panels: the compensation, and the invariant it leaves behind.

    `case` restricts the figure to one outer-wall limit; None pools both.  The
    normalisation is per experimental point, and K_H/K_H_ref = (D/D_ref)^-1 is an
    identity once Phi is invariant, so every point collapses onto the same five
    locations regardless of temperature, run or boundary condition -- the two
    limits differ by 6e-17.  One marker set therefore carries every inversion in
    the figure, and the condition it covers is named on the figure rather than
    distinguished by symbol.
    """
    inv = _load_normalised()
    if case is not None:
        inv = inv[inv["case"] == case]
    label = CASE_STYLE[case][1] if case else "both outer-wall limits"
    n_pts = inv[inv["D_mult"] == 1.0].shape[0]
    T_lo, T_hi = inv["T_C"].min(), inv["T_C"].max()

    xs = np.array(sorted(inv["D_mult"].unique()))
    K_H_rel = np.array([inv.loc[inv["D_mult"] == m, "K_H_rel"].mean() for m in xs])
    phi_rel = np.array([inv.loc[inv["D_mult"] == m, "phi_rel"].mean() for m in xs])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25))
    fig.text(0.0, 1.0, f"{label} — {n_pts} experimental points, "
                       f"{T_lo:.0f}–{T_hi:.0f} °C",
             ha="left", va="top", fontsize=8, color=INK2)

    # -- (a) the compensation ---------------------------------------------
    ax = axes[0]
    _style_axes(ax)
    xline = np.logspace(np.log10(xs.min()) - 0.12, np.log10(xs.max()) + 0.12, 100)
    ax.plot(xline, 1.0 / xline, color=MUTED, lw=1.3, zorder=1)
    ax.annotate(r"$(D/D_{\mathrm{ref}})^{-1}$", xy=(xs[3], 1.0 / xs[3]),
                xytext=(10, 10), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.plot(xs, K_H_rel, "o", ms=6.5, mfc="white", mec=C_IDEAL, mew=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D\,/\,D_{\mathrm{ref}}$")
    ax.set_ylabel(r"$K_{\mathrm{H}}\,/\,K_{\mathrm{H,ref}}$")
    ax.set_title("(a)  Recovered Henry coefficient", loc="left")

    # -- (b) the invariant -------------------------------------------------
    ax = axes[1]
    _style_axes(ax)
    ax.axhline(1.0, color=MUTED, lw=1.3, zorder=1)
    ax.plot(xs, phi_rel, "o", ms=6.5, mfc="white", mec=C_IDEAL, mew=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0.98, 1.02)
    ax.set_yticks([0.98, 0.99, 1.00, 1.01, 1.02])
    ax.set_xlabel(r"$D\,/\,D_{\mathrm{ref}}$")
    ax.set_ylabel(r"$\Phi\,/\,\Phi_{\mathrm{ref}}$")
    ax.set_title("(b)  Recovered permeability", loc="left")

    for ax in axes:
        ax.set_xlim(xline[0], xline[-1])

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = f"identifiability_degeneracy{suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}")
    print(f"[saved] {OUTDIR / stem}.pdf (+ .png)  [{label}]")
    plt.close(fig)


def figure_landscape() -> None:
    """The (D, K_H) misfit landscape, kept as a standalone supplementary figure."""
    land = pd.read_csv(OUTDIR / "identifiability_landscape.csv")
    # The coated-wall case keeps the downstream flux positive over the whole grid;
    # with the uncoated wall the bypass floor is negative, so J changes sign at low
    # permeability and log(J/J_exp) is undefined there.
    case = "swap_infinite" if (land["case"] == "swap_infinite").any() \
        else land["case"].iloc[0]
    sub = land[land["case"] == case]
    a = np.array(sorted(sub["a_D"].unique()))
    b = np.array(sorted(sub["b_KH"].unique()))
    Z = np.full((len(b), len(a)), np.nan)
    for _, r in sub.iterrows():
        if r["J"] > 0:
            Z[np.searchsorted(b, r["b_KH"]), np.searchsorted(a, r["a_D"])] = \
                np.log10(r["J"] / r["J_exp"])

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    _style_axes(ax)
    lim = float(np.nanmax(np.abs(Z)))
    pc = ax.pcolormesh(np.log10(a), np.log10(b), Z, cmap=DIVERGING,
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                       shading="nearest")
    cs = ax.contour(np.log10(a), np.log10(b), Z, levels=[0.0],
                    colors=[INK], linewidths=2.4)
    ax.clabel(cs, fmt={0.0: r"$J=J_{\exp}$"}, fontsize=7.5)
    # J depends on the product only, so every contour is a straight line of slope
    # -1; drawn dashed on top of the computed contour, which it coincides with.
    la = np.log10(a)
    ax.plot(la, -la, color="white", lw=1.1, ls=(0, (4, 3)), zorder=5,
            label=r"$D\,K_{\mathrm{H}}$ = const")
    ax.legend(loc="lower left", frameon=False, fontsize=7.5,
              handletextpad=0.5, labelcolor=INK)
    ax.set_xlabel(r"$\log_{10}\,(D\,/\,D_{\mathrm{ref}})$")
    ax.set_ylabel(r"$\log_{10}\,(K_{\mathrm{H}}\,/\,K_{\mathrm{H,ref}})$")
    cb = fig.colorbar(pc, ax=ax, pad=0.02)
    cb.set_label(r"$\log_{10}\,(J_{\mathrm{model}}/J_{\exp})$", fontsize=8)
    cb.outline.set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"identifiability_landscape.{ext}")
    print(f"[saved] {OUTDIR / 'identifiability_landscape.pdf'} (+ .png)")
    plt.close(fig)


def figure_transient() -> None:
    fwd = pd.read_csv(OUTDIR / "identifiability_forward.csv")
    tr = pd.read_csv(OUTDIR / "identifiability_transient.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))

    # -- (a) steady flux is blind to D (2-D model, four decades) -----------
    # Plotted as a ratio against a line at unity rather than as a deviation on a
    # 1e-13 axis: the point is that the flux does not move, not how many digits
    # the solver agrees to.  That number belongs in the caption.
    ax = axes[0]
    _style_axes(ax)
    ax.axhline(1.0, color=MUTED, lw=1.3, zorder=1)
    xs = np.array(sorted(fwd["D_mult"].unique()))
    rel = []
    for m in xs:
        s = fwd[fwd["D_mult"] == m]
        base = fwd[fwd["D_mult"] == 1.0].set_index(["case", "run", "T_C"])["J"]
        rel.append(np.mean([row["J"] / base.loc[(row["case"], row["run"], row["T_C"])]
                            for _, row in s.iterrows()]))
    ax.plot(xs, rel, "o", ms=6.5, mfc="white", mec=C_IDEAL, mew=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_ylim(0.98, 1.02)
    ax.set_yticks([0.98, 0.99, 1.00, 1.01, 1.02])
    ax.set_xlabel(r"$D\,/\,D_{\mathrm{ref}}$   (at fixed $\Phi$)")
    ax.set_ylabel(r"$J_{\mathrm{ss}}\,/\,J_{\mathrm{ss,ref}}$")
    ax.set_title("(a)  Steady-state flux", loc="left")

    # -- (b) the transient does resolve D ----------------------------------
    ax = axes[1]
    _style_axes(ax)
    ax.plot(tr["D_mult"], tr["t_lag_theory_s"] / 60.0, "-", lw=1.3,
            color=MUTED, zorder=1)
    ax.plot(tr["D_mult"], tr["t_lag_num_s"] / 60.0, "o", ms=6.5, mfc="white",
            mec=C_IDEAL, mew=1.5, zorder=3)
    ax.annotate(r"$L^2/6D$", xy=(tr["D_mult"].iloc[1],
                tr["t_lag_theory_s"].iloc[1] / 60.0),
                xytext=(10, 8), textcoords="offset points",
                color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$D\,/\,D_{\mathrm{ref}}$   (at fixed $\Phi$)")
    ax.set_ylabel("permeation time lag  [min]")
    ax.set_title("(b)  Permeation time lag", loc="left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"identifiability_transient.{ext}")
    print(f"[saved] {OUTDIR / 'identifiability_transient.pdf'} (+ .png)")
    plt.close(fig)


if __name__ == "__main__":
    figure_recovered_bc()
    figure_recovered("swap_infinite", "_ideal_coating")
    figure_recovered("swap_transparent", "_uncoated")
    figure_degeneracy()                                        # both limits pooled
    figure_degeneracy("swap_infinite", "_ideal_coating")
    figure_degeneracy("swap_transparent", "_uncoated")
    figure_landscape()
    figure_transient()
