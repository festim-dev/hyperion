"""
Hydrogen concentration fields: the six panels of the domain figure.

Three rows -- full domain, salt region, solid region -- against two columns,
uncoated on the left and ideal coating on the right, with one colour bar per
row. Scales differ between rows but are shared within a row, which is what
makes the left/right comparison meaningful.

Three choices differ from the earlier ParaView exports:

  * VIEW_SALT looks at the outer corner, r > 28.5 mm, rather than at
    mid-channel. Uncoated and ideal salt fields agree to four decimal places
    everywhere inside r = 28.6 mm, since the coating acts on the metal and the
    salt only sees it where the two touch; at the wall the ratio reaches 58.
  * Every row is logarithmic, so the three bars carry the same 10^n ticks. On a
    linear scale the metal rows collapse: four fifths of their nodes sit in the
    bottom tenth of the bar.
  * Each floor is derived rather than typed in -- the decade at or below the
    row's FLOOR_PERCENTILE-th percentile of positive values. This puts the
    metal rows at 10^22; below 10^20 every value is exactly zero, on the
    downstream face of the uncoated wall where the Sieverts condition pins it.

Geometry names and coordinates follow mesh.py, as in plot_domain.py:

    Omega_FLiBe   salt, tag 1, vol_1 in the .bp     r < 39, 24 < y < 29.14
    Omega_Ni      wall, both caps and membrane, tag 2, vol_2

Input:  results/out-species_vol_{1,2}_{uncoated,ideal_coating}.bp
Output: results/concentration_fields.pdf/.svg/.png/_2x.png
"""

from __future__ import annotations

from pathlib import Path

import adios2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, MaxNLocator
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "concentration_fields"

# ── House style, same block as plot_domain.py ────────────────────────────────

_HAVE_ARIAL = "Arial" in {f.name for f in fm.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom" if _HAVE_ARIAL else "dejavusans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
if _HAVE_ARIAL:
    plt.rcParams.update({"mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
                         "mathtext.bf": "Arial:bold"})

PT_CAP, PT_BAR, PT_BOX, PT_SCALE = 9.0, 8.5, 9.5, 7.5
PT_BAR_LABEL = 11.0  # the colour bar title, set larger than its tick labels

INK = "#222222"
C_SALT_BOX = "#E10600"   # the original figure's red and blue, kept
C_SOLID_BOX = "#0B41CD"
CMAP = "viridis"

# ── Geometry, in millimetres, exactly as mesh.py sets it ─────────────────────

R_IN, R_OUT = 39.0, 41.0
Y_MB, Y_MT, Y_FT = 22.0, 24.0, 29.14

# Viewports, (r0, r1, y0, y1) in mm. The full-domain view is cropped above
# 45 mm: the column runs to 111 mm and the top cap is uniform, so the rest is
# white space. domain_map.pdf carries the full column and every coordinate.
# The salt window ends where the wall begins, so the two zoom outlines share
# that edge rather than crossing.
MARK_LEFT = 23.0  # mm, the left edge both outlines stand on

VIEW_FULL = (0.0, R_OUT + 2.5, 0.0, 45.0)
VIEW_SALT = (MARK_LEFT, R_IN, Y_MT, Y_FT)
VIEW_SOLID = (MARK_LEFT, R_OUT, 19.0, 34.0)

CASES = [("uncoated", "uncoated"), ("ideal_coating", "ideal coating")]

PANEL_W = 2.75  # inches; every panel is this wide, heights follow from aspect

# The right margin holds the colour bars. The figure is saved at its own size
# rather than cropped tight, so these margins are what the PDF has.
MARGIN_L, MARGIN_R = 0.02  , 0.95  # inches
PANEL_GAP = 0.35                   # inches between the two columns

BAR_LABEL_X = 1.27  # axes fraction; where every colour bar title stands

# How far past the wall the solid outline runs, on its right edge only, so it
# does not disappear into the wall's own outer face at r = 41.
MARK_OVERSHOOT = 0.5  # mm

# Where a logarithmic bar's floor goes: the decade at or below this percentile
# of the row's positive values. At 1 the salt row floors at 10^23 and spans
# 2.2 decades; at 0.1 it would floor at 10^22 and wash the salt out over three
# decades to gain a quarter of a percent of nodes.
FLOOR_PERCENTILE = 1.0

# One colour bar for all six panels instead of one per row.  Its limits come
# from the same rule the per-row scales use, applied to every panel's data at
# once: floor at the FLOOR_PERCENTILE decade, top at the true maximum.  Only
# the floor clamps, and main() prints how much each row loses to it.
SHARED_SCALE = False
SHARED_BAR_HEIGHT = 0.58  # fraction of the panel column the shared bar spans

# One scale bar length for all six panels, so the reader compares magnifications
# by eye instead of re-reading a number under every picture.  5 mm is the only
# round length that is neither lost in the full-domain view nor wider than the
# salt window: 12% of (a, b), 48% of (c, d), 33% of (e, f).
SCALE_LENGTH = 5.0  # mm


# ── Data ─────────────────────────────────────────────────────────────────────


def load_field(case: str, vol: int):
    """Triangulation in mm plus the nodal concentration, from one .bp."""
    path = RESULTS / f"out-species_vol_{vol}_{case}.bp"
    if not path.exists():
        raise FileNotFoundError(f"Missing field: {path}")
    with adios2.FileReader(str(path)) as reader:
        geom = reader.read("geometry")
        conn = reader.read("connectivity")
        value = reader.read(f"H_{vol}").ravel()
    # connectivity rows are [n_points, i, j, k]; every cell is a triangle here.
    tri = Triangulation(geom[:, 0] * 1e3, geom[:, 1] * 1e3, conn[:, 1:4])
    return tri, value


def inside(tri, value, view):
    """The nodal values that fall within a viewport."""
    r0, r1, y0, y1 = view
    m = (tri.x >= r0) & (tri.x <= r1) & (tri.y >= y0) & (tri.y <= y1)
    return value[m]


def row_scale(fields, view, log: bool):
    """Colour limits taken from the data the panels actually show.

    Linear: zero to the data maximum.  Logarithmic: the decade at or below the
    0.1st percentile of the
    positive values, up to the data maximum.  Returns the norm and the fraction
    of nodes the floor cuts off, which the caller reports on stdout.
    """
    v = np.concatenate([inside(tri, value, view) for tri, value in fields])
    vmax = float(v.max())
    if not log:
        # The maximum itself, not rounded up: a rounded top forces an awkward
        # half-decade tick (1.5x10^25) and leaves the head of the bar unused.
        return Normalize(0.0, vmax), 0.0
    positive = v[v > 0]
    floor = 10.0 ** np.floor(np.log10(np.percentile(positive, FLOOR_PERCENTILE)))
    return LogNorm(float(floor), vmax), float((v < floor).mean())


def decade_ticks(norm):
    lo = int(np.ceil(np.log10(norm.vmin)))
    hi = int(np.floor(np.log10(norm.vmax)))
    return [10.0**k for k in range(lo, hi + 1)]


# ── Drawing ──────────────────────────────────────────────────────────────────


def field_cmap():
    """Viridis, with anything off either end folded into that end's colour.

    No separate colour for the clamped values and no block on the bar: past a
    tick at either end is simply one flat colour, the one that tick already
    carries.  The bar stays a single uninterrupted gradient.
    """
    base = matplotlib.colormaps[CMAP]
    cmap = base.with_extremes(under=base(0.0), over=base(1.0))
    cmap.set_bad(base(0.0))
    return cmap


def draw_field(ax, fields, norm, view, mesh=False):
    """Paint one panel: every (tri, value) pair under a shared norm."""
    cmap = field_cmap()
    for tri, value in fields:
        v = value
        if isinstance(norm, LogNorm):
            # A log scale cannot place the exact zeros on the uncoated
            # downstream face.  Pushing them below the floor gives them the
            # same flat colour every sub-floor value gets.
            v = np.where(value > 0, value, norm.vmin / 10.0)
        ax.tripcolor(tri, v, shading="gouraud", cmap=cmap, norm=norm,
                     rasterized=True)
        if mesh:
            ax.triplot(tri, lw=0.12, color="0.15", alpha=0.55, rasterized=True)

    r0, r1, y0, y1 = view
    ax.set_xlim(r0, r1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()


def add_scale_bar(ax, view):
    """Something to measure the panel against, since it carries no axes.

    Hung below the panel rather than laid over it: the salt row is field edge
    to edge, so anywhere inside it would cover data.  transData still sets the
    length, so the bar stays true to the panel it belongs to.
    """
    bar = AnchoredSizeBar(
        ax.transData, SCALE_LENGTH, f"{SCALE_LENGTH:g} mm", loc="upper left",
        bbox_to_anchor=(0.0, 0.0), bbox_transform=ax.transAxes,
        pad=0.0, borderpad=0.15, sep=2.0, frameon=False, color=INK,
        size_vertical=0.06 * SCALE_LENGTH,
        fontproperties=fm.FontProperties(size=PT_SCALE),
    )
    ax.add_artist(bar)


def sci_tick(value, _pos=None) -> str:
    """A tick that carries its own power of ten, as the log bars' ticks do."""
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / 10.0**exp
    if round(abs(mant), 1) >= 10.0:  # guard against 9.99 -> "10.0"
        exp += 1
        mant = value / 10.0**exp
    m = f"{mant:.1f}".rstrip("0").rstrip(".")
    return rf"$10^{{{exp}}}$" if m == "1" else rf"${m}\times 10^{{{exp}}}$"


BAR_TITLE = r"Concentration [H m$^{-3}$]"


def style_colorbar(cax, norm):
    """Paint and label a bar into an axes someone else has placed.

    Every tick carries its own power of ten, log and linear alike, so the bars
    read the same way and none needs a factor floating above it.  Nothing is
    appended at either end for the clamped values: field_cmap folds them into
    the colours the end ticks already name.
    """
    bar = plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=field_cmap()), cax=cax,
    )
    if isinstance(norm, LogNorm):
        bar.set_ticks(FixedLocator(decade_ticks(norm)))
        bar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    else:
        bar.ax.yaxis.set_major_locator(
            MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10])
        )
        bar.ax.yaxis.set_major_formatter(FuncFormatter(sci_tick))
    bar.ax.minorticks_off()
    bar.ax.tick_params(labelsize=PT_BAR, length=2.5, width=0.6, color=INK)
    bar.outline.set_linewidth(0.6)
    bar.outline.set_edgecolor(INK)
    return bar


def add_row_colorbar(ax, norm):
    """A bar hung off one panel's right edge, matched to its drawn height."""
    bar = style_colorbar(ax.inset_axes([1.04, 0.0, 0.045, 1.0]), norm)
    # Placed against the panel, not against the tick labels, so the titles
    # stand on one vertical line however wide a row's ticks run.
    ax.text(BAR_LABEL_X, 0.5, BAR_TITLE, transform=ax.transAxes, rotation=90,
            ha="center", va="center", fontsize=PT_BAR_LABEL, color=INK)
    return bar


def add_figure_colorbar(fig, panels, norm):
    """One bar for the whole figure, as tall as the column of panels it serves."""
    fig.canvas.draw()
    boxes = [ax.get_position() for ax in panels]
    pw = boxes[0].x1 - boxes[0].x0
    x1 = max(b.x1 for b in boxes)
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    # Shortened and re-centred: run edge to edge and the bar reads as a fourth
    # column of the figure rather than as a legend beside it.
    height = (y1 - y0) * SHARED_BAR_HEIGHT
    bar = style_colorbar(
        fig.add_axes([x1 + 0.04 * pw, (y0 + y1 - height) / 2,
                      0.045 * pw, height]), norm
    )
    fig.text(x1 + (BAR_LABEL_X - 1.0) * pw, (y0 + y1) / 2, BAR_TITLE,
             rotation=90, ha="center", va="center", fontsize=PT_BAR_LABEL,
             color=INK)
    return bar


def solid_outline(view, m=MARK_OVERSHOOT):
    """The metal inside the solid viewport, as a closed path.

    A rectangle over this window would enclose the salt as well, and the salt
    is precisely what row 3 is not showing.  What is actually there is the
    outer wall, r 39 to 41 over the window's full height, plus the membrane,
    r < 39 between y 22 and 24 -- so the outline is the corner they form, with
    only the right edge carried past the wall.
    """
    r0, r1, y0, y1 = view
    return [
        (R_IN, y0), (r1 + m, y0), (r1 + m, y1), (R_IN, y1),
        (R_IN, Y_MT), (r0, Y_MT), (r0, Y_MB), (R_IN, Y_MB),
    ]


def mark_zoom(ax, labels=True):
    """The two windows that rows 2 and 3 magnify, drawn on panel (a).

    They are one corner seen from either side -- the salt's outer edge, and the
    metal it drains into -- so they stand on one left edge, MARK_LEFT, and meet
    on another, r = 39, where the salt outline's right side is the solid
    outline's leg.  Only the solid's outer edge overshoots, by MARK_OVERSHOOT.
    Each label sits out in the cavity, clear of both.
    """
    ax.add_patch(Polygon(solid_outline(VIEW_SOLID), closed=True,
                         facecolor="none", edgecolor=C_SOLID_BOX, lw=1.5,
                         joinstyle="miter", zorder=5))
    r0, r1, y0, y1 = VIEW_SALT
    ax.add_patch(Rectangle((r0, y0), r1 - r0, y1 - y0, facecolor="none",
                           edgecolor=C_SALT_BOX, lw=1.5, zorder=5))
    if not labels:
        return
    # Anchored on MARK_LEFT and running right, so each label sits over its own
    # outline rather than off in the cavity with a gap between the two.
    for label, color, y, va in (
        ("(c,d) salt", C_SALT_BOX, Y_FT + 1.2, "bottom"),
        ("(e,f) solid", C_SOLID_BOX, VIEW_SOLID[2] - 0.4, "top"),
    ):
        ax.text(MARK_LEFT, y, label, ha="left", va=va, fontsize=PT_BOX,
                color=color, fontweight="bold", zorder=6)


def caption(ax, text):
    # Offset in points, not axes fraction: the three rows have very different
    # panel heights and the caption must clear the scale bar in all of them.
    ax.annotate(text, xy=(0.5, 0.0), xycoords="axes fraction",
                xytext=(0, -21), textcoords="offset points",
                ha="center", va="top", fontsize=PT_CAP, color=INK)


# ── Figure ───────────────────────────────────────────────────────────────────


def main():
    salt = {case: load_field(case, 1) for case, _ in CASES}
    solid = {case: load_field(case, 2) for case, _ in CASES}

    rows = [
        # view, log?, the fields one case contributes, mesh?, row name
        (VIEW_FULL, True, lambda c: [salt[c], solid[c]], False, "Full domain"),
        (VIEW_SALT, True, lambda c: [salt[c]], True, "Salt region"),
        (VIEW_SOLID, True, lambda c: [solid[c]], True, "Solid region"),
    ]

    # Colour limits first.  Shared: one set of round decades for every panel,
    # and the report says what each row loses off either end.  Per row: limits
    # derived from the data inside that row's own viewport.
    if SHARED_SCALE:
        every = [f for case, _ in CASES for f in (salt[case], solid[case])]
        shared, _ = row_scale(every, VIEW_FULL, True)
        scales = [(shared, 0.0)] * len(rows)
        print(f"[scale] shared      log    {shared.vmin:.4e} to {shared.vmax:.4e}")
        for view, _log, fields, _mesh, name in rows:
            v = np.concatenate([inside(t, val, view) for case, _ in CASES
                                for t, val in fields(case)])
            print(f"           {name:<12s} in view [{v.min():.3e}, {v.max():.3e}]"
                  f"   below floor {(v < shared.vmin).mean():6.2%}")
    else:
        scales = []
        for view, log, fields, _mesh, name in rows:
            both = [f for case, _ in CASES for f in fields(case)]
            norm, cut = row_scale(both, view, log)
            scales.append((norm, cut))
            print(f"[scale] {name:<12s} {'log' if log else 'linear':<6s} "
                  f"{norm.vmin:.4e} to {norm.vmax:.4e}"
                  + (f"   below floor: {cut:.2%} of nodes" if cut else ""))

    aspects = [(v[3] - v[2]) / (v[1] - v[0]) for v, *_ in rows]
    panel_h = [PANEL_W * a for a in aspects]

    # Room under each panel for its caption, and an equal margin either side.
    cap_h = 0.46
    fig_w = 2 * PANEL_W + PANEL_GAP + MARGIN_L + MARGIN_R
    fig_h = sum(panel_h) + len(rows) * cap_h + 0.25

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        len(rows), 2,
        height_ratios=[h + cap_h for h in panel_h],
        left=MARGIN_L / fig_w, right=1.0 - MARGIN_R / fig_w,
        top=0.995, bottom=0.005,
        wspace=PANEL_GAP / PANEL_W, hspace=0.0,
    )

    right_panels = []
    letters = iter("abcdef")
    for i, (view, _log, fields, mesh, name) in enumerate(rows):
        norm, cut = scales[i]
        for j, (case, case_label) in enumerate(CASES):
            # Each cell holds the panel on top and its caption underneath, so
            # the panels of a row share a baseline whatever their aspect.
            cell = gs[i, j].subgridspec(
                2, 1, height_ratios=[panel_h[i], cap_h], hspace=0.0
            )
            ax = fig.add_subplot(cell[0])
            draw_field(ax, fields(case), norm, view, mesh=mesh)
            add_scale_bar(ax, view)
            caption(ax, f"({next(letters)}) {name}, {case_label}")
            if i == 0:
                mark_zoom(ax, labels=(j == 0))
            if j == len(CASES) - 1:
                right_panels.append(ax)
                if not SHARED_SCALE:
                    add_row_colorbar(ax, norm)

    if SHARED_SCALE:
        add_figure_colorbar(fig, right_panels, scales[0][0])

    for ext in ("pdf", "svg"):
        fig.savefig(f"{OUT}.{ext}", dpi=600)
    fig.savefig(f"{OUT}.png", dpi=110)
    fig.savefig(f"{OUT}_2x.png", dpi=220)
    plt.close(fig)
    print(f"[saved] {OUT}.pdf, .svg, .png, _2x.png")


if __name__ == "__main__":
    main()
