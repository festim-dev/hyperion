"""
Permeation cell and a magnified view of the salt-membrane interface.

Left, an axial section of the Ni-200 vessel inside its argon glovebox. Right, a
circular detail of the region the hydrogen crosses. The section is schematic:
the gas volumes are compressed so that the salt and the membrane stay legible,
while the ratio between those two layers is kept true, so the two thicknesses
marked on the drawing are not contradicted by what is drawn.

L_FLiBe is the one temperature-dependent length. The salt rests on the membrane
at y_mT = 24 mm at every temperature and expands upward, so its thickness is
set per run from Y_FT_BY_TEMP_C in para_swap_pure.py.

In the detail, paired marks denote the molecule and single marks the dissolved
atom: FLiBe follows Henry's law and dissolves hydrogen without dissociating, so
the pair splits on entering the metal rather than at the salt surface.

Colour carries one meaning each: blue FLiBe, grey Ni, green the hydrogen
isotope, brass the coated outer surface, purple the enclosure, grey dashes the
graphical zoom.

Outputs (saved to results/):
    cell_schematic.pdf/.svg/.png  -- the cell and its interface detail
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib import patheffects as pe
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

OUTDIR = Path(__file__).resolve().parents[1] / "results"

# Set False to drop the three interface notes and let the caption carry them.
INTERFACE_LABELS = True

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

PT_L, PT_B, PT_D = 8.5, 8.0, 7.0        # region label / body / note

GREEN = "#2A6E4F"                       # the hydrogen isotope
# The coated outer surface. Checked against the rest of this palette under
# simulated protanopia, deuteranopia and tritanopia; worst pairing dE 10.1
# against the glovebox. Low chroma (C* 32) keeps it from dominating the figure.
COAT = "#9A7C4A"
GLOVE = "#9B7FB5"                       # the enclosure
INK, INK2 = "#222222", "#666666"
NI_FILL, NI_EDGE = "#E3E2DE", "#6F6F69"
SALT_FILL, SALT_INK = "#CBDEF2", "#35699E"
GAS_FILL = "#FFFFFF"
ZOOM = "#8A8A84"                        # the graphical zoom, neutral

L_FLIBE_TEXT = "5.140–5.364 mm"         # 500 to 700 C; see the geometry table
L_NI_TEXT = "2 mm"

# Trimmed to where the interface labels end. AR is computed from the spans, so
# the detail circle stays round through the crop.
XMIN, XMAX = 0.000, 0.9785
YMIN, YMAX = 0.030, 0.970
# The height is set by the detail circle; at this value the section's aspect
# also lands within a per cent of the cell's real 82 : 111.
FIG = (7.48, 3.83 * (YMAX - YMIN))      # 190 x 91 mm
# Inches per axes-unit differ on the two axes; round shapes correct by AR.
AR = (FIG[1] / (YMAX - YMIN)) / (FIG[0] / (XMAX - XMIN))

# ── the section, in schematic units: 13 : 5 keeps the salt-to-membrane ratio ──
LAYERS = [(0, 3, NI_FILL), (3, 18, GAS_FILL), (18, 23, NI_FILL),
          (23, 36, SALT_FILL), (36, 72, GAS_FILL), (72, 75, NI_FILL)]
V_TOT = 75.0
W_OUT, W_IN = 1.00, 0.90                # half-widths; the wall is exaggerated

CELL_CX, CELL_HW = 0.220, 0.145         # x 0.075 .. 0.365
CELL_Y0, CELL_Y1 = 0.095, 0.875
GLOVEBOX = (0.045, 0.395, 0.045, 0.955)

# The source circle sits on the vessel axis, since the detail is a
# one-dimensional stack rather than a view taken at some radius.
ZOOM_C = (0.220, 0.3758)                # the small circle on the section
ZOOM_R = 0.062
DET_C = (0.615, 0.500)                  # the detail
DET_R = 0.165
LAB_X = 0.792


def cx(u):
    return CELL_CX + u * CELL_HW


def cy(v):
    return CELL_Y0 + (v / V_TOT) * (CELL_Y1 - CELL_Y0)


def disc(ax, c, r, **kw):
    """A visually round patch, radius r in x-units."""
    e = Ellipse(c, 2 * r, 2 * r / AR, **kw)
    ax.add_patch(e)
    return e


def on_ellipse(c, r, deg):
    from math import cos, radians, sin
    return (c[0] + r * cos(radians(deg)), c[1] + (r / AR) * sin(radians(deg)))


def arrow(ax, p0, p1, color=GREEN, lw=1.1, ms=7, z=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, shrinkA=0,
                                 shrinkB=0, zorder=z))


def h2(ax, x, y, r=0.0058, z=9):
    for dx in (-r * 0.92, r * 0.92):
        disc(ax, (x + dx, y), r, facecolor=GREEN, edgecolor="none", zorder=z)


def h1(ax, x, y, r=0.0050, z=9):
    disc(ax, (x, y), r, facecolor=GREEN, edgecolor="none", zorder=z)


def exchange(ax, x, y, h=0.038, lw=1.2, ms=8):
    """A two-way surface exchange: one arrow, a head at each end.

    Drawn straddling the line, so the two heads point into the two phases and
    the shaft crosses the surface the exchange happens at.  Two single-headed
    arrows side by side said the same thing with twice the ink and had to pick
    a side for each direction.
    """
    ax.add_patch(FancyArrowPatch((x, y - 0.5 * h), (x, y + 0.5 * h),
                                 arrowstyle="<|-|>", mutation_scale=ms,
                                 color=GREEN, linewidth=lw, shrinkA=0,
                                 shrinkB=0, zorder=9))


def draw_section(ax):
    """The vessel: one light Ni fill for wall, caps and membrane alike."""
    ax.add_patch(Rectangle((cx(-W_OUT), cy(0)), 2 * W_OUT * CELL_HW,
                           cy(V_TOT) - cy(0), facecolor=NI_FILL,
                           edgecolor="none", zorder=1))
    for v0, v1, fc in LAYERS:
        ax.add_patch(Rectangle((cx(-W_IN), cy(v0)), 2 * W_IN * CELL_HW,
                               cy(v1) - cy(v0), facecolor=fc, edgecolor="none",
                               zorder=2))
    # Stroked once each; per-band rectangles would double every shared edge.
    ax.add_patch(Rectangle((cx(-W_IN), cy(3)), 2 * W_IN * CELL_HW,
                           cy(72) - cy(3), facecolor="none", edgecolor=NI_EDGE,
                           lw=0.9, zorder=4))
    for v in (18, 23, 36):
        ax.plot([cx(-W_IN), cx(W_IN)], [cy(v)] * 2, color=NI_EDGE, lw=0.9,
                zorder=4)
    # Stroked last, or the cap fills cover its inner half.
    ax.add_patch(Rectangle((cx(-W_OUT), cy(0)), 2 * W_OUT * CELL_HW,
                           cy(V_TOT) - cy(0), facecolor="none",
                           edgecolor=COAT, lw=1.4, zorder=5))

    ax.text(cx(0.85), cy(66), "Ni-200 vessel", ha="right", va="center",
            fontsize=PT_B, color=INK2, zorder=7)
    ax.text(cx(0), cy(56), r"upstream H$_2$ / D$_2$", ha="center", va="center",
            fontsize=PT_L, color=INK, zorder=7)
    for u, v in ((-0.42, 48), (0.30, 50), (0.10, 43)):
        h2(ax, cx(u), cy(v), z=7)

    # Each label carries a gap in its band fill for the source circle to pass.
    halo = lambda fc: [pe.withStroke(linewidth=2.8, foreground=fc)]
    ax.text(cx(-0.86), cy(32.5), "molten FLiBe", ha="left", va="center",
            fontsize=PT_L, color=SALT_INK, zorder=9,
            path_effects=halo(SALT_FILL))
    ax.text(cx(-0.86), cy(27.0), rf"$L_{{\rm FLiBe}}(T)$ = {L_FLIBE_TEXT}",
            ha="left", va="center", fontsize=PT_D, color=SALT_INK, zorder=9,
            path_effects=halo(SALT_FILL))
    ax.text(cx(-0.86), cy(20.5), rf"Ni membrane,  $L_{{\rm Ni}}$ = {L_NI_TEXT}",
            ha="left", va="center", fontsize=PT_D, color=INK2, zorder=9,
            path_effects=halo(NI_FILL))

    ax.text(cx(0), cy(9), "downstream Ar sweep", ha="center", va="center",
            fontsize=PT_L, color=INK, zorder=7)
    for u, v in ((-0.34, 14), (0.36, 13)):
        h2(ax, cx(u), cy(v), z=7)

    ax.text(cx(0), cy(0) - 0.034, r"Al$_2$O$_3$-coated external surface",
            ha="center", va="center", fontsize=PT_D, color=COAT, zorder=7)


def draw_enclosure(ax):
    x0, x1, y0, y1 = GLOVEBOX
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.012",
                                facecolor="none", edgecolor=GLOVE, lw=1.1,
                                zorder=3))
    ax.text(x0 + 0.014, y1 - 0.030, "Ar-filled glovebox", ha="left",
            va="center", fontsize=PT_B, color=GLOVE, zorder=7)


def draw_detail(ax):
    """The circular zoom: gas, FLiBe, Ni, gas, and what crosses between them."""
    circle = disc(ax, DET_C, DET_R, facecolor=GAS_FILL, edgecolor=ZOOM,
                  linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    Y_GS, Y_FN, Y_ND = 0.622, 0.462, 0.400      # the three surfaces
    x0, x1 = DET_C[0] - DET_R, DET_C[0] + DET_R
    for ya, yb, fc in ((Y_FN, Y_GS, SALT_FILL), (Y_ND, Y_FN, NI_FILL)):
        band = Rectangle((x0, ya), x1 - x0, yb - ya, facecolor=fc,
                         edgecolor="none", zorder=4)
        ax.add_patch(band)
        band.set_clip_path(circle)
    for yv in (Y_GS, Y_FN, Y_ND):
        ln, = ax.plot([x0, x1], [yv] * 2, color=NI_EDGE, lw=0.9, zorder=5)
        ln.set_clip_path(circle)

    # Paired marks for the molecule, single for the atom: the pair survives the
    # salt and splits only on entering the metal.
    for x, y in ((0.556, 0.700), (0.660, 0.722), (0.700, 0.665)):
        h2(ax, x, y)
    for x, y in ((0.545, 0.560), (0.660, 0.545), (0.700, 0.585)):
        h2(ax, x, y)
    for x, y in ((0.560, 0.434), (0.672, 0.426), (0.716, 0.442)):
        h1(ax, x, y)
    for x, y in ((0.578, 0.330), (0.668, 0.300)):
        h2(ax, x, y)

    # Every one of the three is an equilibrium, so every one is reversible.
    # The lower two are kept short: the membrane band is 0.05 of the frame, so
    # arrows straddling both its edges would otherwise run into each other.
    exchange(ax, DET_C[0], Y_GS, h=0.056)
    exchange(ax, DET_C[0], Y_FN, h=0.048)
    exchange(ax, DET_C[0], Y_ND, h=0.048)

    if INTERFACE_LABELS:
        # Each label sits at the height of the surface it names, just clear of
        # the circle, so its own position identifies it and no leader is
        # needed.  The column is aligned rather than tracking the circle's
        # edge: three leaders, or three ragged left margins, were more marking
        # than three short phrases are worth.
        for yv, t in ((Y_GS, "gas–FLiBe equilibrium"),
                      (Y_FN, "FLiBe–Ni partitioning"),
                      (Y_ND, "Ni–gas equilibrium")):
            ax.text(LAB_X, yv, t, ha="left", va="center", fontsize=PT_D,
                    color=INK2, zorder=7)


def draw_zoom_lines(ax):
    """The graphical zoom: neutral grey, so it is not read as a boundary."""
    disc(ax, ZOOM_C, ZOOM_R, facecolor="none", edgecolor=ZOOM, linewidth=0.9,
         linestyle=(0, (4, 3)), zorder=8)
    for a_s, a_d in ((35, 145), (-35, 215)):
        p0 = on_ellipse(ZOOM_C, ZOOM_R, a_s)
        p1 = on_ellipse(DET_C, DET_R, a_d)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ZOOM, lw=0.8,
                ls=(0, (1.5, 2.5)), zorder=6)


CAPTION = (
    "Permeation cell and the interface it is built around. Left: axial section "
    "of the Ni-200 vessel inside the argon glovebox, drawn schematically -- the "
    "gas volumes are compressed so that the salt and the membrane remain "
    "legible, while the ratio between those two layers is kept true. "
    f"$L_\\mathrm{{FLiBe}}$ varies with temperature over {L_FLIBE_TEXT} as the "
    "salt expands from its fixed base on the membrane. Right: magnified view "
    "of the salt--membrane region. Paired marks denote the molecule, the form "
    "dissolved in molten FLiBe; single marks the atom, dissolved in Ni. The "
    "molecule therefore dissociates on entering the metal rather than at the "
    "salt surface."
)


def main() -> None:
    fig, ax = plt.subplots(figsize=FIG)
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.axis("off")

    draw_section(ax)
    draw_enclosure(ax)
    draw_zoom_lines(ax)
    draw_detail(ax)

    fig.tight_layout(pad=0.15)
    stem = OUTDIR / "cell_schematic"
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=96)
    fig.savefig(f"{stem}_2x.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {stem}.pdf, .svg, .png (96 dpi proof), _2x.png")
    print(f"[caption] {CAPTION}")


if __name__ == "__main__":
    main()
