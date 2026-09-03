"""
Introduction figure: why one measured flux does not determine one permeability.

Two panels, named in the caption rather than in the frame:

  (a)  a simplified section of the cell carrying the three arrow families that
       matter -- the primary path through the salt and the membrane, the
       side-wall path through the Ni structure, and exchange with the external
       boundary, drawn two-way because hydrogen in the surrounding atmosphere
       can re-enter the vessel as well as leave it
  (b)  the two readings of the same measured number, set against each other

Flow direction follows the SWAP configuration as implemented in
para_swap_pure.py: top_cap, top_sidewall and liquid_surface carry P_up;
mid_membrane, bottom_cap and bottom_sidewall carry P_down. The side-wall path
therefore bypasses both the salt and the membrane.

Geometry is schematic. The cell is 111 mm tall against a 41 mm radius, so the
gas volumes are compressed to keep the salt and the membrane legible. No
results appear here; magnitudes belong in the discussion.

Outputs (saved to results/):
    framework_diagram.pdf/.svg/.png  -- the two-panel figure
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

OUTDIR = Path(__file__).resolve().parents[1] / "results"

# The fallbacks are metric-compatible or wider than Arial, so a layout that
# fits without it fits with it.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

# Three levels only: the question, the two readings, everything else.
PT_P, PT_H, PT_B = 10.0, 9.0, 8.0      # panel label / heading / body

# One hue per meaning: green and orange for the two pathways in (a), blue for
# the present framework in (b). The salt keeps a light tint because it is a
# material, not a pathway. Green against orange is the red-green confusion
# pair; #2A6E4F clears it (protan dE 12.2). Both pathways are also directly
# labelled, so colour reinforces rather than carries the identity.
GREEN, ORANGE, BLUE = "#2A6E4F", "#D97A43", "#3F6FAF"
INK, INK2 = "#222222", "#666666"         # major text and arrows / explanatory
OUTLINE = "#8A8A84"
NI_FILL, SALT_FILL, GAS_FILL = "#E3E2DE", "#CBDEF2", "#FFFFFF"
SALT_INK = "#35699E"      # the salt's own label: the same blue, dark enough to read
LW_OUT, LW = 1.6, 1.0     # outer perimeter / inner outline and band separators

W_OUT, W_IN = 1.00, 0.84          # half-widths; the wall is exaggerated
LAYERS = [(0, 3, NI_FILL), (3, 19, GAS_FILL), (19, 24, NI_FILL),
          (24, 33, SALT_FILL), (33, 51, GAS_FILL), (51, 54, NI_FILL)]
H_TOT = 54.0
CX, CY0, CY1, CHW = 0.155, 0.320, 0.940, 0.130


def _px(u):
    return CX + u * CHW


def _py(v):
    return CY0 + (v / H_TOT) * (CY1 - CY0)


def arrow(ax, p0, p1, color=INK, lw=1.5, ls="-", cs=None, z=6, ms=12):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=0, shrinkB=0, zorder=z,
                                 connectionstyle=cs or "arc3,rad=0"))


def draw_cell(ax):
    # Bands are separated by rules, not by their own rectangles, which would
    # double the strokes on every shared segment.
    ax.add_patch(Rectangle((_px(-W_OUT), _py(0)), 2 * W_OUT * CHW,
                           _py(H_TOT) - _py(0), facecolor=NI_FILL,
                           edgecolor="none", zorder=0))
    for v0, v1, fc in LAYERS:
        ax.add_patch(Rectangle((_px(-W_IN), _py(v0)), 2 * W_IN * CHW,
                               _py(v1) - _py(v0), facecolor=fc,
                               edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((_px(-W_IN), _py(3)), 2 * W_IN * CHW,
                           _py(51) - _py(3), facecolor="none",
                           edgecolor=OUTLINE, lw=LW, zorder=3))
    for v in (19, 24, 33):
        ax.plot([_px(-W_IN), _px(W_IN)], [_py(v)] * 2, color=OUTLINE, lw=LW,
                zorder=3)
    # Stroked last, or the cap fills cover its inner half.
    ax.add_patch(Rectangle((_px(-W_OUT), _py(0)), 2 * W_OUT * CHW,
                           _py(H_TOT) - _py(0), facecolor="none",
                           edgecolor=OUTLINE, lw=LW_OUT, zorder=4))

    for v, t, c in [(45, "upstream gas", INK2), (28.5, "FLiBe", SALT_INK),
                    (21.5, "Ni membrane", INK2), (8, "downstream gas", INK2)]:
        ax.text(_px(0), _py(v), t, ha="center", va="center", fontsize=PT_B,
                color=c, zorder=5)
    ax.text(_px(-W_OUT) - 0.008, _py(50), "Ni\nstructure", ha="right",
            va="center", fontsize=PT_B, color=INK2, zorder=5)

    arrow(ax, (_px(-0.45), _py(38)), (_px(-0.45), _py(13)), color=GREEN, lw=2.2)
    ax.text(_px(-0.60), _py(28), "Primary pathway", ha="center", va="center",
            fontsize=PT_B, color=GREEN, rotation=90, zorder=6)
    arrow(ax, (_px(0.48), _py(41)), (_px(0.92), _py(41)), color=ORANGE, lw=1.8)
    arrow(ax, (_px(0.92), _py(40)), (_px(0.92), _py(12)), color=ORANGE, lw=1.8)
    arrow(ax, (_px(0.92), _py(11)), (_px(0.48), _py(11)), color=ORANGE, lw=1.8)
    ax.text(_px(0.60), _py(28), "Structural pathway", ha="center",
            va="center", fontsize=PT_B, color=ORANGE, rotation=90, zorder=6)
    # Two-way: hydrogen can re-enter from the surrounding atmosphere.
    arrow(ax, (_px(W_OUT), _py(31)), (_px(W_OUT) + 0.024, _py(31)),
          color=INK, lw=1.3, ls=(0, (3, 2)), ms=10)
    arrow(ax, (_px(W_OUT) + 0.024, _py(17)), (_px(W_OUT), _py(17)),
          color=INK, lw=1.3, ls=(0, (3, 2)), ms=10)
    ax.text(_px(W_OUT) + 0.006, _py(24), "External\nexchange", ha="left",
            va="center", fontsize=PT_B, color=INK, zorder=6)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.48, 3.30))
    ax.set_xlim(0, 1)
    # Clipped to the drawing; a tight bbox still keeps the whole invisible axes.
    ax.set_ylim(0.20, 1.0)
    ax.axis("off")

    draw_cell(ax)
    ax.text(CX, 0.975, "(a)", ha="center", va="center", fontsize=PT_P,
            color=INK)
    ax.text(0.015, 0.245, r"$J_{\mathrm{measured}}$ reflects multiple transport"
            " pathways", ha="left", va="center", fontsize=PT_B, color=INK2)

    # ── (b) the two readings, converging on one statement ────────────────────
    L, R, MID = 0.530, 0.845, 0.6875
    ax.text(MID, 0.975, "(b)", ha="center", va="center", fontsize=PT_P,
            color=INK)

    for i, t in enumerate(["Conventional 1-D", "interpretation"]):
        ax.text(L, 0.870 - i * 0.058, t, ha="center", va="center",
                fontsize=PT_H, color=INK2, weight="bold")
    for i, t in enumerate(["represents the intended axial",
                           "pathway using a 1-D formulation"]):
        ax.text(L, 0.700 - i * 0.055, t, ha="center", va="center",
                fontsize=PT_B, color=INK2)

    # Only the heading is blue, so the colour marks the framework, not a line.
    for i, t in enumerate(["Present multidimensional", "framework"]):
        ax.text(R, 0.870 - i * 0.058, t, ha="center", va="center",
                fontsize=PT_H, color=BLUE, weight="bold")
    for i, t in enumerate(["explicitly resolves transport through",
                           "FLiBe, the Ni structure, and",
                           "external boundaries"]):
        ax.text(R, 0.700 - i * 0.055, t, ha="center", va="center",
                fontsize=PT_B, color=INK2)

    for x in (L, R):
        ax.plot([x, x], [0.530, 0.500], color=INK2, lw=1.1, zorder=1)
    ax.plot([L, R], [0.500, 0.500], color=INK2, lw=1.1, zorder=1)
    arrow(ax, (MID, 0.500), (MID, 0.390), color=INK, lw=1.4)
    ax.text(MID, 0.325, "Influence on inferred FLiBe permeability",
            ha="center", va="center", fontsize=PT_H, color=INK, weight="bold")

    fig.tight_layout(pad=0.2)
    stem = OUTDIR / "framework_diagram"
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    # a proof at the final printed size: 7.48 in at 96 dpi is what 100 % zoom
    # looks like on screen, which is the size the labels have to survive
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=96)
    fig.savefig(f"{stem}_2x.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {stem}.pdf, .svg, .png (96 dpi proof), _2x.png")


if __name__ == "__main__":
    main()
