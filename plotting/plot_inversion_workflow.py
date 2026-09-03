"""
Flowchart of the pointwise FLiBe permeability inversion in para_swap_pure.py.

Prescribed inputs and the Ni dry-run constraint converge on the FESTIM forward
model; the trial solubility enters from the left; the simulated flux is compared
with the measured one; the loop returns until the comparison passes.

The figure is drawn in K_H where the code iterates on phi. At the prescribed D
these are the same iteration, since log10(K_H) and log10(phi) differ by a
constant, so the stopping test may be written on either.

STOP_RULE selects the test shown in the decision box:
    "flux"     residual form, |J_sim - J_exp| < eps
    "bracket"  what the code checks, |delta log10 phi| < 3e-3

Sized for a double-column figure (190 mm). _check_fit reports any label wider
than the shape holding it.

Outputs (saved to results/):
    inversion_workflow.pdf/.svg/.png  -- the flowchart
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

OUTDIR = Path(__file__).resolve().parents[1] / "results"

STOP_RULE = "flux"             # "flux" = residual test; "bracket" = what the code checks

_HAVE_ARIAL = "Arial" in {f.name for f in fm.fontManager.ttflist}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    # Match the math font to the text font; every box mixes the two in one line.
    "mathtext.fontset": "custom" if _HAVE_ARIAL else "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
if _HAVE_ARIAL:
    plt.rcParams.update({
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
    })

PT_H, PT_B, PT_A = 9.0, 8.0, 7.5      # heading / body / branch label

# One hue per meaning: the unknown being solved for, the forward model, the
# measurement matched against, and the Ni constraint carried in from the dry
# runs. Checked against each other under simulated protanopia, deuteranopia and
# tritanopia; worst pairing dE 10.5.
BLUE, GREEN, ORANGE, PLUM = "#3F6FAF", "#2A6E4F", "#D97A43", "#A6377E"
INK, INK2 = "#222222", "#666666"
OUTLINE = "#8A8A84"

# Low-saturation fills, so a heading can keep the hue of its own box.
FILL = {BLUE: "#E4ECF7", GREEN: "#DFEBE4", ORANGE: "#F8E9DE",
        PLUM: "#F3E5EE", INK: "#FFFFFF"}

# Orange text takes a darker step: on its own tint it is otherwise 2.8:1.
TEXT = {BLUE: BLUE, GREEN: GREEN, ORANGE: "#A85520", PLUM: PLUM, INK: INK}

LINE = 0.034                   # line pitch inside a box
_FIT: list = []                # (artist, x0, x1) checked once the canvas exists


def box(ax, x0, x1, y0, y1, lines, color=INK, ls="-", pad=0.010):
    """A rounded box holding one bold heading line and any number of body lines."""
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={pad}",
        facecolor=FILL[color], edgecolor=color if color is not INK else OUTLINE,
        linewidth=1.1, linestyle=ls, zorder=3))
    cx = 0.5 * (x0 + x1)
    top = 0.5 * (y0 + y1) + (len(lines) - 1) * LINE / 2
    for i, t in enumerate(lines):
        a = ax.text(cx, top - i * LINE, t, ha="center", va="center",
                    fontsize=PT_H if i == 0 else PT_B,
                    color=TEXT[color] if i == 0 else INK2,
                    weight="bold" if i == 0 else "normal", zorder=4)
        _FIT.append((a, x0, x1))


def diamond(ax, cx, cy, hw, hh, lines):
    ax.add_patch(Polygon([(cx, cy + hh), (cx + hw, cy), (cx, cy - hh),
                          (cx - hw, cy)], closed=True, facecolor="#FFFFFF",
                         edgecolor=OUTLINE, linewidth=1.1, zorder=3))
    top = cy + (len(lines) - 1) * LINE / 2
    for i, t in enumerate(lines):
        y = top - i * LINE
        a = ax.text(cx, y, t, ha="center", va="center",
                    fontsize=PT_H if i == 0 else PT_B,
                    color=INK if i == 0 else INK2,
                    weight="bold" if i == 0 else "normal", zorder=4)
        # A rhombus is only full width on its centre line.
        half = hw * (1.0 - abs(y - cy) / hh)
        _FIT.append((a, cx - half, cx + half))


def arrow(ax, p0, p1, color=INK, lw=1.4, ms=11, z=5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, linewidth=lw, shrinkA=0,
                                 shrinkB=0, zorder=z,
                                 connectionstyle="arc3,rad=0"))


def _check_fit(fig, ax):
    """Report any label wider than the shape holding it, in axes units."""
    fig.canvas.draw()
    inv = ax.transAxes.inverted()
    bad = []
    for artist, x0, x1 in _FIT:
        bb = artist.get_window_extent(fig.canvas.get_renderer())
        (ax0, _), (ax1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
        over = max(x0 - ax0, ax1 - x1)
        if over > 0.002:
            bad.append((over, artist.get_text()))
    for over, t in sorted(bad, reverse=True):
        print(f"  [overflow {over:+.3f}] {t}")
    return bad


# ── Layout ────────────────────────────────────────────────────────────────────
# Three columns: the model and what follows it down the middle, the trial value
# and its update on the left so the return leg is one straight segment, the
# measurement on the right at the row that consumes it. The rhombus uses three
# vertices -- loop out left, exit down, measurement in right.
CX = 0.500

LCOL = (0.005, 0.265)          # trial value, and its update
SPINE = (0.315, 0.685)         # model, simulated flux
TARGET = (0.735, 0.995)        # measured flux, right of the test
OUTB = (0.285, 0.715)          # exit, below the test
IN = [(0.005, 0.325), (0.340, 0.660), (0.675, 0.995)]

Y_IN = (0.850, 0.995)
Y_MOD = (0.620, 0.755)         # the model row: trial box and model box
Y_JS = (0.455, 0.570)
Y_DIA, DIA_HW, DIA_HH = 0.300, 0.150, 0.095
Y_UPD = Y_TGT = (0.235, 0.365)
Y_OUT = (0.005, 0.140)


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.48, 4.45))
    ax.set_xlim(0, 1)
    # Clipped to the drawing; a tight bbox still keeps the whole invisible axes.
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # ── the three families of input ──────────────────────────────────────────
    feeds = [
        (["Experimental conditions",
          r"$T$,  $P_{\mathrm{up}}$,  $P_{\mathrm{down}}$,",
          "gas species,  geometry"], INK, INK2),
        (["Prescribed model inputs",
          r"$D_{\mathrm{FLiBe}}(T)$,",
          "external boundary condition"], INK, INK2),
        # Phi_Ni(T) is measured, not chosen, so it carries its own hue.
        (["Ni-only dry-run constraint",
          "Hydrogen-isotope permeability",
          r"in Ni,  $\Phi_{\mathrm{Ni}}(T)$"], PLUM, PLUM),
    ]
    for span, (lines, color, _) in zip(IN, feeds):
        box(ax, *span, *Y_IN, lines, color=color)

    # Three landing points, so the arrowheads do not collide. Each arrow takes
    # the colour of the box it leaves.
    for (x0, x1), dx, (_, _, acol) in zip(IN, (-0.075, 0.0, +0.075), feeds):
        arrow(ax, (0.5 * (x0 + x1), Y_IN[0]), (CX + dx, Y_MOD[1]), color=acol,
              lw=1.2)

    # ── the trial value, and the model it enters ─────────────────────────────
    box(ax, *LCOL, *Y_MOD,
        ["Trial FLiBe solubility", r"$K_{H,\mathrm{FLiBe}}^{(n)}$"], color=BLUE)
    # One line, but the row keeps the height the trial box beside it needs.
    box(ax, *SPINE, *Y_MOD, ["FESTIM 2-D model"], color=GREEN)
    arrow(ax, (LCOL[1], 0.5 * sum(Y_MOD)), (SPINE[0], 0.5 * sum(Y_MOD)),
          color=BLUE)

    box(ax, *SPINE, *Y_JS,
        ["Simulated flux", r"$J_{\mathrm{sim}}$"], color=GREEN)
    arrow(ax, (CX, Y_MOD[0]), (CX, Y_JS[1]), color=GREEN)

    # ── the test ─────────────────────────────────────────────────────────────
    arrow(ax, (CX, Y_JS[0]), (CX, Y_DIA + DIA_HH), color=GREEN)
    diamond(ax, CX, Y_DIA, DIA_HW, DIA_HH,
            ["Bracket converged?",
             r"$\Delta\log_{10}K_{H,\mathrm{FLiBe}} < 3\times10^{-3}$"]
            if STOP_RULE == "bracket" else
            # The question mark sits inside the mathtext, or the line is set in
            # two weights.
            [r"$|J_{\mathrm{sim}} - J_{\mathrm{exp}}| < \epsilon \;?$"])

    box(ax, *TARGET, *Y_TGT,
        ["Experimental target",
         r"$J_{\mathrm{exp}} \pm u(J_{\mathrm{exp}})$"],
        color=ORANGE)
    arrow(ax, (TARGET[0], Y_DIA), (CX + DIA_HW, Y_DIA), color=ORANGE, lw=1.2)

    # ── the two branches ─────────────────────────────────────────────────────
    box(ax, *LCOL, *Y_UPD,
        ["Update trial solubility",
         r"$K_{H,\mathrm{FLiBe}}^{(n+1)}$ by bisection"], color=BLUE)
    arrow(ax, (CX - DIA_HW, Y_DIA), (LCOL[1], Y_DIA), color=INK)
    ax.text(0.5 * (LCOL[1] + CX - DIA_HW), Y_DIA + 0.014, "No", ha="center",
            va="bottom", fontsize=PT_A, color=INK)

    box(ax, *OUTB, *Y_OUT,
        ["Inferred FLiBe permeability",
         r"$\Phi_{\mathrm{FLiBe}} = D_{\mathrm{FLiBe}}\,K_{H,\mathrm{FLiBe}}$"], color=BLUE)
    arrow(ax, (CX, Y_DIA - DIA_HH), (CX, Y_OUT[1]), color=INK)
    ax.text(CX + 0.012, 0.5 * (Y_DIA - DIA_HH + Y_OUT[1]), "Yes", ha="left",
            va="center", fontsize=PT_A, color=INK)

    # ── the return ───────────────────────────────────────────────────────────
    arrow(ax, (0.5 * sum(LCOL), Y_UPD[1]), (0.5 * sum(LCOL), Y_MOD[0]),
          color=BLUE)

    fig.tight_layout(pad=0.15)
    bad = _check_fit(fig, ax)

    stem = OUTDIR / "inversion_workflow"
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    # 96 dpi proof at the final printed size.
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=96)
    fig.savefig(f"{stem}_2x.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {stem}.pdf, .svg, .png (96 dpi proof), _2x.png"
          f"{'' if not bad else f'  — {len(bad)} label(s) overflow'}")


if __name__ == "__main__":
    main()
