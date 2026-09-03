"""
Axisymmetric computational domain of the permeation cell, with its coordinates.

The section is drawn at true scale, with the salt-membrane corner magnified
isotropically alongside. Only Omega_FLiBe and Omega_Ni are meshed; the gas
volumes enter the model through boundary conditions on the surfaces that bound
them.

Names and dimensions follow the physical groups of mesh.py:

    Omega_FLiBe   fluid, tag 1        r < 39, 24 < z < 29.14
    Omega_Ni      solid, tag 2        wall, both caps and the membrane
    Gamma_Ni-up   tags 5, 6           top cap and upper side wall, at P_up
    Gamma_FLiBe   tag 8               salt free surface, at P_up
    Gamma_Interface  tag 99           salt-metal coupling, two curves: the
                                      salt's underside and its outer edge
    Gamma_Ni-down tags 7, 9, 10       lower wall, membrane underside, bottom cap
    Gamma_ext     tag 3               outer surface
    symmetry      tags 41-44          r = 0

The axial coordinate is written z, following the cylindrical (r, z) convention;
mesh.py builds the section as a plane figure and calls the same coordinate y.

Outputs (saved to results/):
    domain_map.pdf/.svg/.png  -- the domain and its magnified corner
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Rectangle

OUTDIR = Path(__file__).resolve().parents[1] / "results"

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

PT_L, PT_B, PT_D = 8.5, 8.0, 7.0

# Okabe-Ito, one per boundary condition.
C_UP   = "#0072B2"   # Gamma_Ni-up      Sieverts at P_up
C_SALT = "#009E73"   # Gamma_FLiBe      Henry at P_up
C_INT  = "#D55E00"   # Gamma_Interface  salt-metal coupling
C_DOWN = "#CC79A7"   # Gamma_Ni-down    Sieverts at P_down
C_EXT  = "#7A4A2F"   # Gamma_ext        outer surface, mesh.py tag 3 "out"
INK, INK2, AXIS = "#222222", "#666666", "#8A8A84"
NI_FILL, SALT_FILL, GAS_FILL = "#E3E2DE", "#CBDEF2", "#FFFFFF"
NI_EDGE = "#9A9A94"

# ── geometry, in millimetres, exactly as mesh.py sets it ─────────────────────
R_IN, R_OUT = 39.0, 41.0
Z_BT, Z_MB, Z_MT = 2.0, 22.0, 24.0
Z_FT = 29.14
Z_TIN, Z_TOUT = 109.1, 111.1

# All coordinates live in the section; the magnified corner carries only names.
ZOOM = (25.0, 45.0, 19.0, 33.0)         # r0, r1, z0, z1 of the magnified corner
FIG = (5.91, 4.20)      # 150 x 107 mm


def draw_domain(ax, lw=1.6):
    """Fills, then every named boundary stroked over them."""
    # The four Ni pieces are filled without outlines: they are one fused body in
    # mesh.py, so a per-rectangle outline would draw seams that do not exist.
    # Every real edge is stroked below in the colour of its physical group.
    for r0, r1, z0, z1 in ((R_IN, R_OUT, 0, Z_TOUT), (0, R_IN, 0, Z_BT),
                           (0, R_IN, Z_MB, Z_MT), (0, R_IN, Z_TIN, Z_TOUT)):
        ax.add_patch(Rectangle((r0, z0), r1 - r0, z1 - z0, facecolor=NI_FILL,
                               edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((0, Z_MT), R_IN, Z_FT - Z_MT, facecolor=SALT_FILL,
                           edgecolor="none", zorder=2))

    def seg(pts, color):
        for (a, b), (c, d) in zip(pts[:-1], pts[1:]):
            ax.plot([a, c], [b, d], color=color, lw=lw, zorder=5,
                    solid_capstyle="butt")

    # tags 5, 6 -- top cap underside, then the wall down to the salt surface
    seg([(0, Z_TIN), (R_IN, Z_TIN), (R_IN, Z_FT)], C_UP)
    # tag 8 -- the free surface of the salt
    seg([(0, Z_FT), (R_IN, Z_FT)], C_SALT)
    # tag 99 -- two curves: the salt's underside and its outer edge
    seg([(0, Z_MT), (R_IN, Z_MT), (R_IN, Z_FT)], C_INT)
    # tags 9, 7, 10 -- membrane underside, lower wall, bottom cap
    seg([(0, Z_MB), (R_IN, Z_MB), (R_IN, Z_BT), (0, Z_BT)], C_DOWN)
    # tag 3 -- the outer surface
    seg([(0, 0), (R_OUT, 0), (R_OUT, Z_TOUT), (0, Z_TOUT)], C_EXT)


def pt(ax, r, y, dx, dy, ha, va, fs=None):
    """A vertex, marked and named where it is.

    A tick column with leaders back to the geometry was the alternative and it
    filled the panel with lines that belong to no boundary; the coordinate
    written at the corner it describes needs no apparatus at all.
    """
    ax.plot([r], [y], marker="o", ms=2.6, color=INK, zorder=9, clip_on=False)
    ax.text(r + dx, y + dy, f"({r:g}, {y:g})", ha=ha, va=va,
            fontsize=fs or PT_D, color=INK, zorder=9, clip_on=False)


def draw_axis_line(ax, z0, z1):
    ax.plot([0, 0], [z0, z1], color=AXIS, lw=0.9,
            ls=(0, (6, 2.5, 1.5, 2.5)), zorder=6)


def label_zoom(ax):
    """The names, set where the thing they name is drawn."""
    ax.text(ZOOM[0] + 5.0, 0.5 * (Z_MT + Z_FT), r"$\Omega_{\rm FLiBe}$",
            ha="center", va="center", fontsize=PT_L, color="#2A6E8F", zorder=8)
    ax.text(ZOOM[0] + 5.0, 0.5 * (Z_MB + Z_MT), r"$\Omega_{\rm Ni}$",
            ha="center", va="center", fontsize=PT_B, color=INK2, zorder=8)
    ax.text(R_IN - 0.8, Z_FT + 2.6, r"$\Gamma_{\rm Ni-up}$", ha="right",
            va="center", fontsize=PT_B, color=C_UP, zorder=8)
    ax.text(R_IN - 0.8, Z_FT + 0.7, r"$\Gamma_{\rm FLiBe}$", ha="right",
            va="bottom", fontsize=PT_B, color=C_SALT, zorder=8)
    ax.text(R_IN - 0.8, Z_MT + 0.7, r"$\Gamma_{\rm Interface}$", ha="right",
            va="bottom", fontsize=PT_B, color=C_INT, zorder=8)
    ax.text(R_IN - 0.8, Z_MB - 0.9, r"$\Gamma_{\rm Ni-down}$", ha="right",
            va="top", fontsize=PT_B, color=C_DOWN, zorder=8)
    ax.text(R_OUT + 0.4, 27.5, r"$\Gamma_{\rm ext}$", ha="left", va="center",
            fontsize=PT_B, color=C_EXT, zorder=8)


CAPTION = (
    "Axisymmetric computational domain, drawn to scale, with the coordinates "
    "of its vertices in mm; the salt--membrane corner is magnified "
    "isotropically alongside. Only $\\Omega_{\\mathrm{FLiBe}}$ and "
    "$\\Omega_{\\mathrm{Ni}}$ are meshed, the gas volumes being represented by "
    "boundary conditions on the surfaces that bound them. "
    "$\\Gamma_{\\mathrm{Interface}}$ is two curves, not one: the salt column "
    "meets nickel along its outer edge as well as its underside. The salt "
    "stands on the membrane at $y = 24$~mm at every temperature and expands "
    "upward, so its free surface is the one vertex that moves."
)


def main() -> None:
    fig = plt.figure(figsize=FIG)

    XL, XR, ZB, ZT = -40.0, 70.0, -7.0, 118.0          # the section window
    H1 = 0.800
    w1 = ((XR - XL) / (ZT - ZB)) * (H1 * FIG[1]) / FIG[0]
    ax1 = fig.add_axes([0.035, 0.100, w1, H1])
    ax1.set_xlim(XL, XR)
    ax1.set_ylim(ZB, ZT)

    H2 = 0.360
    w2 = ((ZOOM[1] - ZOOM[0]) / (ZOOM[3] - ZOOM[2])) * (H2 * FIG[1]) / FIG[0]
    ax2 = fig.add_axes([0.550, 0.170, w2, H2])
    ax2.set_xlim(ZOOM[0], ZOOM[1])
    ax2.set_ylim(ZOOM[2], ZOOM[3])

    for ax, lw in ((ax1, 1.5), (ax2, 2.2)):
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        draw_domain(ax, lw=lw)

    # ── (a): the four outer corners, and the two the caps add ────────────────
    draw_axis_line(ax1, -2, Z_TOUT + 2)
    ax1.text(1.4, 0.60 * Z_TOUT, "symmetry, $r = 0$", ha="left", va="center",
             fontsize=PT_D, color=INK2, rotation=90, zorder=6)
    # Each coordinate is set against its own dot, no leaders. At true scale
    # z = 22 and 24 are 2 mm apart on a 111 mm column, so that pair is nudged
    # apart; the corners at z = 0 and 111.1 sit just outside the section.
    #
    # The salt's top face is the only vertex that moves: the salt rests on the
    # membrane at z = 24 at every temperature and expands upward, so it is
    # written 24 + L_FLiBe(T) rather than as a fixed number.
    Z_FT_TXT = r"$24 + L_{\rm FLiBe}(T)$"
    for side, rows in ((-1, ((0.0, 0.0, "0", -4.0), (0.0, Z_MB, "22", 20.5),
                             (0.0, Z_MT, "24", 25.5),
                             (0.0, Z_FT, Z_FT_TXT, 30.8),
                             (0.0, Z_TOUT, "111.1", 115.0))),
                       (+1, ((R_OUT, 0.0, "0", -4.0),
                             (R_OUT, Z_TOUT, "111.1", 115.0)))):
        for r, z, txt, z_lab in rows:
            ax1.plot([r], [z], marker="o", ms=2.6, color=INK, zorder=9,
                     clip_on=False)
            ax1.text(-2.2 if side < 0 else 42.6, z_lab, f"({r:g}, {txt})",
                     ha="right" if side < 0 else "left", va="center",
                     fontsize=PT_D, color=INK, zorder=9, clip_on=False)

    # The cap corners label into the cavity, offset 4 mm to clear the boundary
    # line running through the vertex.
    for z, z_lab, txt in ((Z_BT, 6.0, "2"), (Z_TIN, 105.0, "109.1")):
        ax1.plot([R_IN], [z], marker="o", ms=2.6, color=INK, zorder=9)
        ax1.text(R_IN - 2.0, z_lab, f"({R_IN:g}, {txt})", ha="right",
                 va="center", fontsize=PT_D, color=INK, zorder=9)

    ax1.text(0.5 * R_OUT, ZB - 3.0, "coordinates $(r,\\ z)$ in mm",
             ha="center", va="top", fontsize=PT_D, color=INK2)

    # ── the window, opened out onto (b) ──────────────────────────────────────
    ax1.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0],
                            ZOOM[3] - ZOOM[2], facecolor="none",
                            edgecolor=INK2, lw=0.8, ls=(0, (3, 2.5)), zorder=8))
    ax2.add_patch(Rectangle((ZOOM[0], ZOOM[2]), ZOOM[1] - ZOOM[0],
                            ZOOM[3] - ZOOM[2], facecolor="none",
                            edgecolor=INK2, lw=0.8, ls=(0, (3, 2.5)),
                            zorder=9, clip_on=False))
    # Corner to corner, so the leaders say which region was magnified.
    for yv in (ZOOM[3], ZOOM[2]):
        fig.add_artist(matplotlib.patches.ConnectionPatch(
            xyA=(ZOOM[1], yv), coordsA=ax1.transData,
            xyB=(ZOOM[0], yv), coordsB=ax2.transData,
            color=INK2, lw=0.7, ls=(0, (2, 2.5)), zorder=1))

    # ── (b): the names, and the three coordinates the stack turns on ─────────
    label_zoom(ax2)

    print(f"[caption] {CAPTION}")
    stem = OUTDIR / "domain_map"
    for ext in ("pdf", "svg"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=96)
    fig.savefig(f"{stem}_2x.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[saved] {stem}.pdf, .svg, .png, _2x.png")


if __name__ == "__main__":
    main()
