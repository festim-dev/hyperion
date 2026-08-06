"""Cylindrical (axisymmetric) mesh for the FLiBe/Ni permeation cell with an
explicit *gas pocket* between the FLiBe cup bottom and the mid-Ni membrane.

Geometry (r,y) — axisymmetric around r = 0
==========================================

   y_tOut +-------------+ <- top_cap_Ni (full disk)
          |             |
   y_tIn  +---------+   |  liquid_surface (full disk at y_fT)
          |         |   |
          |  FLiBe  |Ni |  outer Ni sidewall (annulus r in [x_in, x_out])
          |         |   |
   y_fT   +---------+   |    .
          |         |   |    .
   y_lg   +---------+   |  <- liquid_gas_surface (top of bubble = FLiBe bottom)
          | bubble  |   |   <- gas pocket  (full disk, r in [0, r_b])
   y_sg   +-+-------+   |  <- solid_gas_surface (top of mid-Ni, partial)
   y_mT   +-+-------+   |  <- liquid_Ni_interface (annular ring r in [r_b, x_in])
          |         |   |
          | mid Ni  |   |
   y_mB   +---------+---+
          |             |
          | outer Ni    |
          | (annulus)   |
   y_bT   +-------------+ <- bottom_cap_Ni (full disk)
          |             |
   0      +-------------+

The bubble is a *thin disk* of inner radius 0, outer radius r_b, thickness t_b,
sitting on top of the mid-Ni membrane.  Its volume V_b = pi * r_b^2 * t_b.

Surfaces relevant to the bubble dynamics
----------------------------------------
- liquid_gas_surface (tag 35): top of the bubble (touches FLiBe).
- solid_gas_surface (tag 36): bottom of the bubble (touches mid-Ni).
  Neither gets a boundary condition in transport_model.py, so both default
  to no-flux: the gas pocket blocks H exchange over the area it covers.
- liquid_Ni_interface (tag 99): the open part of the FLiBe-Ni horizontal
  interface (annular ring around the bubble). Normal Henry/Sievert coupling
  applies here — H2 still permeates through the bubble-free part.
"""

import gmsh


# --- key coordinates (m) ---------------------------------------------------
X_IN = 0.039        # inner cylinder radius
X_OUT = 0.041       # outer cylinder radius (Ni wall)
Y0 = 0.0            # bottom
Y_bT = 0.002        # top of bottom Ni cap
Y_mB = 0.022        # bottom of mid Ni membrane
Y_mT = 0.024        # top of mid Ni membrane (= bottom of bubble pocket)
Y_tIn = 0.1091      # bottom of top Ni cap
Y_tOut = 0.1111     # top of top Ni cap


def generate_axisym_bubble_mesh(
    y_ft: float,
    r_b: float = 0.020,    # m, bubble radius (default 20 mm — about 26% of cross-section area)
    t_b: float = 5e-4,     # m, bubble thickness (default 0.5 mm)
    mesh_size: float = 4e-4,
    fname: str = "axisym_bubble_mesh.msh",
):
    """Generate the cylindrical mesh with a bubble pocket.

    Parameters
    ----------
    y_ft : float
        FLiBe top surface height (varies with T due to thermal expansion).
    r_b : float
        Bubble outer radius (m).  Must be < X_IN.
    t_b : float
        Bubble thickness (m).
    mesh_size : float
        Characteristic mesh length (m).
    fname : str
        Output mesh file.

    Returns
    -------
    dict with the geometric volumes/physical groups used.
    """
    if r_b >= X_IN:
        raise ValueError(f"r_b={r_b} must be < X_IN={X_IN}")
    if y_ft <= Y_mT + t_b:
        raise ValueError(f"y_ft={y_ft} too small for bubble thickness t_b={t_b}")

    y_lg = Y_mT + t_b      # bubble top   == FLiBe bottom over the bubble

    gmsh.initialize()
    gmsh.model.add("HYPERION_AXISYM_BUBBLE")

    # ----- create the Ni rectangles (same as mesh.py) -----
    bot_cap = gmsh.model.occ.addRectangle(0, Y0, 0, X_IN, Y_bT - Y0, tag=1)
    mid_membrane = gmsh.model.occ.addRectangle(0, Y_mB, 0, X_IN, Y_mT - Y_mB, tag=2)
    top_cap = gmsh.model.occ.addRectangle(0, Y_tIn, 0, X_IN, Y_tOut - Y_tIn, tag=3)
    outer_wall = gmsh.model.occ.addRectangle(X_IN, Y0, 0, X_OUT - X_IN,
                                             Y_tOut - Y0, tag=4)

    # ----- FLiBe (with a HOLE for the bubble) -----
    # The FLiBe occupies r=[0,X_IN], y=[Y_mT, y_ft] EXCEPT for the bubble
    # pocket (r=[0,r_b], y=[Y_mT, Y_mT+t_b]).  We build it as the difference
    # of a full FLiBe rectangle minus the bubble rectangle, leaving a
    # genuine void where the gas pocket sits.
    flibe_full = gmsh.model.occ.addRectangle(0, Y_mT, 0, X_IN, y_ft - Y_mT, tag=5)
    bubble_void = gmsh.model.occ.addRectangle(0, Y_mT, 0, r_b, t_b, tag=6)
    flibe_with_hole, _ = gmsh.model.occ.cut(
        [(2, flibe_full)], [(2, bubble_void)], tag=10, removeObject=True, removeTool=True
    )

    # ----- combine all Ni parts into one volume -----
    ni_solid, _ = gmsh.model.occ.fuse(
        [(2, bot_cap), (2, mid_membrane), (2, top_cap)], [(2, outer_wall)], tag=20
    )

    # ----- fragment to ensure interfaces are well-defined -----
    gmsh.model.occ.fragment(
        [(2, 10)],
        [(2, 20)],
    )
    gmsh.model.occ.synchronize()

    # ----- identify the surfaces to tag them -----
    # We'll use coordinate-based classification of curve centers.
    boundary_flibe = gmsh.model.getBoundary([(2, 10)], oriented=False, recursive=False)
    boundary_ni = gmsh.model.getBoundary([(2, 20)], oriented=False, recursive=False)

    s_flibe = {c[1] for c in boundary_flibe if c[0] == 1}
    s_bubble = set()
    s_ni = {c[1] for c in boundary_ni if c[0] == 1}

    def com(ctag):
        return gmsh.model.occ.getCenterOfMass(1, ctag)

    # Mark physical groups
    # ----- volumes -----
    flibe_pg = gmsh.model.addPhysicalGroup(2, [10], tag=1)
    gmsh.model.setPhysicalName(2, flibe_pg, "fluid")
    ni_pg = gmsh.model.addPhysicalGroup(2, [20], tag=2)
    gmsh.model.setPhysicalName(2, ni_pg, "solid")
    # The bubble interior is NOT a volume we want to mesh in FESTIM (we treat
    # it analytically via mass balance), so we skip a physical group for it
    # — gmsh will still mesh it, but FESTIM doesn't include it as a domain.

    # ----- boundary classification -----
    eps = 1e-7
    out_curves = []          # outer surface of cell (out_surf)
    bottom_cap = []          # bottom of bottom Ni
    top_cap = []             # top of top Ni
    liquid_surface = []      # FLiBe top
    liquid_gas = []          # FLiBe bottom over bubble
    solid_gas = []           # Ni top over bubble (top of mid Ni)
    flibe_ni_horiz = []      # FLiBe-Ni horizontal interface (annular ring outside bubble)
    flibe_ni_vert = []       # FLiBe-Ni vertical interface (r=X_IN, y in [y_lg, y_ft])
    mid_membrane_top_open = []  # top of mid Ni outside bubble (= flibe_ni_horiz from Ni side)
    mid_membrane_bottom = []  # bottom of mid Ni at y=Y_mB
    bottom_sidewall = []     # outer Ni at y=Y0..Y_bT  (actually the bottom of the outer-Ni annulus is also bottom_cap)
    # We'll be pragmatic: anything outside is in "out_curves"; anything at the
    # very bottom (y=0) is bottom_cap.

    all_curves = (s_flibe | s_ni)
    for ctag in all_curves:
        x, y, _ = com(ctag)
        # outer boundary (r = X_OUT)
        if abs(x - X_OUT) < eps:
            out_curves.append(ctag); continue
        # bottom of cell (y=0, r in [0, X_OUT])
        if abs(y - Y0) < eps:
            bottom_cap.append(ctag); continue
        # top of cell (y=Y_tOut)
        if abs(y - Y_tOut) < eps:
            top_cap.append(ctag); continue
        # FLiBe top surface (liquid_surface at y_ft)
        if abs(y - y_ft) < eps and x < X_IN + eps:
            liquid_surface.append(ctag); continue
        # ---- bubble surfaces ----
        # bubble TOP: y = y_lg = Y_mT + t_b, r in [0, r_b]  (FLiBe-gas)
        if abs(y - y_lg) < eps and x < r_b + eps:
            liquid_gas.append(ctag); continue
        # bubble SIDE: r = r_b, y in [Y_mT, y_lg]            (FLiBe-gas)
        if abs(x - r_b) < eps and (Y_mT - eps < y < y_lg + eps):
            liquid_gas.append(ctag); continue
        # bubble BOTTOM: y = Y_mT, r in [0, r_b]             (Ni-gas)
        if abs(y - Y_mT) < eps and x < r_b + eps:
            solid_gas.append(ctag); continue
        # ---- liquid_Ni_interface ----
        # horizontal: y = Y_mT, r in [r_b, X_IN]
        if abs(y - Y_mT) < eps and x > r_b - eps and x < X_IN + eps:
            flibe_ni_horiz.append(ctag); continue
        # vertical: r = X_IN, y in [Y_mT, y_ft]
        if abs(x - X_IN) < eps and (Y_mT - eps < y < y_ft + eps):
            flibe_ni_vert.append(ctag); continue
        # ---- mid Ni bottom (downstream) ----
        if abs(y - Y_mB) < eps and x < X_IN + eps:
            mid_membrane_bottom.append(ctag); continue
        # other internal curves (top of bot_cap, bottom of top_cap): skip

    def addpg(curves, tag, name):
        if not curves:
            return None
        pg = gmsh.model.addPhysicalGroup(1, curves, tag=tag)
        gmsh.model.setPhysicalName(1, pg, name)
        return pg

    addpg(out_curves, 3, "out")
    addpg(top_cap, 5, "top_cap_Ni")
    addpg(bottom_cap, 10, "bottom_cap_Ni")
    addpg(liquid_surface, 8, "liquid_surface")
    addpg(liquid_gas, 35, "liquid_gas_surface")
    addpg(solid_gas, 36, "solid_gas_surface")
    addpg(flibe_ni_horiz + flibe_ni_vert, 99, "liquid_Ni_interface")
    addpg(mid_membrane_bottom, 9, "mid_membrane_Ni")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)

    info = dict(
        r_b=r_b, t_b=t_b, V_b=3.14159265 * r_b ** 2 * t_b,
        A_bubble_top=3.14159265 * r_b ** 2,
        A_bubble_bot=3.14159265 * r_b ** 2,
        A_mid_membrane=3.14159265 * X_IN ** 2,
        y_ft=y_ft, y_lg=y_lg, y_mT=Y_mT,
        fname=fname,
    )
    gmsh.finalize()
    return info


if __name__ == "__main__":
    info = generate_axisym_bubble_mesh(
        y_ft=0.02930, r_b=0.020, t_b=5e-4,
        mesh_size=6e-4,
        fname="axisym_bubble_test.msh",
    )
    print("=" * 60)
    print("Bubble pocket geometry:")
    for k, v in info.items():
        print(f"  {k:>16s} = {v}")
