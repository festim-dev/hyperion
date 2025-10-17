import gmsh

y_ft = 0.02930  # default; will be overwritten by set_y_ft(...)


def set_y_ft(value: float) -> None:
    """Set FLiBe thickness [m] before calling generate_mesh()."""
    global y_ft
    y_ft = float(value)


def generate_mesh(mesh_size=2e-4, fname="mesh.msh"):
    # Initialize the GMSH API
    gmsh.initialize()
    gmsh.model.add("HYPERION")

    # ---------- key coordinates (m) ----------
    x_in = 0.039  # inner vertical Ni wall (3.90 cm)
    x_out = 0.041  # outer vertical Ni wall (4.10 cm)

    y0 = 0.000  # bottom
    y_bT = 0.002  # 0.20 cm (bottom Ni top)
    y_mB = 0.022  # 2.20 cm (middle thin Ni bottom)
    y_mT = 0.024  # 2.40 cm (middle thin Ni top)
    # thisckness of salt layer in 500 C is 0.00514 m
    y_fT = 0.02914
    y_tIn = 0.1091  # 10.91 cm (top Ni bottom / inner top)
    y_tOut = 0.1111  # 11.11 cm (top Ni outer top)

    # make rectangles
    solid_bottom = gmsh.model.occ.addRectangle(0, y0, 0, x_in, y_bT - y0, tag=1)
    solid_middle = gmsh.model.occ.addRectangle(0, y_mB, 0, x_in, y_mT - y_mB, tag=2)
    solid_top = gmsh.model.occ.addRectangle(0, y_tIn, 0, x_in, y_tOut - y_tIn, tag=3)
    solid_right = gmsh.model.occ.addRectangle(
        x_in, y0, 0, x_out - x_in, y_tOut - y0, tag=4
    )
    salt_rectangle = gmsh.model.occ.addRectangle(0, y_mT, 0, x_in, y_fT - y_mT, tag=5)

    # fuse all solid parts
    gmsh.model.occ.fuse([(2, 1), (2, 2), (2, 3)], [(2, 4)], tag=10)

    # make a fragment
    outDimTags, outDimTagsMap = gmsh.model.occ.fragment(
        [(2, 5)],  # object entities (dimTag pairs)
        [(2, 10)],  # tool entities (dimTag pairs)
    )
    # synchronize CAD kernel with GMSH model
    gmsh.model.occ.synchronize()
    print(gmsh.model.getEntities(2))

    # mark volumes

    fluid = gmsh.model.addPhysicalGroup(2, [5], tag=1)
    gmsh.model.setPhysicalName(2, fluid, "fluid")

    solid = gmsh.model.addPhysicalGroup(2, [10], tag=2)
    gmsh.model.setPhysicalName(2, solid, "solid")

    # mark boundaries
    print(gmsh.model.getEntities(1))
    out = gmsh.model.addPhysicalGroup(1, [5, 6, 7], tag=3)
    gmsh.model.setPhysicalName(1, out, "out")

    left_bc_liquid = gmsh.model.addPhysicalGroup(1, [4], tag=41)
    gmsh.model.setPhysicalName(1, left_bc_liquid, "left_bc_liquid")

    left_bc_top_Ni = gmsh.model.addPhysicalGroup(1, [8], tag=42)
    gmsh.model.setPhysicalName(1, left_bc_top_Ni, "left_bc_top_Ni")

    left_bc_middle_Ni = gmsh.model.addPhysicalGroup(1, [11], tag=43)
    gmsh.model.setPhysicalName(1, left_bc_middle_Ni, "left_bc_middle_Ni")

    left_bc_bottom_Ni = gmsh.model.addPhysicalGroup(1, [15], tag=44)
    gmsh.model.setPhysicalName(1, left_bc_bottom_Ni, "left_bc_bottom_Ni")

    top_cap_Ni = gmsh.model.addPhysicalGroup(1, [9], tag=5)
    gmsh.model.setPhysicalName(1, top_cap_Ni, "top_cap_Ni")

    top_sidewall_Ni = gmsh.model.addPhysicalGroup(1, [10], tag=6)
    gmsh.model.setPhysicalName(1, top_sidewall_Ni, "top_sidewall_Ni")

    bottom_sidewall_Ni = gmsh.model.addPhysicalGroup(1, [13], tag=7)
    gmsh.model.setPhysicalName(1, bottom_sidewall_Ni, "bottom_sidewall_Ni")

    liquid_surface = gmsh.model.addPhysicalGroup(1, [3], tag=8)
    gmsh.model.setPhysicalName(1, liquid_surface, "liquid_surface")

    mid_membrane_Ni = gmsh.model.addPhysicalGroup(1, [12], tag=9)
    gmsh.model.setPhysicalName(1, mid_membrane_Ni, "mid_membrane_Ni")

    bottom_cap_Ni = gmsh.model.addPhysicalGroup(1, [14], tag=10)
    gmsh.model.setPhysicalName(1, bottom_cap_Ni, "bottom_cap_Ni")

    boundary_liquid = set(
        gmsh.model.getBoundary([(2, 5)], oriented=False, recursive=False)
    )
    boundary_solid = set(
        gmsh.model.getBoundary([(2, 10)], oriented=False, recursive=False)
    )

    interface_curves = list(boundary_liquid.intersection(boundary_solid))

    curve_tags = [c[1] for c in interface_curves]  # [1, 2]
    liquid_Ni_interface = gmsh.model.addPhysicalGroup(1, curve_tags, tag=99)
    gmsh.model.setPhysicalName(1, liquid_Ni_interface, "liquid_Ni_interface")

    # set CharacteristicLengthMax
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)

    # write to file
    gmsh.write(fname)


#    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()
    gmsh.fltk.run()
