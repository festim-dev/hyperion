import gmsh


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
    y_fT = 0.0319  # 3.19 cm (fluid top)
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
    bottom = gmsh.model.addPhysicalGroup(1, [4], tag=3)
    gmsh.model.setPhysicalName(1, bottom, "bottom")

    # set CharacteristicLengthMax
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.generate(2)

    # write to file
    gmsh.write(fname)

    gmsh.finalize()


if __name__ == "__main__":
    gmsh.fltk.run()
