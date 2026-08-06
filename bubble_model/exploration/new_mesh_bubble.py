import gmsh

# Geometry (m)
Lx = 0.078
t_s = 0.002032
t_l = 0.005363582
t_b = 0.001  # bubble gap thickness (m)

Htot = t_s + t_l + t_b


def generate_mesh(mesh_size=2e-4, fname="2Dbubble_fullwidth.msh"):
    gmsh.initialize()
    gmsh.model.add("LIQUID_SOLID_FULLWIDTH_BUBBLE")

    y_int = t_s
    y_liq0 = y_int + t_b  # liquid starts above bubble gap

    # Surfaces
    solid_tag = gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, t_s, tag=1)
    liquid_tag = gmsh.model.occ.addRectangle(0.0, y_liq0, 0.0, Lx, Htot - y_liq0, tag=2)
    gmsh.model.occ.synchronize()

    # Physical groups: domains
    solid_pg = gmsh.model.addPhysicalGroup(2, [solid_tag], tag=1)
    gmsh.model.setPhysicalName(2, solid_pg, "solid")

    liquid_pg = gmsh.model.addPhysicalGroup(2, [liquid_tag], tag=2)
    gmsh.model.setPhysicalName(2, liquid_pg, "liquid")

    # Collect boundary curves for solid & liquid
    b_solid = gmsh.model.getBoundary([(2, solid_tag)], oriented=False, recursive=False)
    b_liquid = gmsh.model.getBoundary(
        [(2, liquid_tag)], oriented=False, recursive=False
    )
    all_curves = set(b_solid) | set(b_liquid)

    # Classify curves by center of mass
    left, right, top, bottom, sg, lg = [], [], [], [], [], []
    for dim, ctag in all_curves:
        if dim != 1:
            continue
        x, y, _ = gmsh.model.occ.getCenterOfMass(1, ctag)

        if abs(x - 0.0) < 1e-8:
            left.append(ctag)
        if abs(x - Lx) < 1e-8:
            right.append(ctag)

        if abs(y - 0.0) < 1e-8:
            bottom.append(ctag)
        if abs(y - Htot) < 1e-8:
            top.append(ctag)

        # Bubble-facing surfaces:
        # solid_gas_surface at y = t_s
        if abs(y - t_s) < 1e-8:
            sg.append(ctag)
        # liquid_gas_surface at y = t_s + t_b
        if abs(y - (t_s + t_b)) < 1e-8:
            lg.append(ctag)

    if len(sg) == 0:
        raise RuntimeError("Could not find solid_gas_surface (y=t_s).")
    if len(lg) == 0:
        raise RuntimeError("Could not find liquid_gas_surface (y=t_s+t_b).")

    pg_left = gmsh.model.addPhysicalGroup(1, left, tag=31)
    gmsh.model.setPhysicalName(1, pg_left, "left_boundary")

    pg_right = gmsh.model.addPhysicalGroup(1, right, tag=32)
    gmsh.model.setPhysicalName(1, pg_right, "right_boundary")

    pg_top = gmsh.model.addPhysicalGroup(1, top, tag=33)
    gmsh.model.setPhysicalName(1, pg_top, "top_boundary")

    pg_bottom = gmsh.model.addPhysicalGroup(1, bottom, tag=34)
    gmsh.model.setPhysicalName(1, pg_bottom, "bottom_boundary")

    pg_lg = gmsh.model.addPhysicalGroup(1, lg, tag=35)
    gmsh.model.setPhysicalName(1, pg_lg, "liquid_gas_surface")

    pg_sg = gmsh.model.addPhysicalGroup(1, sg, tag=36)
    gmsh.model.setPhysicalName(1, pg_sg, "solid_gas_surface")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)
    # gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()
    gmsh.fltk.run()
