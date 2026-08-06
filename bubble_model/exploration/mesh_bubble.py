import gmsh

# Geometry (m)
Lx = 0.078
t_s = 0.002032
t_l = 0.005363582


hole_len = 0.02
# t_b = 1e-4  # bubble gap thickness (m)
t_b = 0.001  # bubble gap thickness (m)

t_extra = hole_len * t_b / Lx

Htot = t_s + t_l + t_extra


def generate_mesh(mesh_size=2e-4, fname="2Dbubble.msh"):
    gmsh.initialize()
    gmsh.model.add("LIQUID_SOLID_HOLE")

    y_int = t_s
    x0 = 0.5 * (Lx - hole_len)
    x1 = 0.5 * (Lx + hole_len)

    # --- surfaces ---
    # solid
    gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, t_s, tag=1)

    # liquid_left
    gmsh.model.occ.addRectangle(0.0, y_int, 0.0, x0, Htot - y_int, tag=2)
    # liquid_mid
    gmsh.model.occ.addRectangle(
        x0, y_int + t_b, 0.0, x1 - x0, Htot - (y_int + t_b), tag=3
    )
    # liquid_right
    gmsh.model.occ.addRectangle(x1, y_int, 0.0, Lx - x1, Htot - y_int, tag=4)

    # liquid domain is the union of left, mid, right
    gmsh.model.occ.fuse([(2, 2), (2, 3)], [(2, 4)], tag=10)

    # -------------------
    # Fragment bubble with solid+liquid to enforce:
    # - solid-liquid contact outside hole
    # - solid-bubble and liquid-bubble in the hole
    # -------------------
    gmsh.model.occ.fragment([(2, 1)], [(2, 10)])
    gmsh.model.occ.synchronize()

    # -------------------
    # Physical groups for domains and boundaries. Use intersection logic to find interfaces.
    # -------------------
    solid = gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.setPhysicalName(2, solid, "solid")

    liquid = gmsh.model.addPhysicalGroup(2, [10], tag=2)
    gmsh.model.setPhysicalName(2, liquid, "liquid")

    # -------------------
    # Boundaries: use intersection logic
    # -------------------
    boundary_liquid = set(
        gmsh.model.getBoundary([(2, 10)], oriented=False, recursive=False)
    )
    boundary_solid = set(
        gmsh.model.getBoundary([(2, 1)], oriented=False, recursive=False)
    )

    # liquid-solid interface (wet contact only; bubble removes the middle)
    ls_curves = list(boundary_liquid.intersection(boundary_solid))
    ls_tags = [c[1] for c in ls_curves]
    gmsh.model.addPhysicalGroup(1, ls_tags, tag=101)
    gmsh.model.setPhysicalName(1, 101, "liquid_solid_interface")

    # surface
    print("Curves:", gmsh.model.getEntities(1))
    left_boundary = gmsh.model.addPhysicalGroup(1, [6, 12], tag=31)
    gmsh.model.setPhysicalName(1, left_boundary, "left_boundary")

    right_boundary = gmsh.model.addPhysicalGroup(1, [2, 10], tag=32)
    gmsh.model.setPhysicalName(1, right_boundary, "right_boundary")

    top_boundary = gmsh.model.addPhysicalGroup(1, [11], tag=33)
    gmsh.model.setPhysicalName(1, top_boundary, "top_boundary")

    bottom_boundary = gmsh.model.addPhysicalGroup(1, [1], tag=34)
    gmsh.model.setPhysicalName(1, bottom_boundary, "bottom_boundary")

    liquid_gas_surface = gmsh.model.addPhysicalGroup(1, [7, 8, 9], tag=35)
    gmsh.model.setPhysicalName(1, liquid_gas_surface, "liquid_gas_surface")

    solid_gas_surface = gmsh.model.addPhysicalGroup(1, [4], tag=36)
    gmsh.model.setPhysicalName(1, solid_gas_surface, "solid_gas_surface")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)
    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()
    # gmsh.fltk.run()
