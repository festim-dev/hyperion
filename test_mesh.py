import gmsh

# Geometry (m)
Lx = 0.078
t_s = 0.002032
t_l = 0.005363582
Htot = t_s + t_l


def generate_mesh(mesh_size=2e-4, fname="2D.msh"):
    gmsh.initialize()
    gmsh.model.add("LIQUID_SOLID_HOLE")

    y_int = t_s
    # solid
    gmsh.model.occ.addRectangle(0.0, 0.0, 0.0, Lx, t_s, tag=1)
    # liquid
    gmsh.model.occ.addRectangle(0.0, y_int, 0.0, Lx, Htot - y_int, tag=2)

    gmsh.model.occ.fragment([(2, 1)], [(2, 2)])
    gmsh.model.occ.synchronize()

    solid = gmsh.model.addPhysicalGroup(2, [1], tag=1)
    gmsh.model.setPhysicalName(2, solid, "solid")

    liquid = gmsh.model.addPhysicalGroup(2, [2], tag=2)
    gmsh.model.setPhysicalName(2, liquid, "liquid")

    boundary_liquid = set(
        gmsh.model.getBoundary([(2, 2)], oriented=False, recursive=False)
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
    left_boundary = gmsh.model.addPhysicalGroup(1, [4, 7], tag=31)
    gmsh.model.setPhysicalName(1, left_boundary, "left_boundary")

    right_boundary = gmsh.model.addPhysicalGroup(1, [2, 5], tag=32)
    gmsh.model.setPhysicalName(1, right_boundary, "right_boundary")

    top_boundary = gmsh.model.addPhysicalGroup(1, [6], tag=33)
    gmsh.model.setPhysicalName(1, top_boundary, "top_boundary")

    bottom_boundary = gmsh.model.addPhysicalGroup(1, [1], tag=34)
    gmsh.model.setPhysicalName(1, bottom_boundary, "bottom_boundary")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(fname)
    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh()
    # gmsh.fltk.run()
