// Units: meters.  Right-half model; symmetry at x=0 (middle of the geometry).
SetFactory("OpenCASCADE");

// ---------- mesh sizes ----------
lc_solid = 1.0e-4; // target mesh size in solids (Ni)
lc_fluid = 1.0e-5; // target mesh size in fluid

// ---------- key coordinates (m) ----------
x_in   = 0.039;    // inner vertical Ni wall (3.90 cm)
x_out  = 0.041;    // outer vertical Ni wall (4.10 cm)

y0     = 0.000;     // bottom
y_bT   = 0.002;     // 0.20 cm (bottom Ni top)
y_mB   = 0.022;     // 2.20 cm (middle thin Ni bottom)
y_mT   = 0.024;     // 2.40 cm (middle thin Ni top)
y_fT   = 0.0319;    // 3.19 cm (fluid top)
y_tIn  = 0.1091;    // 10.91 cm (top Ni bottom / inner top)
y_tOut = 0.1111;    // 11.11 cm (top Ni outer top)

eps = 1e-8;

// ---------- primitives ----------
// Rectangle(id) = {x, y, z, dx, dy, dz}
Rectangle(1) = {0,     y0,   0,  x_in,       y_bT-y0,   0};          // Ni bottom
Rectangle(2) = {0,     y_mB, 0,  x_in,       y_mT-y_mB, 0};          // Ni middle thin
Rectangle(3) = {0,     y_tIn,0,  x_in,       y_tOut-y_tIn, 0};       // Ni top
Rectangle(4) = {x_in,  y0,   0,  x_out-x_in, y_tOut-y0,  0};         // Ni right wall
Rectangle(5) = {0,     y_mT, 0,  x_in,       y_fT-y_mT,  0};         // Fluid

// local mesh sizes
MeshSize { PointsOf{ Surface{5}; } } = lc_fluid;
MeshSize { PointsOf{ Surface{1,2,3,4}; } } = lc_solid;

// ---------- imprint & fragment ----------
BooleanFragments{ Surface{1,2,3,4,5}; Delete; }{}

// ---------- select final entities (use curly braces {} with BoundingBox) ----------
//something[] = Surface In BoundingBox{xmin, ymin, zmin, xmax, ymax, zmax};
ni_bottom[] = Surface In BoundingBox{0-eps,     y0-eps,   -eps, x_in+eps,     y_bT+eps,   eps};
ni_middle[] = Surface In BoundingBox{0-eps,     y_mB-eps, -eps, x_in+eps,     y_mT+eps,   eps};
ni_top[]    = Surface In BoundingBox{0-eps,     y_tIn-eps,-eps, x_in+eps,     y_tOut+eps, eps};
ni_right[]  = Surface In BoundingBox{x_in-eps,  y0-eps,   -eps, x_out+eps,    y_tOut+eps, eps};
fluid_s[]   = Surface In BoundingBox{0-eps,     y_mT-eps, -eps, x_in+eps,     y_fT+eps,   eps};

// ---------- physical SURFACES (IDs for FESTIM) ----------
// Give both a name and a fixed id so the 2D “volumes” appear in $PhysicalNames
Physical Surface("solid_ni", 1) = {ni_bottom[], ni_middle[], ni_top[], ni_right[]}; // solid_ni
Physical Surface("liquid",   2) = {fluid_s[]};                                      // liquid

// ---------- physical CURVES (boundaries) ----------
sym1[] = Curve In BoundingBox{0-eps, y0-eps,   -eps, 0+eps, y_bT+eps,   eps};
sym2[] = Curve In BoundingBox{0-eps, y_mB-eps, -eps, 0+eps, y_mT+eps,   eps};
sym3[] = Curve In BoundingBox{0-eps, y_mT-eps, -eps, 0+eps, y_fT+eps,   eps};
sym4[] = Curve In BoundingBox{0-eps, y_tIn-eps,-eps, 0+eps, y_tOut+eps, eps};
Physical Curve("Left_symmetry_bc") = {sym1[], sym2[], sym3[], sym4[]};    // symmetry

c_fluid_bottom[] = Curve In BoundingBox{0-eps,    y_mT-eps, -eps, x_in+eps, y_mT+eps, eps};
c_fluid_right[]  = Curve In BoundingBox{x_in-eps, y_mT-eps, -eps, x_in+eps, y_fT+eps, eps};
Physical Curve("liquid_gas_interface") = {c_fluid_bottom[]};               // fluid_wall_bottom
Physical Curve("Fluid_wall_right") = {c_fluid_right[]};                   // fluid_wall_right

c_fluid_top[] = Curve In BoundingBox{0-eps, y_fT-eps, -eps, x_in+eps, y_fT+eps, eps};
Physical Curve("Fluid_free_surface") = {c_fluid_top[]};                     // fluid_free_surface

c_ext_bottom[] = Curve In BoundingBox{0-eps,    y0-eps,    -eps, x_out+eps, y0+eps,    eps};
c_ext_right[]  = Curve In BoundingBox{x_out-eps,y0-eps,    -eps, x_out+eps, y_tOut+eps,eps};
c_ext_top[]    = Curve In BoundingBox{0-eps,    y_tOut-eps,-eps, x_out+eps, y_tOut+eps,eps};
Physical Curve("Ni_external") = {c_ext_bottom[], c_ext_right[], c_ext_top[]}; // ni_external (outside the domain)

// ---- extra interface picks for named labels (used only for mesh refinement) ----
// Top_Ni_bottom: bottom edge of top Ni at y = y_tIn, x in [0, x_in]
c_top_ni_bottom[]  = Curve In BoundingBox{0-eps,    y_tIn-eps, -eps, x_in+eps,  y_tIn+eps, eps};
// Mem_Ni_bottom: bottom edge of middle thin Ni at y = y_mB, x in [0, x_in]
c_mem_ni_bottom[]  = Curve In BoundingBox{0-eps,    y_mB-eps, -eps, x_in+eps,  y_mB+eps,  eps};
// Tottom_Ni_top: top edge of bottom Ni at y = y_bT, x in [0, x_in]
c_bottom_ni_top[]  = Curve In BoundingBox{0-eps,    y_bT-eps, -eps, x_in+eps,  y_bT+eps,  eps};
// Up_Ni_Left: inner vertical Ni wall facing upper H2, x = x_in, y in [y_mT, y_tIn]
c_inner_H2_upper[] = Curve In BoundingBox{x_in-eps, y_mT-eps, -eps, x_in+eps,  y_tIn+eps, eps};
// Ds_Ni_Left: inner vertical Ni wall facing lower H2, x = x_in, y in [y_bT, y_mB]
c_inner_H2_lower[] = Curve In BoundingBox{x_in-eps, y_bT-eps, -eps, x_in+eps,  y_mB+eps,  eps};
Physical Curve("Top_Ni_bottom") = {c_top_ni_bottom[]};
Physical Curve("Mem_Ni_bottom") = {c_mem_ni_bottom[]};
Physical Curve("Tottom_Ni_top") = {c_bottom_ni_top[]};
Physical Curve("Up_Ni_Left")    = {c_inner_H2_upper[]};
Physical Curve("Ds_Ni_Left")    = {c_inner_H2_lower[]};

// —— Mesh controls ——
//Mesh.Algorithm = 6;      // Frontal-Delaunay, triangle;
//Mesh.ElementOrder = 1;   // Linear;

// —— global limits that the background field can use ——
Mesh.CharacteristicLengthFromPoints    = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthMin = 1.0e-6;   // fine
Mesh.CharacteristicLengthMax = 2.0e-4;   // coarse
