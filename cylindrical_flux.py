import ufl
import numpy as np
import dolfinx
import festim as F


# here we define our own SurfaceFlux class that takes into account the cylindrical coordinate system


class CylindricalFlux(F.SurfaceFlux):
    azimuth_range: tuple[float, float] = (0.0, 2 * np.pi)

    def compute(
        self,
        u: dolfinx.fem.Function | ufl.indexed.Indexed,
        ds: ufl.Measure,
        entity_maps=None,
    ):
        """Computes the value of the flux at the surface

        J =int(- D * grad(c) . n * r dS)

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
        """
        from scifem import assemble_scalar

        # obtain mesh normal from field
        # if case multispecies, solution is an index, use sub_function_space
        if isinstance(u, ufl.indexed.Indexed):
            mesh = self.field.sub_function_space.mesh
        else:
            mesh = u.function_space.mesh
        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)
        r = x[0]

        flux = assemble_scalar(
            dolfinx.fem.form(
                -self.D * r * ufl.dot(ufl.grad(u), n) * ds(self.surface.id),
                entity_maps=entity_maps,
            )
        )

        flux *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = flux
        self.data.append(self.value)
