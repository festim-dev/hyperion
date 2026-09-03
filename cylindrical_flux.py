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
        restriction: str | None = None,
        subdomain_id: int | None = None,
    ):
        """Computes the value of the flux at the surface

        J =int(- D * grad(c) . n * r dS)

        Args:
            u: field for which the flux is computed
            ds: surface measure of the model
            entity_maps: entity maps relating parent mesh and submesh
            restriction: side of an interior facet to evaluate the flux on
            subdomain_id: id to index ds with, when it is not the surface's own
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

        if subdomain_id is None:
            subdomain_id = self.surface.id
        integrand = -self.D * r * ufl.dot(ufl.grad(u), n)
        if restriction is not None:
            integrand = ufl.as_ufl(integrand)(restriction)

        flux = assemble_scalar(
            dolfinx.fem.form(
                integrand * ds(subdomain_id),
                entity_maps=entity_maps,
            )
        )

        flux *= self.azimuth_range[1] - self.azimuth_range[0]

        self.value = flux
        self.data.append(self.value)
