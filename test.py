import festim as F
from mpi4py import MPI
import numpy as np
import ufl
import basix
import dolfinx
import matplotlib.pyplot as plt
import os


class DiscontinuousMesh1D(F.Mesh1D):
    def __init__(self, vertices1, vertices2, **kwargs):
        self.vertices1 = vertices1
        self.vertices2 = vertices2
        self.vertices = np.concatenate((vertices1, vertices2))
        mesh = self.generate_mesh()
        F.Mesh.__init__(self, mesh=mesh, **kwargs)

    def generate_mesh(self):
        if MPI.COMM_WORLD.rank == 0:
            mesh_points = np.reshape(self.vertices, (len(self.vertices), 1))

            n1 = self.vertices1.shape[0]
            n2 = self.vertices2.shape[0]

            indexes1 = np.arange(n1)
            indexes2 = np.arange(n1, n1 + n2)

            cells1 = np.stack((indexes1[:-1], indexes1[1:]), axis=-1)
            cells2 = np.stack((indexes2[:-1], indexes2[1:]), axis=-1)
            cells = np.concatenate((cells1, cells2))
        else:
            mesh_points = np.empty((0, 1), dtype=np.float64)
            cells = np.empty((0, 2), dtype=np.int64)

        degree = 1
        domain = ufl.Mesh(
            basix.ufl.element(basix.ElementFamily.P, "interval", degree, shape=(1,))
        )

        return dolfinx.mesh.create_mesh(
            comm=MPI.COMM_WORLD, cells=cells, x=mesh_points, e=domain
        )

    def check_borders(self, volume_subdomains):
        # Keep FESTIM happy if it calls this; we don’t need extra checks here.
        pass


# PARAMETERS
# c_up = 1.0
# c_down = 0.0
c_up = 0.0
c_down = 1.0

Te_Cs = (500, 550, 600, 650, 700)
temperatures = tuple(float(T) + 273.15 for T in Te_Cs)
os.makedirs("plots", exist_ok=True)
fig, ax = plt.subplots(1, 1, sharex=True)

for temperature in temperatures:
    # PROBLEM (no bubble)
    problem = F.HydrogenTransportProblemDiscontinuous()

    salt = F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0.2, solubility_law="henry")
    metal = F.Material(D_0=2, E_D=0, K_S_0=3, E_K_S=0.2, solubility_law="sievert")

    left = F.SurfaceSubdomain1D(id=3, x=0.0)
    right = F.SurfaceSubdomain1D(id=4, x=2.0)

    left_volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=salt)
    right_volume = F.VolumeSubdomain1D(id=2, borders=[1.0, 2.0], material=metal)

    # Interface coupling at x=1.0 between the two volumes
    interface = F.Interface(
        id=5, subdomains=[left_volume, right_volume], penalty_term=1e3
    )
    problem.interfaces = [interface]
    problem.subdomains = [left_volume, right_volume, left, right]

    problem.surface_to_volume = {
        left: left_volume,
        right: right_volume,
    }

    H = F.Species("H", subdomains=problem.volume_subdomains)
    problem.species = [H]

    problem.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, species=H, value=c_up),
        F.FixedConcentrationBC(subdomain=right, species=H, value=c_down),
    ]

    vertices_salt = np.linspace(0, 1, num=500)
    vertices_metal = np.linspace(1, 2, num=500)[1:]
    vertices = np.concatenate([vertices_salt, vertices_metal])
    problem.mesh = F.Mesh1D(vertices)

    problem.temperature = temperature

    problem.settings = F.Settings(
        atol=1e-9,
        rtol=1e-7,
        final_time=1000,
    )

    problem.settings.stepsize = F.Stepsize(
        initial_value=0.0001,
        growth_factor=1.01,
        cutback_factor=0.9,
        target_nb_iterations=4,
    )

    # Track downstream (pick left or right)
    flux_out = F.SurfaceFlux(field=H, surface=left, filename=None)
    problem.exports = [flux_out]

    problem.initialise()
    # dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    problem.run()

    t = np.asarray(flux_out.t, dtype=float)
    j = np.asarray(flux_out.data, dtype=float)

    T_C = temperature - 273.15
    filename = f"plots/nobubble_flux_c_up_{c_up:.3f}_T_{T_C:.0f}C.csv"

    np.savetxt(
        filename,
        np.column_stack([t, j]),
        delimiter=",",
        header="time,flux",
        comments="",
    )

    print(f"Saved {filename}")

    ax.plot(flux_out.t, flux_out.data, label=f"flux_out, T={temperature}K")

ax.set_ylabel("Flux (H/s)")
ax.set_xlabel("Time")
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

plt.tight_layout()
plt.show()
