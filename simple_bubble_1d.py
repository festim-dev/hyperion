import festim as F
from mpi4py import MPI
import numpy as np
import ufl
import basix
import dolfinx
import matplotlib.pyplot as plt


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
            indexes1 = np.arange(self.vertices1.shape[0])
            indexes2 = np.arange(
                self.vertices1.shape[0],
                self.vertices1.shape[0] + self.vertices2.shape[0],
            )

            # cells = np.stack((indexes[:-1], indexes[1:]), axis=-1)
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
        """Checks that the borders of the subdomain are within the domain

        Args:
            volume_subdomains (list of festim.VolumeSubdomain1D): the volume subdomains

        Raises:
            Value error: if borders outside the domain
        """
        pass


class Custom1DProblem(F.HydrogenTransportProblemDiscontinuous):
    def iterate(self):
        global old_N_b
        super().iterate()

        j_liquid_gas = float(flux_lg.value)
        j_solid_gas = float(flux_sg.value)

        Fnet = j_liquid_gas + j_solid_gas

        new_N_b = max(old_N_b + float(self.dt.value) * Fnet, 0.0)
        new_pb = max(new_N_b * F.R * temperature / V_b, 0.0)

        bc_liquid_gas.value = new_pb * K_H
        bc_solid_gas.value = (new_pb**0.5) * K_S if new_pb > 0.0 else 0.0

        idx_bclg = self.boundary_conditions.index(bc_liquid_gas)
        idx_bcsg = self.boundary_conditions.index(bc_solid_gas)
        self.bc_forms[idx_bclg] = self.create_dirichletbc_form(bc_liquid_gas)
        self.bc_forms[idx_bcsg] = self.create_dirichletbc_form(bc_solid_gas)

        old_N_b = new_N_b
        all_pbs.append(new_pb)


# PARAMETERS

c_up = 1.0
c_down = 0.0
V_b = 10000


temperatures = (500, 550, 600, 650, 700)

fig, axs = plt.subplots(2, 1, sharex=True)

for temperature in temperatures:
    all_pbs = []
    old_N_b = 0.000001

    # PROBLEM
    problem = Custom1DProblem()

    problem.mesh = DiscontinuousMesh1D(
        vertices1=np.linspace(0.0, 1.0, 500),
        vertices2=np.linspace(2.0, 3.0, 500),
    )

    salt = F.Material(D_0=1, E_D=0, K_S_0=2, E_K_S=0.2, solubility_law="henry")
    metal = F.Material(D_0=2, E_D=0, K_S_0=3, E_K_S=0.2, solubility_law="sievert")

    dolfinx_mesh = problem.mesh.mesh

    K_H = salt.get_solubility_coefficient(mesh=dolfinx_mesh, temperature=temperature)
    K_S = metal.get_solubility_coefficient(mesh=dolfinx_mesh, temperature=temperature)

    K_H = float(K_H)
    K_S = float(K_S)

    left = F.SurfaceSubdomain1D(id=3, x=0.0)
    right = F.SurfaceSubdomain1D(id=4, x=3.0)
    salt_air_interface = F.SurfaceSubdomain1D(id=5, x=1.0)
    air_metal_interface = F.SurfaceSubdomain1D(id=6, x=2.0)

    left_volume = F.VolumeSubdomain1D(id=1, borders=[0.0, 1.0], material=salt)
    right_volume = F.VolumeSubdomain1D(id=2, borders=[2.0, 3.0], material=metal)
    problem.subdomains = [
        left_volume,
        right_volume,
        left,
        right,
        salt_air_interface,
        air_metal_interface,
    ]

    problem.surface_to_volume = {
        salt_air_interface: left_volume,
        air_metal_interface: right_volume,
        left: left_volume,
        right: right_volume,
    }

    problem.interfaces = []

    H = F.Species("H", subdomains=problem.volume_subdomains)
    problem.species = [H]

    bubble_pressure = 0.0

    bc_liquid_gas = F.FixedConcentrationBC(
        subdomain=salt_air_interface, species=H, value=salt.K_S_0 * bubble_pressure
    )
    bc_solid_gas = F.FixedConcentrationBC(
        subdomain=air_metal_interface,
        species=H,
        value=(bubble_pressure**0.5) * metal.K_S_0 if bubble_pressure > 0.0 else 0.0,
    )
    problem.boundary_conditions = [
        F.FixedConcentrationBC(subdomain=left, species=H, value=c_up),
        F.FixedConcentrationBC(subdomain=right, species=H, value=c_down),
        bc_liquid_gas,
        bc_solid_gas,
    ]

    problem.temperature = temperature

    problem.settings = F.Settings(
        atol=1e-10,
        rtol=1e-10,
        final_time=5000,
    )

    problem.settings.stepsize = F.Stepsize(
        initial_value=0.0001,
        growth_factor=1.01,
        cutback_factor=0.9,
        target_nb_iterations=4,
    )

    flux_lg = F.SurfaceFlux(field=H, surface=salt_air_interface, filename=None)
    flux_sg = F.SurfaceFlux(field=H, surface=air_metal_interface, filename=None)
    flux_down = F.SurfaceFlux(field=H, surface=right, filename=None)

    problem.exports = [flux_lg, flux_sg, flux_down]

    problem.initialise()
    problem.run()

    times = flux_lg.t

    # axs[0].plot(times, flux_lg.data, label=f"j_liquid_gas, T={temperature}K")
    # axs[0].plot(times, flux_sg.data, label=f"j_solid_gas,  T={temperature}K")
    axs[0].plot(times, flux_down.data, label=f"j_down,  T={temperature}K")
    axs[1].plot(times, all_pbs, label=f"p_b, T={temperature}K")

axs[0].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
axs[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

axs[0].set_ylabel("Downstream flux (H/s)")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Bubble pressure (Pa)")

plt.tight_layout()
plt.show()
