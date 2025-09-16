from mods.simulation import Simulation
import matplotlib.pyplot as plt

kwargs = {
    'L': 3000,
    'dispersal_range': 90,
    'land_quality': 0.001,
    'precipitation': 0.06,
    'spawn_rate': 100,
}

sim = Simulation(**kwargs, verbose=True)

sim.initiate_uniform_radii(n=1000, r_min=0.1, r_max=0.3)
sim.run(T=10000, max_population=25_000)
sim.data_buffer.plot()
sim.state_buffer.plot()
sim.density_field_buffer.plot()
plt.show()
