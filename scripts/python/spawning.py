from mods.simulation import Simulation
import matplotlib.pyplot as plt

land_quality = 0.001
L = 7000
r_min = 0.1
r_max = 30
growth_rate = 0.1
precipitation = 0.000

kwargs = {
    'land_quality': land_quality,
    'L': L,
    'r_min': r_min,
    'r_max': r_max,
    'growth_rate': growth_rate,
    'precipitation': precipitation,
    'density_range': 100,
}

sim = Simulation(**kwargs)


# sim.initiate_uniform_radii(n=1000, r_min=r_min, r_max=r_max)
sim.spin_up(1000)
sim.run(n_iter=4000, max_population=25_000)
sim.data_buffer.plot()
sim.density_field_buffer.plot()
sim.state_buffer.plot()
plt.show()
