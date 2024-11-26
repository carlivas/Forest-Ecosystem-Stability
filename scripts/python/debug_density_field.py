import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_domain_sides

num_plants = 200
n_iter = 2500
L = 40_00
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)
dens0 = num_plants / L**2

seed = np.random.randint(0, 1000)
np.random.seed(seed)

sim_kwargs = {
    'seed': seed,
    'n_iter': n_iter,

    'dens0': dens0,
    '_m': _m,
    'num_plants': num_plants,
    'land_quality': 0.01,

    'density_check_radius': 100 * _m,
    'buffer_size': 25,
}

plant_kwargs = {
    'r_min': 0.1 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.1 * _m,
    'dispersal_range': 100 * _m,
    'species_germination_chance': 0.004,
}

sim = Simulation(**sim_kwargs)

sim.initiate_uniform_lifetimes(
    n=num_plants, t_min=1, t_max=300, **plant_kwargs)

for i in range(n_iter//250):
    sim.run(n_iter=250)

    sim.state_buffer.plot(title=f'state_buffe, {seed=}')
    sim.density_field_buffer.plot(title=f'density_field_buffer')
    # sim.data_buffer.plot(title=f'data_buffer')
    plt.show()
