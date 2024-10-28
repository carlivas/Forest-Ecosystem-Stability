import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from plant import Plant
from simulation import Simulation
import json
import os

num_plants = 100
n_iter = 200
m2pp = m2_per_plant = 10_000  # m2/plant


half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height
_m = np.sqrt(A_bound/(m2pp*num_plants))

print(f'{_m = }')

sgcs = [0.0, 0.01, 0.03, 0.05, 0.07, 0.1]
lqs = [-0.001, 0.0, 0.001]

k = 0
# seed = np.random.randint(0, 1_000)
seed = 3
np.random.seed(seed)

# Initialize simulation
plant_kwargs = {
    'r_min': 0.1 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.5 * _m,
    'dispersal_range': 100 * _m,
    'species_germination_chance': sgcs[0],
}
sim_kwargs = {
    'seed': seed,
    'n_iter': n_iter,
    'half_width': half_width,
    'half_height': half_height,

    'm2_per_plant': m2_per_plant,
    '_m': _m,
    'num_plants': num_plants,
    'land_quality': lqs[0],

    'kt_leafsize': 10,

    # 'density_field_resolution': min(max(10, int(np.sqrt(1/_m))), 200),
    'density_field_resolution': 50,
    'density_check_radius': 100*_m,

    'density_field_buffer_size': 10,
    'density_field_buffer_skip': 10,
    'density_field_buffer_preset_times': [0, 10, 22, 23, 24],

    'state_buffer_size': 10,
    'state_buffer_skip': 10,
    'state_buffer_preset_times': [0, 10, 22, 23, 24],
}


print(f'density_check_radius = {sim_kwargs["density_check_radius"]}')

print(
    f'density_field_resolution = {sim_kwargs["density_field_resolution"]}')

sim = Simulation(**sim_kwargs)
sim.initiate_uniform_lifetimes(
    n=num_plants, t_min=1, t_max=300, **plant_kwargs)

sim.run(n_iter=n_iter)

sim.state_buffer.plot()

sim.density_field_buffer.plot(size=2)

plt.show()
