import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_domain_sides

num_plants = 3
n_iter = 1
L = 3000
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)
dens0 = num_plants / L**2

seed = 42
np.random.seed(seed)

sim_kwargs = {
    'seed': seed,
    'n_iter': n_iter,

    'dens0': dens0,
    '_m': _m,
    'num_plants': num_plants,
    'land_quality': 0.01,

    'density_check_radius': 100 * _m,
    'buffer_size': 3,
}

plant_kwargs = {
    'r_min': 0.1 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.1 * _m,
    'dispersal_range': 90 * _m,
    'species_germination_chance': 0.004,
}

sim = Simulation(**sim_kwargs)

# max_densities = []
# int_densities = []
# sum_areas = []

# n_dists = 3
# Xs = np.array([[0.2, 0.2], [0.5, 0.5], [0.8, 0.8]]) - 0.5
# for i in range(n_dists):
#     X = Xs[i]
#     positions = np.random.normal(X, i/50, size=(num_plants//n_dists, 2))
#     for pos in positions:
#         r = 30 * (1 - i/n_dists) * _m
#         print(f'\nA = {np.pi*r**2}')
#         plant = Plant(pos, r, **plant_kwargs)
#         sim.add(plant)

#     # Update necessary data structures
#     sim.update_kdtree()
#     sim.density_field.update()
#     max_densities.append(float(sim.density_field.values.max()))
#     int_densities.append(
#         float(sim.density_field.values.sum() * (sim.density_field.xx[1] - sim.density_field.xx[0])**2))
#     sum_areas.append(
#         float(np.array([plant.area for plant in sim.state]).sum()))

#     data = sim.data_buffer.analyze_and_add(sim.state, t=i)
#     sim.biomass = data[0]
#     sim.population = data[1]

#     sim.state_buffer.add(state=sim.get_state(), t=i)
#     sim.density_field_buffer.add(field=sim.density_field.get_values(), t=i)
#     sim.density_field.plot()

# print(f'{max_densities=}')
# print(f'{int_densities=}')
# print(f'{sum_areas=}')

sim.initiate_packed_distribution(r=10*_m, **plant_kwargs)

max_density = float(sim.density_field.values.max())
int_density = float(sim.density_field.values.sum() *
                    (sim.density_field.xx[1] - sim.density_field.xx[0])**2)
sum_area = float(np.array([plant.area for plant in sim.state]).sum())

print(f'{max_density=}')
print(f'{int_density=}')
print(f'{sum_area=}')

sim.density_field_buffer.plot()
sim.state_buffer.plot()

plt.show()
