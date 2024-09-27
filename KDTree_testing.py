import matplotlib.pyplot as plt
import numpy as np

from KDTree_plant import Plant
from KDTree_simulation import Simulation

import rendering as rendering


seed = 0
np.random.seed(seed)

num_plants = 10_000
n_iter = 10_000

half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height

m2pp = m2_per_plant = 1_000  # m2/plant
_m = np.sqrt(A_bound/(m2pp*num_plants))
print(f'1 m = {_m} u')
print(f'1 u = {1/_m} m')

# Initialize simulation
plant_kwargs = {
    'r_min': 0.01 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.01 * _m,
    'reproduction_range': 100 * _m,
    'reproduction_chance': 0.035,
}

sim_kwargs = {
    'half_width': half_width,
    'half_height': half_height,
    'num_plants': num_plants,
    'kt_leafsize': 10,
    'land_quality': -0.1,
    'density_check_radius': 300 * _m
}

sim = Simulation(**sim_kwargs)


plants = [
    Plant(
        np.random.uniform(-half_width, half_width, 2),
        r=np.random.uniform(plant_kwargs['r_min'], plant_kwargs['r_max']),
        **plant_kwargs
    )
    for _ in range(num_plants)
]

sim.add(plants)
sim.update_kdtree()


def biomass(sim):
    return sum(plant.area for plant in sim.plants)


states = []
biomass_arr = []
num_arr = []
# simulation loop
try:
    for i in range(n_iter):
        sim.step()
        l = len(sim.plants)
        if l == 0:
            break

        biomass_arr.append(biomass(sim))
        num_arr.append(l)

        if i > 10 and i % 10:
            # growth in the last 10 iterations
            delta_plants = num_arr[-1] - num_arr[-10]
            print(
                f'Iteration {i+1:^5}/{n_iter}  |   Plants left {l:>5}  |   Delta (10 iter) {delta_plants}.' + ' '*20, end='\r')
        if i % 1000 == 0:
            states.append(sim.plants.copy())
except KeyboardInterrupt:
    print('\nInterrupted by user...')


plt.figure()
# ρ = 500  # kg/m3
# biomass_arr = np.array(biomass_arr) * 1*_m / (ρ * _m**3)
plt.plot(biomass_arr)
plt.title('Biomass over time')
plt.xlabel('Iteration')
plt.ylabel('Biomass')  # [$\mathrm{kg}$]')
plt.show()

plt.figure()
# num_arr = np.array(num_arr)/(A_bound/_m**2)
plt.plot(num_arr)
plt.title('Density over time')
plt.xlabel('Iteration')
plt.ylabel('Density')  # [plants/$\mathrm{m}^2$]')
plt.show()

for state in states:
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_height, half_height)
    for plant in state:
        circle = plt.Circle(plant.pos, plant.r, color='green',
                            fill=True, transform=ax.transData)
        ax.add_patch(circle)

    ax.set_xlim(-half_width, half_width)
    ax.set_ylim(-half_height, half_height)
    ax.set_aspect('equal', 'box')
    plt.title('Plant distribution')
    plt.xlabel('Width (u)')
    plt.ylabel('Height (u)')

plt.show()
print('\nDone!')
