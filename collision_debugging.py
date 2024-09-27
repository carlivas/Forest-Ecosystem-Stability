import matplotlib.pyplot as plt
import numpy as np

from plant import Plant
from simulation import Simulation


seed = 0
np.random.seed(seed)

half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height

# Initialize simulation
plant_kwargs = {
    'r_min': 0.001,
    'r_max': 1,
    'growth_rate': 0.01,
    'reproduction_range': 0.1,
    'reproduction_chance': 0.025,
}

sim_kwargs = {
    'half_width': half_width,
    'half_height': half_height,
    'num_plants': 1,
    'kt_leafsize': 10,
    'land_quality': -0.1,
    'density_check_radius': 0.3
}

sim = Simulation(**sim_kwargs)


plants = [Plant(np.array([0, 0]), **plant_kwargs),
          Plant(np.array([0.21, 0]), **plant_kwargs)]

sim.add(plants)
sim.update_kdtree()

states = [sim.plants.copy()]
n_iter = 10
for i in range(n_iter):
    sim.step()
    states.append(sim.plants.copy())

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
for i, state in enumerate([states[0], states[n_iter//2], states[-1]]):
    print(i, len(state))
    ax[i] = plt.gca()
    ax[i].set_xlim(-half_width, half_width)
    ax[i].set_ylim(-half_height, half_height)
    ax[i].set_aspect('equal', 'box')
    ax[i].set_xlabel('Width (u)')
    ax[i].set_ylabel('Height (u)')
    ax[i].set_title(f'Iteration {i*n_iter//2}')
    fig.suptitle('Plant distribution')

    for plant in state:
        circle = plt.Circle(plant.pos, plant.r, color='green',
                            fill=False, transform=ax[i].transData)
        ax[i].add_patch(circle)


plt.show()
print('\nDone!')
