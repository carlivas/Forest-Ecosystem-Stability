import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from plant import Plant
from simulation import Simulation

seed = 0
np.random.seed(seed)

num_plants = 1_0000
n_iter = 25

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
    'growth_rate': 0.1 * _m,
    'dispersal_range': 100 * _m,
    'species_germination_chance': 0.2,
}

sim_kwargs = {
    'half_width': half_width,
    'half_height': half_height,
    'num_plants': num_plants,
    'kt_leafsize': 10,
    'land_quality': -0.1,
    'density_check_radius': 100 * _m,
    'density_check_resolution': 25,
}
sim = Simulation(**sim_kwargs)

d = 10
dist_min = plant_kwargs['r_min']**(1/d)
dist_max = plant_kwargs['r_max']**(1/d)
plants = [
    Plant(
        pos=np.random.uniform(-half_width, half_width, 2),
        r=np.random.uniform(dist_min, dist_max)**d,
        **plant_kwargs
    )
    for _ in range(num_plants)
]

sim.add(plants)
sim.update_kdtree()
sim.update_density_field()

save_skip = 1


def biomass(state):
    return sum([plant.area for plant in state])


states = []
density_field_arr = []
# simulation loop
try:
    for i in range(n_iter):
        sim.step()
        state = sim.get_state()
        density_field = sim.get_density_field()
        if len(state) == 0:
            break

        if i % save_skip == 0:
            states.append(state)
            density_field_arr.append(density_field.get_values())

        print(
            f'Iteration {i+1:^5}/{n_iter}. Plants left {len(state)}' + ' '*20, end='\r')
except KeyboardInterrupt:
    print('\nInterrupted by user...')

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

Q_L = sim_kwargs['land_quality']


def update(frame):
    ax[0].clear()
    ax[1].clear()

    ax[0].set_xlim(-half_width, half_width)
    ax[0].set_ylim(-half_height, half_height)
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel('Width (u)')
    ax[0].set_ylabel('Height (u)')
    for plant in states[frame]:
        circle = plt.Circle(plant.pos, plant.r, color='green',
                            fill=True, transform=ax[0].transData)
        ax[0].add_patch(circle)

    ax[1].set_xlim(-half_width, half_width)
    ax[1].set_ylim(-half_height, half_height)
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlabel('Width (u)')
    ax[1].set_ylabel('Height (u)')

    xx = np.linspace(-half_width, half_width, sim.density_field.shape[0])
    yy = np.linspace(-half_height, half_height, sim.density_field.shape[1])
    X, Y = np.meshgrid(xx, yy)
    contour = ax[1].contourf(X, Y, density_field_arr[frame].T - Q_L,
                             cmap='bwr', levels=50)

    fig.suptitle(
        f'Plant distribution and density field (Iteration {frame*save_skip}/{n_iter})')


ani = animation.FuncAnimation(fig, update, frames=len(
    states), repeat=True, interval=300)

plt.show()
print('\nDone!')
