import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from plant import Plant
from simulation import Simulation

seed = 0
np.random.seed(seed)

half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height

# Initialize simulation
plant_kwargs = {
    'r_min': 0.1,
    'r_max': 1,
    'growth_rate': 0.005,
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

plants = [Plant(np.array([0., 0.]), **plant_kwargs),
          Plant(np.array([0.21, 0.]), **plant_kwargs)]

sim.add(plants)
sim.update_kdtree()

states = [sim.state()]
n_iter = 20
for i in range(n_iter):
    sim.step()
    states.append(sim.state())

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-half_width, half_width)
ax.set_ylim(-half_height, half_height)
ax.set_aspect('equal', 'box')
ax.set_xlabel('Width (u)')
ax.set_ylabel('Height (u)')
ax.set_title('Plant distribution')

circles = []


def update(frame):
    for circle in circles:
        circle.remove()
    circles.clear()

    state = states[frame]
    for plant in state:
        if plant.is_colliding:
            c = 'red'
        else:
            c = 'green'
        circle = plt.Circle(plant.pos, plant.r, color=c,
                            fill=False, transform=ax.transData)
        ax.add_patch(circle)
        circles.append(circle)
    return circles


ani = FuncAnimation(fig, update, frames=len(states),
                    blit=True, repeat=True, interval=100)

plt.show()
print('\nDone!')
