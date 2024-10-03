import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from plant import Plant
from simulation import Simulation


def sample_radii_uniform_area(r_min, r_max, n=1):
    # Calculate the corresponding area range
    A_min = np.pi * r_min**2
    A_max = np.pi * r_max**2

    # Sample uniformly in the area range
    areas = np.random.uniform(A_min, A_max, n)

    # Convert areas to radii
    radii = np.sqrt(areas / np.pi)

    if n == 1:
        return radii[0]
    return radii


seed = 0
np.random.seed(seed)

num_plants = 1_000
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
    'growth_rate': 0.1 * _m,
    'reproduction_range': 100 * _m,
    'reproduction_chance': 0.8,
}

sim_kwargs = {
    'half_width': half_width,
    'half_height': half_height,
    'num_plants': num_plants,
    'kt_leafsize': 10,
    'land_quality': -0.2,
    'density_check_radius': 300 * _m
}

sim = Simulation(**sim_kwargs)

d = 10
dist_min = plant_kwargs['r_min']**(1/d)
dist_max = plant_kwargs['r_max']**(1/d)
plants = [
    Plant(
        pos=np.random.uniform(-half_width, half_width, 2),
        # r=sample_radii_uniform_area(
        #     plant_kwargs['r_min'], plant_kwargs['r_max']),
        # r=np.random.uniform(plant_kwargs['r_min'], plant_kwargs['r_max']),
        r=np.random.uniform(dist_min, dist_max)**d,
        # r=plant_kwargs['r_min'],
        **plant_kwargs
    )
    for _ in range(num_plants)
]

sim.add(plants)
sim.update_kdtree()

save_skip = 25


def biomass(state):
    return sum([plant.area for plant in state])


states = []
biomass_arr = []
num_arr = []
# simulation loop
try:
    for i in range(n_iter):
        sim.step()
        state = sim.state()
        l = len(state)

        if l == 0:
            break

        if i % save_skip == 0:
            states.append(state)

        biomass_arr.append(biomass(state))
        num_arr.append(l)

        if i > 1:
            delta_plants = num_arr[-1] - num_arr[-2]
            print(
                f'Iteration {i+1:^5}/{n_iter}  |   Plants left {l:>5}  |   Delta (last iter) {delta_plants}.' + ' '*20, end='\r')
except KeyboardInterrupt:
    print('\nInterrupted by user...')

plt.figure()
# ρ = 500  # kg/m3
# biomass_arr = np.array(biomass_arr) * 1*_m / (ρ * _m**3)
plt.plot(biomass_arr)
plt.title('Biomass over time')
plt.ylim(0, max(biomass_arr)*1.1)
plt.xlabel('Iteration')
plt.ylabel('Biomass')  # [$\mathrm{kg}$]')
plt.show()

plt.figure()
# num_arr = np.array(num_arr)/(A_bound/_m**2)
plt.plot(num_arr)
plt.title('Density over time')
plt.ylim(0, num_plants)
plt.xlabel('Iteration')
plt.ylabel('Density')  # [plants/$\mathrm{m}^2$]')
plt.show()

plt.figure()
plt.title('log(Density) over log(Biomass)')
plt.plot(num_arr, biomass_arr)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Density')
plt.ylabel('Biomass')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(pad=3.0)


def update(frame):
    ax[0].clear()
    ax[1].clear()

    sizes = []

    ax[0].set_xlim(-half_width, half_width)
    ax[0].set_ylim(-half_height, half_height)
    ax[0].set_aspect('equal', 'box')
    ax[0].set_xlabel('Width (u)')
    ax[0].set_ylabel('Height (u)')
    for plant in states[frame]:
        sizes.append(plant.r)
        circle = plt.Circle(plant.pos, plant.r, color='green',
                            fill=True, transform=ax[0].transData)
        ax[0].add_patch(circle)

    ax[1].set_xlabel('Size (u)')
    ax[1].set_ylabel('Frequency')
    ax[1].set_xlim(0, 1.2*plant_kwargs['r_max'])
    ax[1].set_ylim(0, num_plants)

    bins = np.linspace(0, 1.2*plant_kwargs['r_max'], 25)
    size = np.array(sizes)
    ax[1].hist(sizes, bins=bins, color='black')

    fig.suptitle(f'Plant and size distribution (iteration {frame*save_skip})')


ani = animation.FuncAnimation(fig, update, frames=len(
    states), repeat=True, interval=300)

plt.show()
print('\nDone!')
