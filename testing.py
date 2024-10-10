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


seed = np.random.randint(0, 1_000)
np.random.seed(seed)
print(f'{seed = }')

num_plants = 10_000
n_iter = 100_000

half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height

m2pp = m2_per_plant = 500  # m2/plant
_m = np.sqrt(A_bound/(m2pp*num_plants))
print(f'1 m = {_m} u')
print(f'1 u = {1/_m} m')

# Initialize simulation
plant_kwargs = {
    'r_min': 0.01 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.1 * _m,
    'reproduction_range': 100 * _m,
    'reproduction_chance': 0.2
    # 'reproduction_thresholds': (0.12, 0.125),
}

sim_kwargs = {
    'seed': seed,
    'half_width': half_width,
    'half_height': half_height,
    'num_plants': num_plants,
    'kt_leafsize': 10,
    'land_quality': -0.1,
    'density_check_radius': 100 * _m,
    'density_check_resolution': 100,
}

sim = Simulation(**sim_kwargs)

d = 2
dist_min = plant_kwargs['r_min']**d
dist_max = plant_kwargs['r_max']**d
plants = [
    Plant(
        pos=np.random.uniform(-half_width, half_width, 2),
        # r=sample_radii_uniform_area(
        #     plant_kwargs['r_min'], plant_kwargs['r_max']),
        # r=np.random.uniform(plant_kwargs['r_min'], plant_kwargs['r_max']),
        # r=plant_kwargs['r_min'],
        r=np.random.uniform(dist_min, dist_max)**(1/d),
        **plant_kwargs
    )
    for _ in range(num_plants)
]


def biomass(state):
    return sum([plant.area for plant in state])


ΔQ_L = -1e-5

states = []
density_field_arr = []
biomass_arr = []
num_arr = []
land_quality_arr = []

sim.add(plants)
sim.update_kdtree()
sim.update_density_field()
state = sim.get_state()
density_field = sim.get_density_field()
l = len(state)
b = biomass(state)

states.append(state)
density_field_arr.append(density_field.get_values())
biomass_arr.append(b)
num_arr.append(l)
land_quality_arr.append(sim.land_quality)

save_skip = 100

# simulation loop
try:
    i = 1
    while i < n_iter and l > 0:
        Q_L = sim.land_quality + ΔQ_L
        sim.land_quality = Q_L
        sim.step()
        state = sim.get_state()
        density_field = sim.get_density_field()
        l = len(state)
        b = biomass(state)

        if i % save_skip == 0:
            states.append(state)
            density_field_arr.append(density_field.get_values())

        biomass_arr.append(b)
        num_arr.append(l)

        land_quality_arr.append(Q_L)

        string = f'Iter {i+1:>5}  |  (P, B): ({l:>6}, {np.round(b, 4):>6})'
        Δ_num = num_arr[-1] - num_arr[-2]
        Δ_bm = biomass_arr[-1] - biomass_arr[-2]
        string += f'  |  Δ = ({Δ_num:>4}, {np.round(Δ_bm, 4):>6})'
        print(string + f'  |  Q_L = {np.round(Q_L, 3):>6}' + ' '*20, end='\r')
        i += 1
        if l == 0:
            break
except KeyboardInterrupt:
    print('\nInterrupted by user...')

fig, ax = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
# ρ = 500  # kg/m3
# biomass_arr = np.array(biomass_arr) * 1*_m / (ρ * _m**3)
ax[0].plot(biomass_arr, label='Biomass', color='blue')
# ax[0].set_title('Biomass and density over time', fontsize=10)
# ax[0].set_xlabel('Iteration')
# ax[0].set_ylabel('Biomass', fontsize=8)  # [$\mathrm{kg}$]
ax[0].legend(fontsize=8)

# num_arr = np.array(num_arr)/(A_bound/_m**2)
ax[1].plot(num_arr, label='Density', color='red')
# ax[1].set_title('Density over time')
# ax[1].set_xlabel('Iteration')
# ax[1].set_ylabel('Density', fontsize=8)  # [plants/$\mathrm{m}^2$]
ax[1].legend(fontsize=8)

ax[2].plot(land_quality_arr, label='Land quality', color='green')
# ax[2].set_title('Land Quality over time')
ax[2].set_xlabel('Iteration', fontsize=8)
# ax[2].set_ylabel('Land Quality', fontsize=8)
ax[2].legend(fontsize=8)

fig.tight_layout()
fig.subplots_adjust(hspace=0)
plt.show()


# plt.figure()
# plt.title('log(Density) over log(Biomass)')
# plt.plot(num_arr, biomass_arr)
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('Density')
# plt.ylabel('Biomass')
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
Q_L = sim_kwargs['land_quality']


def update(frame):
    print(f'Rendering frame {frame+1}/{len(states)}...', end='\r')

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

    # ax[1].set_xlabel('Size (u)')
    # ax[1].set_ylabel('Frequency')
    # ax[1].set_xlim(0, 1.2*plant_kwargs['r_max'])
    # ax[1].set_ylim(0, num_plants)

    # bins = np.linspace(0, 1.2*plant_kwargs['r_max'], 25)
    # size = np.array(sizes)
    # ax[1].hist(sizes, bins=bins, color='black')

    # fig.suptitle(f'Plant and size distribution (iteration {frame*save_skip})')

    ax[1].set_xlim(-half_width, half_width)
    ax[1].set_ylim(-half_height, half_height)
    ax[1].set_aspect('equal', 'box')
    ax[1].set_xlabel('Width (u)')
    ax[1].set_ylabel('Height (u)')

    xx = np.linspace(-half_width, half_width, sim.density_field.shape[0])
    yy = np.linspace(-half_height, half_height, sim.density_field.shape[1])
    X, Y = np.meshgrid(xx, yy)
    contour = ax[1].contourf(X, Y, density_field_arr[frame].T + Q_L,
                             cmap='viridis', levels=50)

    fig.suptitle(
        f'Plant distribution and reproduction field (Iteration {frame*save_skip})')


ani = animation.FuncAnimation(fig, update, frames=len(
    states), repeat=True, interval=300)


print('\nSaving animation...')
ani.save('plant_simulation.mp4', writer='ffmpeg', fps=2)
print('\nDone!')
