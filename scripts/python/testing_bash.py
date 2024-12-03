import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import sys
import os

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable, scientific_notation_parser

save_folder = os.path.abspath(sys.argv[1])
os.makedirs(save_folder, exist_ok=True)
print(f'testing.sh: Data will be saved in: {save_folder}')

save_results = True
plot_results = False

L = int(sys.argv[2])  # m
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)
dispersal_range = 90  # m

num_plants = int(sys.argv[3])
n_iter = int(sys.argv[4])
lq = scientific_notation_parser(sys.argv[5])
sgc = scientific_notation_parser(sys.argv[6])

dens0 = num_plants / L**2

seed = np.random.randint(0, 1_000_000)
np.random.seed(seed)
# Initialize simulation
plant_kwargs = {
    'r_min': 0.1 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.1 * _m,
    'dispersal_range': dispersal_range * _m,
    'species_germination_chance': sgc,
    'is_dead': False,
    'is_colliding': False,
    'generation': 0,
}

buffer_size = 15
buffer_skip = 10
buffer_preset_times = np.linspace(0, n_iter, buffer_size).astype(int)

sim_kwargs = {
    'seed': seed,
    'n_iter': n_iter,
    'half_width': half_width,
    'half_height': half_height,

    'dens0': dens0,
    '_m': _m,
    'num_plants': num_plants,
    'land_quality': lq,

    'kt_leafsize': 10,

    'density_check_radius': 100 * _m,
    'density_field_resolution': 100,

    'density_field_buffer_size': buffer_size,
    'density_field_buffer_skip': buffer_skip,
    'density_field_buffer_preset_times': buffer_preset_times,

    'state_buffer_size': buffer_size,
    'state_buffer_skip': buffer_skip,
    'state_buffer_preset_times': buffer_preset_times,
}
combined_kwargs = {
    'plant_kwargs': plant_kwargs,
    'sim_kwargs': sim_kwargs
}

# print_nested_dict(combined_kwargs)

sim = Simulation(**sim_kwargs)

sim.initiate_uniform_lifetimes(
    n=num_plants, t_min=1, t_max=300, **plant_kwargs)
# print(f'testing_bash.py: INITIATING DENSE DISTRIBUTION')
# sim.initiate_dense_distribution(n=num_plants, **plant_kwargs)
np.random.seed(np.random.randint(0, 1_000_000))
print()
print(f'testing_bash.py: Simulation initiated. Time: {
      time.strftime("%H:%M:%S")}')
sim.run(n_iter=n_iter, max_population=25_000)

if sim.t > 300:
    np.random.seed(seed)

    print_nested_dict(combined_kwargs)

    if save_results:
        surfix = time.strftime("%Y%m%d-%H%M%S")
        save_kwargs(combined_kwargs, f'{save_folder}/kwargs_{surfix}')
        sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
        sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
        sim.density_field_buffer.save(
            f'{save_folder}/density_field_buffer_{surfix}')

        print('Data saved in folder:', save_folder)

    if plot_results:
        print('Plotting...')
        time = time.strftime("%Y%m%d-%H%M%S")
        title = f'(lq={lq:.3e},   sg={sg:.3e},   dispersal_range={
            (dispersal_range):.3e})'
        sim.state_buffer.plot(title=f'{title}')
        # sim.density_field_buffer.plot(
        #     title=f'{title}')
        sim.data_buffer.plot(title=f'{title}')
        plt.show()
