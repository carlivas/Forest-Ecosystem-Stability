import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import cProfile

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_m2pp, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable

save_folder = f'Data\\temp'
save_results = True
plot_results = False

num_plants = 1000
n_iter = 1000
L = 10_000
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)


m2pp = L**2 / num_plants
print(f'{num_plants=}')

lq = 0.1
sgc = 0.03

seed = 0
np.random.seed(seed)
# Initialize simulation
plant_kwargs = {
    'r_min': 0.1 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.1 * _m,
    'dispersal_range': 100 * _m,
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

    'm2_per_plant': m2pp,
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

sim = Simulation(**sim_kwargs)
sim.initiate_uniform_lifetimes(
    n=num_plants, t_min=1, t_max=300, **plant_kwargs)

print(f'\nSimulation initiated. Time: {time.strftime("%H:%M:%S")}')


def main():
    sim.run(n_iter=n_iter)


if __name__ == '__main__':
    cProfile.run('main()')
