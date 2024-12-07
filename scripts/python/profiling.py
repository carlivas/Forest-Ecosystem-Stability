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
L = 3000
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)

precipitation = 9000e-5
def precipitation_func(t): return precipitation


precipitation_func.__name__ = str(precipitation)

seed = np.random.randint(0, 1_000_000)
np.random.seed(seed)
# Initialize simulation
plant_kwargs = {
    'r_min': 0.1 * _m,
    'r_max': 30 * _m,
    'growth_rate': 0.1 * _m,
    'dispersal_range': 30 * _m,
    'species_germination_chance': 1.,
}

sim_kwargs = {
    'seed': seed,
    'n_iter': n_iter,
    'half_width': half_width,
    'half_height': half_height,

    'dens0': num_plants / L**2,
    '_m': _m,
    'num_plants': num_plants,
    'land_quality': 0.01,

    'precipitation_func': precipitation_func,
    'buffer_size': n_iter,
    'buffer_skip': 1
}
combined_kwargs = {
    'plant_kwargs': plant_kwargs,
    'sim_kwargs': sim_kwargs
}

sim = Simulation(**sim_kwargs)
sim.initiate_uniform_lifetimes(
    n=num_plants, t_min=1, t_max=300, **plant_kwargs)

print(f'\nSimulation initiated. Time: {time.strftime("%H:%M:%S")}')


def main():
    sim.run(n_iter=n_iter)
    print_nested_dict(combined_kwargs)

    if save_results:
        # surfix = time.strftime("%Y%m%d-%H%M%S")
        surfix = 'temp'
        save_kwargs(combined_kwargs, f'{save_folder}/kwargs_{surfix}')
        sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
        sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
        sim.density_field_buffer.save(
            f'{save_folder}/density_field_buffer_{surfix}')

        print('Data saved in folder:', save_folder)

    if plot_results:
        print('Plotting...')
        title = f'(lq={sim_kwargs['land_quality']:.3e},   sg={plant_kwargs['species_germination_chance']:.3e},   dispersal_range={
            (plant_kwargs['dispersal_range']):.3e})'
        sim.state_buffer.plot(title=f'{title}')
        sim.density_field_buffer.plot(
            title=f'{title}')
        sim.data_buffer.plot(title=f'{title}')
        plt.show()


if __name__ == '__main__':
    cProfile.run('main()')
