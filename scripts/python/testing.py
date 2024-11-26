import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable

save_folder = r'Data\init_density_experiment_SPH\ensemble'
save_results = True
plot_results = False

num_plantss = [500, 1_000, 1_500]
n_iter = 10_000
L = 30_00  # m
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)

n_searches = 30
for i in range(n_searches):
    lq = np.random.normal(0.0, 0.5)
    sgc = np.random.rand() * 0.05
    for n in num_plantss:
        dens0 = n / L**2
        print(f'\n{i+1} / {n_searches} | {lq=:.2f}, {sgc=:.2f}')
        print(f'{n=} plants, {dens0=} plants/m^2')

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

            'dens0': dens0,
            '_m': _m,
            'num_plants': n,
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
            n=n, t_min=1, t_max=300, **plant_kwargs)
        np.random.seed(np.random.randint(0, 1_000_000))
        print(f'\nSimulation initiated. Time: {time.strftime("%H:%M:%S")}')
        sim.run(n_iter=n_iter, max_population=15_000)
        if sim.t < 600:
            continue

        np.random.seed(seed)

        print_nested_dict(combined_kwargs)

        if save_results:
            surfix = time.strftime("%Y%m%d-%H%M%S")
            save_kwargs(combined_kwargs, f'{save_folder}/kwargs_{surfix}')
            sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
            sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
            sim.density_field_buffer.save(
                f'{save_folder}/density_field_buffer_{surfix}')

            print('Data saved.')

        if plot_results:
            print('Plotting...')
            surfix = time.strftime("%Y%m%d-%H%M%S")

            sim.state_buffer.plot(title=f'state_buffer_{surfix}, L = {L} m')
            sim.density_field_buffer.plot(
                title=f'density_field_buffer_{surfix}, L = {L} m')
            sim.data_buffer.plot(title=f'data_buffer_{surfix}, L = {L} m')
            plt.show()
