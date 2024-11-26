import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable

save_folder = f'Data\\temp'
save_results = True
plot_results = True

# num_plantss = [1_000, 1_500, 2_000, 2_500]p[::-1]
num_plantss = [2000]
n_iter = 10_000
L = 10_000
half_width = half_height = 0.5
_m = _m_from_domain_sides(L, S_bound=2*half_width)


for n in num_plantss:
    dens0 = n / L**2
    print(f'{n=} plants')

    lq = 0.1
    sgc = 0.04

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
    sim.run(n_iter=n_iter)

    np.random.seed(seed)
    sim.data_buffer.finalize()

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

        print_nested_dict(combined_kwargs)
        sim.state_buffer.plot(title=f'state_buffer_{surfix}, L = {L} m')
        sim.density_field_buffer.plot(
            title=f'density_field_buffer_{surfix}, L = {L} m')
        sim.data_buffer.plot(title=f'data_buffer_{surfix}, L = {L} m')
        plt.show()
