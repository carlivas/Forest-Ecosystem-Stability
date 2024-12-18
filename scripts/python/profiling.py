import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import cProfile
import pstats

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_m2pp, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable

save_folder = f'Data\\temp'
save_results = False
plot_results = True

num_plants = 2000

kwargs = {
    'L': 3000,
    'dispersal_range': 90,
    'land_quality': 0.001,
    'precipitation': 0.06,
    'spawn_rate': 1,

    'buffer_size': 20,
    'buffer_skip': 10
}

sim = Simulation(**kwargs)
sim.initiate_uniform_radii(n=num_plants, r_min=0.1, r_max=30)
print(f'\nSimulation initiated. Time: {time.strftime("%H:%M:%S")}')


# Profile the simulation run
profiler = cProfile.Profile()
profiler.enable()

sim.run(n_iter=200)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime').print_stats()

if save_results:
    # surfix = time.strftime("%Y%m%d-%H%M%S")
    surfix = 'profiling'
    save_kwargs(combined_kwargs, f'{save_folder}/kwargs_{surfix}')
    sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
    sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
    sim.density_field_buffer.save(
        f'{save_folder}/density_field_buffer_{surfix}')

    print('Data saved in folder:', save_folder)

if plot_results:
    print_nested_dict(kwargs)
    print('Plotting...')
    sim.state_buffer.plot()
    sim.density_field_buffer.plot()
    sim.data_buffer.plot()
    plt.show()
