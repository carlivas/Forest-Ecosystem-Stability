import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_m2pp, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable

save_results = True
plot_results = True

num_plants = 2000

kwargs = {
    'L': 3000,
    'dispersal_range': 90,
    'precipitation': 0.06,
    'spawn_rate': 1,
    'growth_rate': 0.1,

    'time_step': 0.1,

    'buffer_size': 40,
    'buffer_skip': 10
}
save_folder = f'Data\\temp\\time_step{kwargs["time_step"]}'

sim = Simulation(verbose=True, **kwargs,)
sim.initiate_uniform_radii(n=num_plants, r_min=0.1, r_max=30)

print(f'\nSimulation initiated. Time: {time.strftime("%H:%M:%S")}')

sim.run(T=400)

print_nested_dict(sim.kwargs)

if save_results:
    surfix = time.strftime("%Y%m%d-%H%M%S")
    save_kwargs(sim.kwargs, f'{save_folder}/kwargs_{surfix}')
    sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
    sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
    sim.density_field_buffer.save(
        f'{save_folder}/density_field_buffer_{surfix}')

    print('Data saved in folder:', save_folder)

if plot_results:
    print('Plotting...')
    L = kwargs['L']
    N0 = num_plants
    title = f'{L =}, $N_0$ = {N0}'
    sim.state_buffer.plot(title=f'{title}')
    sim.density_field_buffer.plot(
        title=f'{title}')
    sim.data_buffer.plot(title=f'{title}')
    plt.show()
