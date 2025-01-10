import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_m2pp, _m_from_domain_sides
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable

save_results = True
plot_results = True

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)

num_plants = 3000
T = 1000
kwargs = {
    'seed': seed,
    'L': 4500,
    'T': T,
    'dispersal_range': 90,
    'precipitation': 7000e-5,
    'spawn_rate': 1,
    'growth_rate': 0.1,

    'time_step': 1,
    
    'density_field_resolution': 100,

    'buffer_size': 21,
    'buffer_skip': T//20,
    
    'verbose': True,
}

save_folder = f'Data/temp/kwarg_cleanup'

sim = Simulation(**kwargs)
sim.initiate_uniform_radii(n=num_plants, r_min=0.1, r_max=30)

print(f'\nSimulation initiated. Time: {time.strftime("%H:%M:%S")}')

sim.run(T=T)



sim.print_dict()

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
    
if save_results:
    # surfix = time.strftime("%Y%m%d-%H%M%S")
    surfix = 'test'
    sim.save_dict(path = f'{save_folder}/kwargs_{surfix}')
    sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
    sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
    sim.density_field_buffer.save(
        f'{save_folder}/density_field_buffer_{surfix}')

    print('Data saved in folder:', save_folder)

