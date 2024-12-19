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

num_plants = int(sys.argv[2])
precipitation = float(sys.argv[3])

T = 20000

# Initialize simulation
kwargs = {
    'T': T,
    'L': 6000,
    'precipitation': precipitation,
    'spawn_rate': 1,
    
    'buffer_size': 400,
    'buffer_skip': 50,
}


print(f'\ntesting_bash.py: Simulation initiating...')
print(f'Time: {time.strftime("%H:%M:%S")}')
sim = Simulation(**kwargs)
sim.initiate_uniform_radii(n=num_plants, r_min=0.1, r_max=30)

sim.run(T=T)

print_nested_dict(kwargs)

if save_results:
    time.sleep(1)
    surfix = time.strftime("%Y%m%d-%H%M%S")
    save_kwargs(kwargs, f'{save_folder}/kwargs_{surfix}')
    sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
    sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
    sim.density_field_buffer.save(
        f'{save_folder}/density_field_buffer_{surfix}')

    print('Data saved in folder:', save_folder)

if plot_results:
    print('Plotting...')
    time = time.strftime("%Y%m%d-%H%M%S")
    title = f'(n={num_plants}, precipitation={precipitation})'
    sim.state_buffer.plot(title=f'{title}')
    sim.density_field_buffer.plot(
        title=f'{title}')
    sim.data_buffer.plot(title=f'{title}')
    plt.show()
