import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer
from mods.utilities import print_nested_dict

path = sys.argv[1]
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs' in f][::-1]
print(f'sim_nums: {sim_nums}')

print_kwargs = bool(int(sys.argv[2]))
plot_data = bool(int(sys.argv[3]))
plot_states = bool(int(sys.argv[4]))
plot_density_field = bool(int(sys.argv[5]))

detailed_plot = not bool(int(sys.argv[6]))

prev_mod = 0
p = 0
for i, n in enumerate(sim_nums):
    print(f'\nplotting.py: sim {i+1} / {len(sim_nums)}')
    print(f'plotting.py: Loading sim {n}...')

    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{n}.json', typ='series').to_dict()
    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv')
    num_plants = data_buffer_arr['Population'].iloc[0]

    # if num_plants < 1000 or num_plants > 2000:
    #     print(f'plotting.py: Skipping sim {n} due to low population...')
    #     continue

    L = kwargs.get('L', -1)

    subpath = path.replace('Data\\', '')
    title = f'{subpath}\\{n}   ($N_0$={num_plants:.0f}, L={L} m)'

    if print_kwargs:
        print('plotting.py: Loaded kwargs, now printing...')
        print()
        print_nested_dict(kwargs)
        print()

    if plot_data:
        data_buffer = DataBuffer(data=data_buffer_arr)
        print('plotting.py: Loaded data_buffer, now plotting...')
        data_buffer.plot(title=title)

    if plot_states:
        state_buffer_arr = pd.read_csv(
            f'{load_folder}/state_buffer_{n}.csv', header=None)
        state_buffer = StateBuffer(
            data=state_buffer_arr, kwargs=kwargs)
        print('plotting.py: Loaded state_buffer, now plotting...')
        state_buffer.plot(size=2, title=title, fast=detailed_plot)

    if plot_density_field:
        density_field_buffer_arr = pd.read_csv(
            f'{load_folder}/density_field_buffer_{n}.csv', header=None)
        density_field_buffer = FieldBuffer(
            data=density_field_buffer_arr)
        print('plotting.py: Loaded density_field_buffer, now plotting...')

        density_field_buffer.plot(
            size=2, title=title)

    p += int(plot_data) + \
        int(plot_states) + int(plot_density_field)
    mod = p % 8
    if mod < prev_mod or i >= len(sim_nums) - 1:
        plt.show()

    prev_mod = mod

print('plotting.py: Done.\n')
