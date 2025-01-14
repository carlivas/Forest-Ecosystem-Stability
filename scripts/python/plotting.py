import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer, HistogramBuffer
from mods.utilities import print_nested_dict

path = r'Data\temp\memory_test'
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

surfixes = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs' in f][::-1]
print(f'surfixes: {surfixes}')

print_kwargs = 0
plot_data = 0
plot_states = 1
plot_density_field = 0
plot_histograms = 0

fast_plot = 0

prev_mod = 0
p = 0
for i, surfix in enumerate(surfixes):
    print(f'\nplotting.py: sim {i+1} / {len(surfixes)}')
    print(f'plotting.py: Loading sim \'{surfix}\'...')

    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{surfix}.json', typ='series').to_dict()
    data_buffer_df = pd.read_csv(
        f'{load_folder}/data_buffer_{surfix}.csv')
    num_plants = data_buffer_df['Population'].iloc[0]

    # if num_plants < 1000 or num_plants > 2000:
    #     print(f'plotting.py: Skipping sim {n} due to low population...')
    #     continue

    L = kwargs.get('L', -1)

    subpath = path.replace('Data\\', '')
    title = f'{subpath}\\{surfix}   ($N_0$={num_plants:.0f}, L={L} m)'

    if print_kwargs:
        print('plotting.py: Loaded kwargs, now printing...')
        print()
        print_nested_dict(kwargs)
        print()

    if plot_data:
        data_buffer = DataBuffer(data=data_buffer_df)
        print('plotting.py: Loaded data_buffer, now plotting...')
        data_buffer.plot(title=title)

    if plot_states:
        state_buffer_df = pd.read_csv(
            f'{load_folder}/state_buffer_{surfix}.csv', header=None)
        state_buffer = StateBuffer(
            data=state_buffer_df, kwargs=kwargs)
        print('plotting.py: Loaded state_buffer, now plotting...')
        # state_buffer.plot(size=2, title=title, fast=fast_plot)
        ani = state_buffer.animate(title=title, fast=fast_plot)
        ani.save(f'{load_folder}/state_animation_{surfix}.mp4', writer='ffmpeg', dpi=300)

    if plot_density_field:
        density_field_buffer_df = pd.read_csv(
            f'{load_folder}/density_field_buffer_{surfix}.csv', header=None)
        density_field_buffer = FieldBuffer(
            data=density_field_buffer_df)
        print('plotting.py: Loaded density_field_buffer, now plotting...')

        density_field_buffer.plot(
            size=2, title=title)
    
    if plot_histograms:
        size_buffer_df = pd.read_csv(
            f'{load_folder}/size_buffer_{surfix}.csv', header=0)
        biomass_buffer_df = pd.read_csv(
            f'{load_folder}/biomass_buffer_{surfix}.csv', header=0)      
        size_buffer = HistogramBuffer(
            data=size_buffer_df)
        biomass_buffer = HistogramBuffer(
            data=biomass_buffer_df)
        print('plotting.py: Loaded size_buffer, now plotting...')
        ani_size = size_buffer.animate(title='Sizes', density=True, xscale=kwargs['L'], xlabel='$m$')
        ani_biomass = biomass_buffer.animate(title='Biomass', density=True, xscale=kwargs['L']**2, xlabel='$m^2$')
        

    p += int(plot_data) + \
        int(plot_states) + int(plot_density_field) + 2 * int(plot_histograms)
    mod = p % 8
    if mod <= prev_mod or i >= len(surfixes) - 1:
        plt.show()

    prev_mod = mod

print('plotting.py: Done.\n')
