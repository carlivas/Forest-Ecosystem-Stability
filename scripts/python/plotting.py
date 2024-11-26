import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer
from mods.utilities import print_nested_dict

load_folder = r'Data\init_density_experiment_SPH\ensemble'
sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'data_buffer' in f][::-1]
print(f'sim_nums: {sim_nums}')

print_kwargs = True
plot_data = True
plot_states = True
plot_density_field = False

fast = False


def plot_kwargs_func(kwargs, title=None):
    fig, ax = plt.subplots()
    if title is not None:
        ax.set_title(f'kwargs ({title})', fontsize=10)
    ax.axis('off')
    ax.axis('tight')

    table_data = []
    for key, subdict in kwargs.items():
        table_data.append([key, ""])
        for subkey, value in subdict.items():
            table_data.append([f"    {subkey}", value])  # Increased indent

    table = ax.table(cellText=table_data, colLabels=[
                     "Key", "Value"], cellLoc='left', loc='center')

    table.scale(0.9, 0.9)
    table.auto_set_font_size(True)

    # Make table lines thinner
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.1)

    # Make column labels bold
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')

    fig.tight_layout()


prev_mod = 0
p = 0
for i, n in enumerate(sim_nums):
    print(f'\nplotting.py: sim {i+1} / {len(sim_nums)}')
    print(f'plotting.py: Loading sim {n}...')

    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{n}.json', typ='series').to_dict()
    sim_kwargs = kwargs['sim_kwargs']
    plant_kwargs = kwargs['plant_kwargs']
    lq = sim_kwargs['land_quality']
    sg = plant_kwargs['species_germination_chance']
    # dens0 = sim_kwargs['dens0']
    num_plants = sim_kwargs['num_plants']
    title = f'{n}   (lq={lq:.3e},   sg={sg:.3e})'  # ,   dens0={(dens0):.3e})'

    if print_kwargs:
        print('plotting.py: Loaded kwargs...')
        # plot_kwargs_func(kwargs, title=title)
        print_nested_dict(kwargs)
        print()

    if plot_data:
        data_buffer_arr = pd.read_csv(
            f'{load_folder}/data_buffer_{n}.csv')
        data_buffer = DataBuffer(data=data_buffer_arr)
        print('plotting.py: Loaded data_buffer...')
        data_buffer.plot(title=title)

    if plot_states:
        state_buffer_arr = pd.read_csv(
            f'{load_folder}/state_buffer_{n}.csv')
        state_buffer = StateBuffer(
            data=state_buffer_arr, plant_kwargs=plant_kwargs)
        print('plotting.py: Loaded state_buffer...')
        state_buffer.plot(size=2, title=title, fast=fast)

    if plot_density_field:
        density_field_buffer_arr = pd.read_csv(
            f'{load_folder}/density_field_buffer_{n}.csv', header=None)
        density_field_buffer = FieldBuffer(
            data=density_field_buffer_arr, skip=sim_kwargs['density_field_buffer_skip'], sim_kwargs=sim_kwargs)
        print('plotting.py: Loaded density_field_buffer...')

        density_field_buffer.plot(
            size=2, title=title)

    p += int(plot_data) + \
        int(plot_states) + int(plot_density_field)
    mod = p % 10
    if mod < prev_mod or i >= len(sim_nums) - 1:
        plt.show()

    prev_mod = mod
