import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer


def plot_kwargs(kwargs, title=None):
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


load_folder = r'Data\temp'
sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'data_buffer' in f][::-1]

p = 0
for i, n in enumerate(sim_nums):
    print(f'\nplotting.py: sim {i+1} / {len(sim_nums)}')
    print(f'plotting.py: Loading sim {n}...')

    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{n}.json', typ='series').to_dict()
    sim_kwargs = kwargs['sim_kwargs']
    plant_kwargs = kwargs['plant_kwargs']
    # lq = sim_kwargs['land_quality']
    # sg = plant_kwargs['species_germination_chance']
    m2pp = sim_kwargs['m2_per_plant']
    print('plotting.py: Loaded kwargs...')

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv')
    data_buffer = DataBuffer(data=data_buffer_arr)
    print('plotting.py: Loaded data_buffer...')

    state_buffer_arr = pd.read_csv(
        f'{load_folder}/state_buffer_{n}.csv')
    state_buffer = StateBuffer(
        data=state_buffer_arr, plant_kwargs=plant_kwargs)
    print('plotting.py: Loaded state_buffer...')

    density_field_buffer_arr = pd.read_csv(
        f'{load_folder}/density_field_buffer_{n}.csv', header=None)
    density_field_buffer = FieldBuffer(
        data=density_field_buffer_arr, skip=sim_kwargs.get('density_field_buffer_skip'), sim_kwargs=sim_kwargs)
    print('plotting.py: Loaded density_field_buffer...')

    # title = f'sim {n}, lq = {lq:.3f}, sg = {sg:.3f}'
    title = f'sim {n}, m2pp = {m2pp:.3f}'
    # plot_kwargs(kwargs, title=title)
    data_buffer.plot(title=title)
    state_buffer.plot(size=2, title=title)
    # density_field_buffer.plot(
    #     size=2, title=title)
    p += 2

    if p % 20 == 0 or i >= len(sim_nums) - 1:
        plt.show()
