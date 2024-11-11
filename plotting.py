import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from mods.plant import Plant
from mods.simulation import Simulation
from mods.state_buffer import StateBuffer
from mods.data_buffer import DataBuffer
from mods.field_buffer import FieldBuffer


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
    lq = sim_kwargs['land_quality']
    sg = plant_kwargs['species_germination_chance']
    print('plotting.py: Loaded kwargs...')

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
    # if data_buffer_arr.shape[0] < 10000:
    #     print('plotting.py: Skipping sim due to small data_buffer...')
    #     continue
    data_buffer = DataBuffer(data=data_buffer_arr)
    print('plotting.py: Loaded data_buffer...')

    state_buffer_arr = pd.read_csv(
        f'{load_folder}/state_buffer_{n}.csv', header=None).to_numpy()
    state_buffer = StateBuffer(
        data=state_buffer_arr, plant_kwargs=plant_kwargs)
    print('plotting.py: Loaded state_buffer...')

    density_field_buffer_arr = pd.read_csv(
        f'{load_folder}/density_field_buffer_{n}.csv', header=None).to_numpy()
    density_field_buffer = FieldBuffer(
        data=density_field_buffer_arr, skip=sim_kwargs.get('density_field_buffer_skip'), sim_kwargs=sim_kwargs)
    print('plotting.py: Loaded density_field_buffer...')

    # plot_kwargs(kwargs, title=f'{load_folder} - sim {n}')
    data_buffer.plot(title=f'sim {n}, lq = {lq:.3f}, sg = {sg:.3f}')
    state_buffer.plot(size=2, title=f'sim {n}, lq = {lq:.3f}, sg = {sg:.3f}')
    # density_field_buffer.plot(
    #     size=2, title=f'{load_folder} - sim {n}')
    p += 2

    if p % 20 == 0 or i >= len(sim_nums) - 1:
        plt.show()
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# for i, n in enumerate(sim_nums):
#     kwargs = pd.read_json(
#         f'{load_folder}/kwargs_{n}.json', typ='series').to_dict()
#     plant_kwargs = kwargs['plant_kwargs']
#     sim_kwargs = kwargs['sim_kwargs']
#     plot_kwargs(kwargs, title=f'{load_folder} - sim {n}')
#     print('plotting.py: Loaded kwargs...')

#     density_field_buffer_arr = pd.read_csv(
#         f'{load_folder}/density_field_buffer_{n}.csv', header=None).to_numpy()
#     density_field_buffer = FieldBuffer(
#         data=density_field_buffer_arr, skip=sim_kwargs.get('density_field_buffer_skip'), sim_kwargs=sim_kwargs)
#     density_field_buffer.plot(
#         size=2, title=f'{load_folder} - sim {n}')

#     state_buffer_arr = pd.read_csv(
#         f'{load_folder}/state_buffer_{n}.csv', header=None).to_numpy()
#     state_buffer = StateBuffer(
#         data=state_buffer_arr, plant_kwargs=plant_kwargs)
#     print('plotting.py: Loaded state_buffer...')
#     state_buffer.plot(size=2)

#     data_buffer_arr = pd.read_csv(
#         f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
#     data_buffer = DataBuffer(data=data_buffer_arr)
#     print('plotting.py: Loaded data_buffer...')
#     data_buffer.plot(title=f'{load_folder} - sim {n}')

#     plt.show()
