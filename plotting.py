import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from plant import Plant
from simulation import Simulation, StateBuffer, DataBuffer, FieldBuffer


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
    # table.set_fontsize(7)
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


load_folder = r'Data\20241028_114615'
n_files = len([f for f in os.listdir(load_folder) if 'data_buffer' in f])
start_idx = 0
# for i in range(start_idx, start_idx + n_files):
for i in range(start_idx, start_idx + 10):
    surfix = str(i)
    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{surfix}.json', typ='series').to_dict()
    plant_kwargs = kwargs['plant_kwargs']
    sim_kwargs = kwargs['sim_kwargs']
    # plot_kwargs(kwargs, title=f'{load_folder} - sim {surfix}')
    print('plotting.py: Loaded kwargs...')

    # density_field_buffer_arr = pd.read_csv(
    #     f'{load_folder}/density_field_buffer_{surfix}.csv', header=None).to_numpy()
    # density_field_buffer = FieldBuffer(
    #     data=density_field_buffer_arr, skip=sim_kwargs.get('density_field_buffer_skip'), sim_kwargs=sim_kwargs)
    # density_field_buffer.plot(
    #     size=2, title=f'{load_folder} - sim {surfix}')

    # state_buffer_arr = pd.read_csv(
    #     f'{load_folder}/state_buffer_{surfix}.csv', header=None).to_numpy()
    # state_buffer = StateBuffer(
    #     data=state_buffer_arr, plant_kwargs=plant_kwargs)
    # print('plotting.py: Loaded state_buffer...')
    # state_buffer.plot(size=2)

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{surfix}.csv', header=None).to_numpy()
    data_buffer = DataBuffer(data=data_buffer_arr)
    print('plotting.py: Loaded data_buffer...')
    data_buffer.plot(title=f'{load_folder} - sim {surfix}')

plt.show()
