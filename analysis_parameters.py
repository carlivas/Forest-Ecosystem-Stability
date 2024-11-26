import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json
import scipy

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer

load_folder = r'Data\lq_rc_ensemble_n100'
sim_names = [f.split('.')[0].replace('data_buffer_', '')
             for f in os.listdir(load_folder) if 'data_buffer_' in f]

# new_sim_nums = []
# for i, n in enumerate(sim_names):
#     data_buffer_arr = pd.read_csv(
#         f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
#     if data_buffer_arr.shape[0] == 10000:
#         new_sim_nums.append(n)
# sim_names = new_sim_nums

state_buffers = []
density_field_buffers = []
data_buffers = []
kwargs = []
end_populations = []
analyzed_sim_nums = []
for i, n in enumerate(sim_names):
    with open(os.path.join(load_folder, f'kwargs_{n}.json'), 'r') as file:
        kwargs.append(json.load(file))
    sim_kwargs = kwargs[i]['sim_kwargs']
    plant_kwargs = kwargs[i]['plant_kwargs']

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
    data_buffer_arr = data_buffer_arr[~np.isnan(data_buffer_arr).all(axis=1)]
    data_buffers.append(DataBuffer(data=data_buffer_arr))

    # state_buffer_arr = pd.read_csv(
    #     f'{load_folder}/state_buffer_{n}.csv', header=None).to_numpy()
    # state_buffers.append(StateBuffer(
    #     data=state_buffer_arr, plant_kwargs=plant_kwargs))

    # density_field_buffer_arr = pd.read_csv(
    #     f'{load_folder}/density_field_buffer_{n}.csv', header=None).to_numpy()
    # density_field_buffers.append(FieldBuffer(
    #     data=density_field_buffer_arr, skip=sim_kwargs['density_field_buffer_skip'], sim_kwargs=sim_kwargs))

    population_data = data_buffers[len(analyzed_sim_nums)].values[:, 2]
    end_population = population_data[~np.isnan(population_data)][-1]
    end_populations.append(end_population)
    analyzed_sim_nums.append(n)


fig, ax = plt.subplots()
# norm = plt.Normalize(vmin=0, vmax=max(end_populations))
norm = plt.Normalize(vmin=0, vmax=20_000)
for i, n in enumerate(analyzed_sim_nums):
    print(f'plotting.py: sim {i+1} / {len(analyzed_sim_nums)}', end='\r')
    lq = kwargs[i]['sim_kwargs']['land_quality']
    sgc = kwargs[i]['plant_kwargs']['species_germination_chance']

    last_time = data_buffers[i].values[:, 0][-1]
    alpha = np.clip((last_time)/5e3, 0.2, 1)
    # alpha = 1
    color = plt.cm.ScalarMappable(
        norm=norm, cmap='coolwarm').to_rgba(end_populations[i])
    scatter = ax.scatter(lq, sgc, color=color, alpha=alpha, edgecolors='none')
    # annotation = ax.text(x, y, str(n), fontsize=6)

ax.set_xlabel('land quality', fontsize=8)
ax.set_ylabel('germination chance', fontsize=8)
ax.set_title(
    'End population vs. land quality and germination chance\n(visibility prop to sim lifetime)', fontsize=10)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax)
cbar.set_label('End population')

ax.tick_params(axis='both', which='major', labelsize=7)
cbar.ax.tick_params(labelsize=7)

plt.show()
