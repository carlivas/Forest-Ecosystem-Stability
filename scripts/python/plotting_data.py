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

darkgrey = np.array([30,  30,  30, 255])/255
grey = np.array([128, 128, 128, 255])/255
white = np.array([225, 225, 225, 255])/255

path = sys.argv[1]
if not os.path.exists(path):
    raise FileNotFoundError(f"The specified path does not exist: {path}")

load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')


sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs' in f]
print(f'sim_nums: {sim_nums}')


biomass_ylim = 0
population_ylim = 0
cut = 20


max_num_plants = 0
kwargs_list = []
data_buffer_list = []
for i, n in enumerate(sim_nums):
    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{n}.json', typ='series').to_dict()
    data_buffer = pd.read_csv(f'{load_folder}/data_buffer_{n}.csv')
    data_buffer_list.append(data_buffer)
    kwargs_list.append(kwargs)
    sim_kwargs = kwargs.get('sim_kwargs')
    num_plants = sim_kwargs.get('num_plants')
    max_num_plants = max(max_num_plants, num_plants)

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

teal = np.array([0, 128, 128, 255])/255
green = np.array([0, 128, 0, 255])/255


def teal_cmap(t): return white * (1 - t) + teal * t
def green_cmap(t): return white * (1 - t) + green * t


norm = plt.Normalize(vmin=0, vmax=max_num_plants)
sm = []
sm.append(plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(
    [teal_cmap(t) for t in np.linspace(0, 1, 256)]), norm=norm))
sm.append(plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(
    [green_cmap(t) for t in np.linspace(0, 1, 256)]), norm=norm))


for i, n in enumerate(sim_nums):
    kwargs = kwargs_list[i]
    data_buffer_df = data_buffer_list[i]
    sim_kwargs = kwargs.get('sim_kwargs')
    plant_kwargs = kwargs.get('plant_kwargs')
    num_plants = sim_kwargs.get('num_plants')

    label = f'({n})  $P_0 = {int(num_plants):>5}$'

    time = data_buffer_df['Time']
    biomass = data_buffer_df['Biomass']
    population = data_buffer_df['Population']

    if np.max(biomass[cut:]) > biomass_ylim:
        biomass_ylim = np.max(biomass[cut:])
    if np.max(population[cut:]) > population_ylim:
        population_ylim = np.max(population[cut:])

    ax[0].plot(time, biomass, label=label,
               color=teal_cmap(num_plants / max_num_plants), alpha=0.5, lw=0.5)
    ax[1].plot(time, population, label=label,
               color=green_cmap(num_plants / max_num_plants), alpha=0.5, lw=0.5)

ax[0].set_xlabel('Time', color=white, fontsize=9)
ax[0].set_title('Biomass', color=white, fontsize=9)
ax[1].set_xlabel('Time', color=white, fontsize=9)
ax[1].set_title('Population', color=white, fontsize=9)
ax[0].set_ylim(0, biomass_ylim)
ax[1].set_ylim(0, population_ylim)


for ax, sm in zip(ax, sm):
    ax.set_xlim(0, 10000)
    # ax.legend(facecolor='grey', fontsize=3, loc='upper right')
    ax.grid(True, color=grey, linewidth=0.5)
    ax.set_facecolor(darkgrey)
    ax.tick_params(axis='x', colors=grey, labelsize=8)
    ax.tick_params(axis='y', colors=grey, labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Initial population', color=white, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=grey)
    cbar.outline.set_edgecolor(grey)
    cbar.ax.yaxis.set_tick_params(labelcolor=grey)

fig.set_facecolor(darkgrey)
fig.suptitle(f'{path}', color=white, fontsize=10)
# fig.legend(facecolor='grey', fontsize=4, loc='upper right', bbox_to_anchor=(1.1, 1.05))

surfix = load_folder.split('\\')[-1]  # + '_zoom'

plt.savefig(f'{load_folder}/data_combined_{surfix}.png', dpi=300)

plt.show()
print('plotting.py: Done.\n')
