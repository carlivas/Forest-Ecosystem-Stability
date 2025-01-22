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

path = '../../Data/starting_contenders'
save_plot = True

if not os.path.exists(path):
    raise FileNotFoundError(f"The specified path does not exist: {path}")

load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

for root, dirs, files in os.walk(load_folder):
    sim_nums = [f.split('_')[-1].split('.')[0] for f in files if 'kwargs' in f]
    if not sim_nums:
        continue
    print(f'sim_nums: {sim_nums}')

    biomass_ylim = 0
    population_ylim = 0
    cut = 20

    max_num_plants = 0
    kwargs_list = []
    data_buffer_list = []
    for i, n in enumerate(sim_nums):
        kwargs = pd.read_json(
            f'{root}/kwargs_{n}.json', typ='series').to_dict()
        data_buffer = pd.read_csv(f'{root}/data_buffer_{n}.csv')
        data_buffer_list.append(data_buffer)
        kwargs_list.append(kwargs)
        num_plants = data_buffer['Population'].iloc[0]
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

    sm = []
    bin_colors = ['#013E31', '#358A5C', '#7CBB95', '#C83E46', '#8f6949', '#643735']
    for i in range(2):
        sm.append(plt.cm.ScalarMappable(cmap=plt.cm.colors.ListedColormap(
            bin_colors), norm=norm))

    max_time = 0
    for i, n in enumerate(sim_nums):
        kwargs = kwargs_list[i]
        data_buffer_df = data_buffer_list[i]
        time = data_buffer_df['Time']
        biomass = data_buffer_df['Biomass']
        population = data_buffer_df['Population']
        num_plants = population.iloc[0]
        if time.iloc[-1] > max_time:
            max_time = time.iloc[-1]

        label = f'({n})  $P_0 = {int(num_plants):>5}$'

        if np.max(biomass[cut:]) > biomass_ylim:
            biomass_ylim = np.max(biomass[cut:])
        if np.max(population[cut:]) > population_ylim:
            population_ylim = np.max(population[cut:])

        bins = np.linspace(0, max_num_plants, 6)  # Define 6 bin sides
        bin_colors_teal = [teal_cmap(i / (len(bins) - 1))
                        for i in range(len(bins))]
        bin_colors_green = [green_cmap(i / (len(bins) - 1))
                            for i in range(len(bins))]

        def get_bin_color(value, bins, colors):
            for i in range(len(bins) - 1):
                if bins[i] <= value < bins[i + 1]:
                    return colors[i]
            return colors[-1]

        color_teal = get_bin_color(num_plants, bins, bin_colors_teal)
        color_green = get_bin_color(num_plants, bins, bin_colors_green)
        color = get_bin_color(num_plants, bins, bin_colors)

        ax[0].plot(time, biomass, label=label, color=color, alpha=1, lw=0.3)
        ax[1].plot(time, population, label=label,
                color=color, alpha=1, lw=0.3)

    ax[0].set_xlabel('Time', color=white, fontsize=9)
    ax[0].set_ylabel('Biomass', color=white, fontsize=9)
    ax[1].set_xlabel('Time', color=white, fontsize=9)
    ax[1].set_ylabel('Population', color=white, fontsize=9)
    ax[0].set_ylim(0, biomass_ylim)
    ax[1].set_ylim(0, population_ylim)

    for ax, sm in zip(ax, sm):
        ax.set_xlim(0, max_time)
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
    
    subpath = root.split('Data/')[-1]
    fig.suptitle(f'{subpath}', color=white, fontsize=10)

    surfix = root.split('/')[-1]

    if save_plot:
        save_path = f'{root}/_data_combined_{surfix}.png'
        print(f'Saving data plot in {save_path = }')
        plt.savefig(save_path, dpi=300)

    plt.show()
print('plotting_data.py: Done.\n')
