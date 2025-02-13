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

path = '../../Data/starting_contenders/partial48775395' # REMEMBER TO SET THE PATH
sim_num_max = 19 # REMEMBER TO SET SIM_NUM_MAX AS TO NOT HIT THE RUNNING SIMULATION DATA
save_plot = True

if not os.path.exists(path):
    raise FileNotFoundError(f"The specified path does not exist: {path}")

load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

for root, dirs, files in os.walk(load_folder):
    sim_nums = [f.split('_')[-1].split('.')[0] for f in files if 'kwargs' in f]
    sim_nums = [n for n in sim_nums if 'checkpoint' not in n]
    sim_nums = [n for n in sim_nums if int(n.split('-')[-1]) <= sim_num_max]
    sim_nums = sorted(sim_nums, key=lambda n: int(n.split('-')[-1]))
    if not sim_nums:
        continue
        
    print(f'sim_nums: {sim_nums}')
    print(f'{len(sim_nums) = }')
    
    biomass_ylim = 0
    precipitation_ylim = 0
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

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    max_time = 0
    for i, n in enumerate(sim_nums):
        kwargs = kwargs_list[i]
        data_buffer_df = data_buffer_list[i]
        time = data_buffer_df['Time']
        biomass = data_buffer_df['Biomass']
        precipitation = kwargs['precipitation'] * np.ones_like(time)
        if time.iloc[-1] > max_time:
            max_time = time.iloc[-1]

        colors = ['#358A5C', '#7CBB95', '#C83E46', '#8f6949', '#643735', '#013E31']
        ax[0].plot(time, biomass, color=colors[i % len(colors)], alpha=1, lw=0.3)
        ax[1].plot(time, precipitation,
                   color=colors[i%len(colors)], alpha=1, lw=2)

    ax[0].set_xlabel('Time', color=white, fontsize=9)
    ax[0].set_ylabel('Biomass', color=white, fontsize=9)
    ax[1].set_xlabel('Time', color=white, fontsize=9)
    ax[1].set_ylabel('Precipitation', color=white, fontsize=9)

    for a in ax:
        a.set_xlim(0, max_time)
        a.grid(True, color=grey, linewidth=0.5)
        a.set_facecolor(darkgrey)
        a.tick_params(axis='x', colors=grey, labelsize=8)
        a.tick_params(axis='y', colors=grey, labelsize=8)
        for spine in a.spines.values():
            spine.set_visible(False)

    fig.set_facecolor(darkgrey)

    subpath = load_folder.split('Data/')[-1]
    fig.suptitle(f'{subpath}', color=white, fontsize=10)

    surfix = load_folder.split('/')[-1]

    if save_plot:
        save_path = f'{load_folder}/_successive_{surfix}.png'
        print(f'Saving data plot in {save_path=}')
        plt.savefig(save_path, dpi=300)

    plt.show()
print('plotting_successive.py: Done.\n')
