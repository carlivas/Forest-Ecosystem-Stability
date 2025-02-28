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

path = 'D:/695774818_finished'
save_plot = True

if not os.path.exists(path):
    raise FileNotFoundError(f"The specified path does not exist: {path}")

load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

for root, dirs, files in os.walk(load_folder):
    aliases = [f.split('-')[-1].split('.')[0] for f in files if 'kwargs' in f]
    aliases = [s for s in aliases if 'checkpoint' not in s]
    aliases = sorted(aliases, key=lambda x: int(x.split('-')[-1]))
    if not aliases:
        continue
    print(f'aliases: {aliases}')

    biomass_ylim = 0
    precipitation_ylim = 0
    cut = 20

    max_num_plants = 0
    kwargs_list = []
    data_buffer_list = []
    for i, alias in enumerate(aliases):
        kwargs = pd.read_json(
            f'{root}/kwargs-{alias}.json', typ='series').to_dict()
        data_buffer = pd.read_csv(f'{root}/data_buffer-{alias}.csv')
        data_buffer_list.append(data_buffer)
        kwargs_list.append(kwargs)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    max_time = 0
    for i, alias in enumerate(aliases):
        num = int(alias.split('-')[-1])
        kwargs = kwargs_list[i]
        data_buffer_df = data_buffer_list[i]
        time = data_buffer_df['Time']
        biomass = data_buffer_df['Biomass']
        precipitation = pd.DataFrame(kwargs['precipitation'] * np.ones_like(time))
        if time.iloc[-1] > max_time:
            max_time = time.iloc[-1]

        colors = ['#358A5C', '#7CBB95', '#C83E46', '#8f6949', '#643735', '#013E31']
        ax[0].plot(time, biomass, color=colors[i%len(colors)], alpha=1, lw=0.3)
        ax[1].plot(time, precipitation, color=colors[i%len(colors)], alpha=1, lw=2)
        
        # Write the simulation number
        # ax[0].text(time.iloc[-1], biomass.iloc[-1], str(num), fontsize=8, color=colors[i%len(colors)], verticalalignment='bottom')
        ax[1].text(time.iloc[-1], precipitation.iloc[-1], str(num), fontsize=8, color=colors[i%len(colors)], verticalalignment='bottom')
        

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

    subpath = root.split('Data/')[-1]
    fig.suptitle(f'{subpath}', color=white, fontsize=10)

    alias = root.split('\\')[-1]

    print(f'{alias=}')

    if save_plot:
        save_path = f'{root}/_successive-{alias}.png'
        print(f'Saving data plot in {save_path=}')
        plt.savefig(save_path, dpi=300)

    plt.show()
print('plotting_successive.py: Done.\n')
