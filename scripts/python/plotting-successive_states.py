import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys

from mods.plant import Plant
from mods.simulation import Simulation, sim_from_data, load_sim_data, plot_simulation_results
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer, HistogramBuffer
from mods.utilities import print_nested_dict

save_fig = True

path = 'D:/partial48775395'
folder = os.path.abspath(path)
print(f'folder: {folder}')

aliases = [f.split('-')[-1].split('.')[0]
            for f in os.listdir(folder) if 'kwargs-' in f][::-1]
aliases = sorted(aliases, key=lambda x: int(x.split('-')[-1]))
print(f'{len(aliases)=}')

fast_plot = False

states = pd.DataFrame()
for i, alias in enumerate(aliases):
    if int(alias.split('-')[-1]) < 7:
        continue
    state_buffer_path = f'{folder}/state_buffer-{alias}.csv'
    state_buffer = StateBuffer(state_buffer_path)
    
    skip_times = 1000
    data = state_buffer.get_data()
    times_unique = data['t'].unique()
    times = times_unique[::skip_times]
    data_trimmed = data[data['t'].isin(times)]
    states = states._append(data_trimmed, ignore_index=True)
    print(f'{len(states)=}')

fig, ax = StateBuffer.plot(data=states, title=f'States_{path.split('/')[-1]}', n_plots=30, fast=fast_plot)
if save_fig:
    fig_path = f'{folder}/_states_{path.split('/')[-1]}.png'
    fig.savefig(fig_path, dpi=600)

# ### ANIMATION ISN'T WORKING QUITE RIGHT YET ###
# ani = StateBuffer.animate(data=states, title=f'States_{path.split("/")[-1]}', fast=fast_plot)
# if save_fig:
#     ani_path = f'{folder}/_states_{path.split("/")[-1]}.mp4'
#     ani.save(ani_path, writer='ffmpeg', fps=1, dpi=600)
plt.show()

print('plotting.py: Done.\n')
