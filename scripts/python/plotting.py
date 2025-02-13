import numpy as np
import os
import matplotlib.pyplot as plt
from mods.simulation import Simulation

show_figs = False
save_figs = True

path = '../../Data/starting_contenders/695774818'
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

aliases = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs' in f][::-1]
print(f'aliases: {aliases}')

for alias in aliases:
    sim = Simulation(folder=load_folder, alias=alias)
    # db_fig, db_ax = sim.data_buffer.plot()
    sb_fig, sb_ax = sim.state_buffer.plot()
    if save_figs:
        # db_save_path = load_folder + '/' + f'_fig_data_' + alias
        # db_fig.savefig(db_save_path, dpi = 600)
        sb_save_path = load_folder + '/' + f'_fig_plants_' + alias
        sb_fig.savefig(sb_save_path, dpi = 600)
    if show_figs:
        plt.show()

print('plotting.py: Done.\n')
