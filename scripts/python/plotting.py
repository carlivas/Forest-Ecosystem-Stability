import numpy as np
import os
import matplotlib.pyplot as plt
from mods.simulation import Simulation
from mods.buffers import DataBuffer, StateBuffer

show_figs = False
save_figs = True

path = '../../Data/linear_precipitation/L1000'
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

aliases = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs' in f][::-1]
print(f'aliases: {aliases}')

for alias in aliases:
    sim = Simulation(folder=load_folder, alias=alias)
    db_fig, db_ax, db_title = sim.data_buffer.plot(size = (8, 7), keys = ['Biomass', 'Precipitation'])
    db_ax[0].set_ylim(-0.05, 0.95)
    sb_fig, sb_ax, sb_title = StateBuffer.plot(sim.state_buffer.get_data())
    if save_figs:
        db_save_path = load_folder + '/figures' + f'/_{db_title}_' + alias
        db_fig.savefig(db_save_path, dpi = 600)
        sb_save_path = load_folder + '/figures' + f'/_{sb_title}_' + alias
        sb_fig.savefig(sb_save_path, dpi = 600)
    if show_figs:
        plt.show()

print('plotting.py: Done.\n')
