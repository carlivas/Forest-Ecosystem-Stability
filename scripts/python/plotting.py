import numpy as np
import os
import matplotlib.pyplot as plt
from mods.simulation import Simulation
from mods.buffers import DataBuffer, StateBuffer
print('\nplotting.py: Running...\n')

do_plots = True
do_animations = True
save_figs = True

path = 'Data/small_domain/small_test400' # Path to the folder containing the buffers
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

aliases = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs' in f][::-1]
print(f'aliases: {aliases}')

for alias in aliases:
    sim = Simulation(folder=load_folder, alias=alias)
    
    if do_plots:
        db_fig, db_ax, db_title = sim.data_buffer.plot(size = (5, 7), keys = ['Biomass', 'Population', 'Precipitation'], title=alias)
        # sb_fig, sb_ax, sb_title = StateBuffer.plot(sim.state_buffer.get_data(), title=alias)
        if save_figs:
            db_save_path = f'{load_folder}/figures/data_buffer_{alias}.png'
            db_fig.savefig(db_save_path, dpi = 600)
            # sb_save_path = f'{load_folder}/figures/state_buffer_{alias}.png'
            # sb_fig.savefig(sb_save_path, dpi = 600)
            
    if do_animations:
        sb_anim, _ = StateBuffer.animate(sim.state_buffer.get_data(), skip = 5, title=alias)
        if save_figs:
            sb_anim.save(f'{load_folder}/figures/state_anim_{alias}.mp4', dpi = 600)
    else:
        plt.show()

print('plotting.py: Done.\n')
