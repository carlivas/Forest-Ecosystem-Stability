import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from mods.simulation import Simulation
from mods.buffers import DataBuffer, StateBuffer, FieldBuffer
print('\nplotting.py: Running...\n')

do_plots = True
do_animations = True
save_figs = True

path = 'Data/dynamics' # Path to the folder containing the buffers
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

aliases = [f.split('-')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs-' in f]
print(f'aliases: {aliases}')

for alias in aliases:
    sim = Simulation(folder=load_folder, alias=alias)
    
    if do_plots:
        db_fig, db_ax, db_title = DataBuffer.plot(data=sim.data_buffer.get_data(), size = (5, 7), keys = ['Biomass', 'Population', 'Precipitation'], title=alias)
        # sb_fig, sb_ax, sb_title = StateBuffer.plot(sim.state_buffer.get_data(), title=alias)
        dfb_fig, dfb_ax, sb_title = FieldBuffer.plot(sim.density_field_buffer.get_data(), title=alias)
        if save_figs:
            db_save_path = f'{load_folder}/figures/data_buffer-{alias}.png'
            db_fig.savefig(db_save_path, dpi = 600)
            # sb_save_path = f'{load_folder}/figures/state_buffer-{alias}.png'
            # sb_fig.savefig(sb_save_path, dpi = 600)
            dfb_save_path = f'{load_folder}/figures/density_field_buffer-{alias}.png'
            dfb_fig.savefig(dfb_save_path, dpi = 600)
            
    if do_animations:
        sb_data = sim.state_buffer.get_data()
        if 'species' not in sb_data.columns:
                warnings.warn(
                    'plotting.py: "species" column not found in state_buffer. Assuming species_id = -1 for all plants.')
                sb_data['species'] = -1
        
        sb_anim, _ = StateBuffer.animate(sb_data, skip = 15, title=alias, fast=False)
        if save_figs:
            sb_anim.save(f'{load_folder}/figures/state_anim-{alias}.mp4', dpi = 600)
    
    if not save_figs:
        plt.show()
    

print('plotting.py: Done.\n')
