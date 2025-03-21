import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from mods.simulation import Simulation
from mods.buffers import DataBuffer, StateBuffer, FieldBuffer
print('\nplotting.py: Running...\n')

do_plots = True
do_animations = False
save_figs = True

path = 'D:/linear_precipitation/L2000' # Path to the folder containing the buffers
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

kwargs_aliases = [f.split('-')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs-' in f]
db_aliases = [f.split('-')[-1].split('.')[0] for f in os.listdir(load_folder) if 'data_buffer-' in f]
sb_aliases = [f.split('-')[-1].split('.')[0] for f in os.listdir(load_folder) if 'state_buffer-' in f]
dfb_aliases = [f.split('-')[-1].split('.')[0] for f in os.listdir(load_folder) if 'density_field_buffer-' in f]
complete_aliases = list(set(kwargs_aliases) & set(db_aliases) & set(sb_aliases) & set(dfb_aliases))

aliases = complete_aliases
print(f'aliases: {aliases}')

for alias in aliases:
    sim = Simulation(folder=load_folder, alias=alias)
    
    if do_plots:
        db_fig, db_ax, db_title = DataBuffer.plot(data=sim.data_buffer.get_data(), size = (7, 7), keys = ['Biomass', 'Population', 'Precipitation'], title=alias)
        sb_fig, sb_ax, sb_title = StateBuffer.plot(sim.state_buffer.get_data(), title=alias)
        # dfb_fig, dfb_ax, sb_title = FieldBuffer.plot(sim.density_field_buffer.get_data(), title=alias)
        
        sb_title = sb_title.replace(' ', '_').lower()
        db_title = db_title.replace(' ', '_').lower()
        if save_figs:
            db_save_path = f'{load_folder}/figures/data_buffer-{alias}.png'
            db_fig.savefig(db_save_path, dpi = 600)
            sb_save_path = f'{load_folder}/figures/state_buffer-{alias}.png'
            sb_fig.savefig(sb_save_path, dpi = 600)
            # dfb_save_path = f'{load_folder}/figures/density_field_buffer-{alias}.png'
            # dfb_fig.savefig(dfb_save_path, dpi = 600)
            
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
