import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
from mods.simulation import Simulation
from mods.buffers import DataBuffer, StateBuffer, FieldBuffer
print('\nplotting.py: Running...\n')

do_plots = True
fast_plots = False
save_figs = True

do_animations = False
fast_animations = False
animation_skip = 5

path = 'Data/temp' # Path to the folder containing the buffers
load_folder = os.path.abspath(path)
print(f'load_folder: {load_folder}')

kwargs_aliases = [f.split('-')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'kwargs-' in f]
db_aliases = [f.split('-')[-1].split('.')[0] for f in os.listdir(load_folder) if 'data_buffer-' in f]
sb_aliases = [f.split('-')[-1].split('.')[0] for f in os.listdir(load_folder) if 'state_buffer-' in f]
# dfb_aliases = [f.split('-')[-1].split('.')[0] for f in os.listdir(load_folder) if 'density_field_buffer-' in f]
complete_aliases = list(set(kwargs_aliases) & set(db_aliases) & set(sb_aliases))
# complete_aliases.sort()
aliases = complete_aliases[::-1]
print(f'aliases: {aliases}')

for i, alias in enumerate(aliases):
    sim = Simulation(folder=load_folder, alias=alias)
    sb_data, db_data, dfb_data = None, None, None

    # COMMENT OUT THE LINES BELOW TO AVOID PLOTTING
    sb_data = sim.state_buffer.get_data()
    db_data = sim.data_buffer.get_data()
    # dfb_data = sim.density_field_buffer.get_data()
    ###############################################

    if do_plots:
        if db_data is not None:
            print(f'plotting.py: Plotting data_buffer for alias {alias}')
            db_fig, db_ax, db_title = DataBuffer.plot(data=db_data, size=(
                7, 7), keys=['Biomass', 'Population', 'Precipitation'], title=alias, dict_to_print=sim.get_kwargs())
            db_title = db_title.replace(' ', '_').lower()

        if sb_data is not None and do_animations is False:
            print(f'plotting.py: Plotting state_buffer for alias {alias}')
            sb_fig, sb_ax, sb_title = StateBuffer.plot(
                sb_data, title=alias, box=sim.box, boundary_condition=sim.boundary_condition, fast=fast_plots)
            sb_title = sb_title.replace(' ', '_').lower()

        if dfb_data is not None:
            print(f'plotting.py: Plotting density_field_buffer for alias {alias}')
            dfb_fig, dfb_ax, dfb_title = FieldBuffer.plot(
                dfb_data, title=alias, box=sim.box, boundary_condition=sim.boundary_condition, fast=fast_plots)
            dfb_title = dfb_title.replace(' ', '_').lower()

        if save_figs:
            if db_data is not None:
                db_save_path = f'{load_folder}/figures/data_buffer-{alias}.png'

                os.remove(db_save_path) if os.path.exists(db_save_path) else None
                db_fig.savefig(db_save_path, dpi=600)
            if sb_data is not None and do_animations is False:
                sb_save_path = f'{load_folder}/figures/state_buffer-{alias}.png'

                os.remove(sb_save_path) if os.path.exists(sb_save_path) else None
                sb_fig.savefig(sb_save_path, dpi=600)
            if dfb_data is not None:
                dfb_save_path = f'{load_folder}/figures/density_field_buffer-{alias}.png'

                os.remove(dfb_save_path) if os.path.exists(dfb_save_path) else None
                dfb_fig.savefig(dfb_save_path, dpi=600)

    if do_animations:
        if sb_data is not None:
            print(f'plotting.py: Animating state_buffer for alias {alias}')
            if 'species' not in sb_data.columns:
                warnings.warn(
                    'plotting.py: "species" column not found in state_buffer. Assuming species_id = -1 for all plants.')
                sb_data['species'] = -1

            sb_anim, _ = StateBuffer.animate(
                sb_data, skip=animation_skip, title=alias, box=sim.box, boundary_condition=sim.boundary_condition, fast=fast_animations)
            if save_figs:
                sb_anim_path = f'{load_folder}/figures/state_anim-{alias}.mp4'
                sb_save_path = f'{load_folder}/figures/state_buffer-{alias}.png'

                os.remove(sb_anim_path) if os.path.exists(sb_anim_path) else None
                os.remove(sb_save_path) if os.path.exists(sb_save_path) else None
                
                sb_anim.save(sb_anim_path, dpi=600)
                
    print(f'\nDone plotting alias: {alias}   ({i+1}/{len(aliases)})\n')

    if not save_figs:
        plt.show()


print('plotting.py: Done.\n')
