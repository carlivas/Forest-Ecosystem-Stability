import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

folder = 'D:/linear_precipitation/L2000'

load_folder = os.path.abspath(folder)
print(f'load_folder: {load_folder}')

aliases_to_remove = []

for root, dirs, files in os.walk(load_folder):
    db_aliases = [f.split('-')[-1].split('.')[0] for f in files if 'data_buffer-' in f]
    sb_aliases = [f.split('-')[-1].split('.')[0] for f in files if 'state_buffer-' in f]
    kwargs_aliases = [f.split('-')[-1].split('.')[0] for f in files if 'kwargs-' in f]
    dfb_aliases = [f.split('-')[-1].split('.')[0] for f in files if 'density_field_buffer-' in f]
    
    all_aliases = [s for s in db_aliases + sb_aliases + kwargs_aliases + dfb_aliases if 'checkpoint' not in s]
    
    complete_aliases = [alias for alias in all_aliases if all_aliases.count(alias) == 4]
    
    aliases_to_remove += [alias for alias in all_aliases if alias not in complete_aliases]
    
    print(f'{len(all_aliases)} aliases found in {root}')
    print(f'{len(complete_aliases)} complete aliases found in {root}')
    
print(f'aliases_to_remove: {aliases_to_remove}')

for root, dirs, files in os.walk(load_folder):
    for alias in aliases_to_remove:
        for f in files:
            if alias in f:
                os.remove(os.path.join(root, f))
                print(f'Removed: {os.path.join(root, f)}')
                
print('Done')
