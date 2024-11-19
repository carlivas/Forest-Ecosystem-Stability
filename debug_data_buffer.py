import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from mods.plant import Plant
from mods.simulation import Simulation
from mods.state_buffer import StateBuffer
from mods.data_buffer_keys import DataBuffer
from mods.field_buffer import FieldBuffer

load_folder = r'Data\data_buff_test'
sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'data_buffer' in f]  # [::-1]

p = 0
for i, n in enumerate(sim_nums[:1]):
    print(f'\nplotting.py: sim {i+1} / {len(sim_nums)}')
    print(f'plotting.py: Loading sim {n}...')

    kwargs = pd.read_json(
        f'{load_folder}/kwargs_{n}.json', typ='series').to_dict()
    sim_kwargs = kwargs['sim_kwargs']
    plant_kwargs = kwargs['plant_kwargs']
    lq = sim_kwargs['land_quality']
    sg = plant_kwargs['species_germination_chance']
    print('plotting.py: Loaded kwargs...')

    data_buffer_df = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=0)
    print(data_buffer_df)
    data_buffer = DataBuffer(data=data_buffer_df)
    print('plotting.py: Loaded data_buffer...')

biomass, time, population = data_buffer.get_data(
    keys=['Biomass', 'Time', 'Population'])

fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(time, biomass, label='Biomass', color='green')
ax[1].plot(time, population, label='Population', color='blue')
plt.show()


# data_buffer.plot(title=f'sim {n}, lq = {lq:.3f}, sg = {sg:.3f}')
# plt.show()
