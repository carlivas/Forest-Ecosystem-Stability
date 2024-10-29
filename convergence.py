import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json

from plant import Plant
from simulation import Simulation, StateBuffer, DataBuffer, FieldBuffer
from scipy.signal import welch

load_folder = r'Data\20241028_114615'

state_buffers = []
density_field_buffers = []
data_buffers = []
kwargs = []

end_populations = []
for idx, i in enumerate([0, 3]):
    with open(os.path.join(load_folder, f'kwargs_{i}.json'), 'r') as file:
        kwargs.append(json.load(file))

    sim_kwargs = kwargs[idx].get('sim_kwargs')
    plant_kwargs = kwargs[idx].get('plant_kwargs')

    # state_buffer_arr = pd.read_csv(
    #     f'{load_folder}/state_buffer_{i}.csv', header=None).to_numpy()
    # state_buffers.append(StateBuffer(
    #     data=state_buffer_arr, plant_kwargs=plant_kwargs))

    # density_field_buffer_arr = pd.read_csv(
    #     f'{load_folder}/density_field_buffer_{i}.csv', header=None).to_numpy()
    # density_field_buffers.append(FieldBuffer(
    #     data=density_field_buffer_arr, skip=sim_kwargs.get('density_field_buffer_skip'), sim_kwargs=sim_kwargs))

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{i}.csv', header=None).to_numpy()
    data_buffers.append(DataBuffer(data=data_buffer_arr))


for idx, i in enumerate([0, 3]):
    data = data_buffers[idx].values
    population = data[:, 2]
    window_size = 1000
    running_avg_population = np.convolve(
        population, np.ones(window_size)/window_size, mode='valid')
    running_avg_population = np.concatenate(
        (np.full(window_size//2, np.nan), running_avg_population)
    )
    running_std_population = [
        np.std(population[:j]) for j in range(len(population))
    ]

    fig, ax = plt.subplots(3, 1, figsize=(6, 6))
    ax[0].plot(running_avg_population, label='Running Average')
    ax[0].plot(population, label='Population')

    ax[1].plot(running_avg_change_population, label='Running Avg Change')

    ax[2].plot(running_std_population, label='Running Std')

    fig.legend()


plt.show()
