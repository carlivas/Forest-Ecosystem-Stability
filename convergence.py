import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json

from plant import Plant
from simulation import Simulation, StateBuffer, DataBuffer, FieldBuffer
from scipy.signal import welch

load_folder = r'Data\lq_rc_ensemble_n100'

state_buffers = []
density_field_buffers = []
data_buffers = []
kwargs = []
end_populations = []

sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'data_buffer' in f]

for idx, i in enumerate(sim_nums):
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

    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    ax[0].plot(population, 'g', label='Population', alpha=0.5)
    ax[0].plot(running_avg_population, 'k--', label='Running Average')

    ax[1].plot(running_std_population, 'r', label='Running Std')

    fig.legend()
    fig.suptitle(f'Sim {i}')

    plt.show()
