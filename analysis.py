import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json

from plant import Plant
from simulation import Simulation, StateBuffer, DataBuffer, FieldBuffer
from scipy.signal import welch

load_folder = r'Data\20241029_100928'
n_files = len([f for f in os.listdir(load_folder) if 'data_buffer' in f])

state_buffers = []
density_field_buffers = []
data_buffers = []
kwargs = []

end_populations = []
for i in range(n_files):
    with open(os.path.join(load_folder, f'kwargs_{i}.json'), 'r') as file:
        kwargs.append(json.load(file))

    sim_kwargs = kwargs[i].get('sim_kwargs')
    plant_kwargs = kwargs[i].get('plant_kwargs')

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

    population_data = data_buffers[i].buffer[:, 2]
    end_population = population_data[~np.isnan(population_data)][-1]
    end_populations.append(end_population)

fig1, ax1 = plt.subplots()
norm = plt.Normalize(vmin=min(end_populations), vmax=max(end_populations))
for i in range(n_files):
    x = kwargs[i].get('sim_kwargs').get('land_quality')
    y = kwargs[i].get('plant_kwargs').get('species_germination_chance')

    color = plt.cm.ScalarMappable(
        norm=norm, cmap='coolwarm').to_rgba(end_populations[i])
    ax1.scatter(x, y, color=color)

ax1.set_xlabel('land quality', fontsize=8)
ax1.set_ylabel('germination chance', fontsize=8)
ax1.set_title(
    'End population vs. land quality and germination chance', fontsize=10)
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='coolwarm'), ax=ax1)
cbar.set_label('End population')

ax1.tick_params(axis='both', which='major', labelsize=7)
cbar.ax.tick_params(labelsize=7)


fig2, ax2 = plt.subplots()
for i in range(n_files):
    if end_populations[i] > 0:
        # Replace with your actual data extraction logic
        data = data_buffers[i].buffer[:, 2]
        # Remove nan values from data
        data[np.isnan(data)] = 0

        # Set nperseg to a value less than or equal to the length of the data
        nperseg = len(data)
        # Set noverlap to a value less than nperseg
        noverlap = nperseg//10
        fs = 1  # Sampling frequency, adjust as needed

        f, Pxx = welch(data,
                       fs=fs, nperseg=nperseg, noverlap=noverlap)

        ax2.plot(f, Pxx, '--.', label=f'Simulation {i}')
        ax2.set_xlabel('Frequency', fontsize=8)
        ax2.set_ylabel('Power Spectral Density', fontsize=8)
        ax2.set_title('Power Spectral Density of Population Data', fontsize=10)
        ax2.legend(fontsize=7)
        ax2.tick_params(axis='both', which='major', labelsize=7)

plt.close(fig1)
plt.show()
