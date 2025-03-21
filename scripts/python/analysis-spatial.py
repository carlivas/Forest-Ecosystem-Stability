import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mods.simulation import Simulation
from mods.plant import Plant, PlantSpecies
from mods.fields import DensityFieldCustom
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer

folder = 'D:/linear_precipitation/L2000'
alias = 'lin_prec_L2000_250314_214044'
save_figs = True

time_window_size = 1000
specific_times = np.arange(11000, 25000, 50)
# time_window_size = 5
# specific_times = np.arange(5, 65, 1)

time_step = specific_times[1] - specific_times[0]
window_size = int(time_window_size // time_step)
step = max(int(window_size//4), 1)
n_steps = int(np.ceil((len(specific_times) - window_size) / step + 1))
print()
print(f'{time_window_size = }, {n_steps = }, {len(specific_times) = }, {window_size = }, {step = }')
print()
sim = Simulation(folder=os.path.abspath(folder), alias=alias)
data = sim.state_buffer.get_specific_data(specific_times)
density_field = DensityFieldCustom(half_width=sim.half_width, half_height=sim.half_height, resolution=100)
fields = np.zeros((len(specific_times), density_field.resolution, density_field.resolution))


for i, time in enumerate(specific_times):
    print(f'analysis-spatial.py: Calculating density fields (time {time}/{specific_times[-1]})', end=' '*50 + '\r')
    plants = []
    state_data = data[data['t'] == time].values
    for j in range(len(state_data)):
        id, x, y, r, species, t = state_data[j]
        id = int(id)
        species = int(species)
        
        plants.append(sim.species_list[species].create_plant(id, x, y, r))
    
    density_field.update(plants)
    fields[i] = density_field.values.reshape(density_field.resolution, density_field.resolution)
print()

def spatial_correlation(fields, i, j):
    neighbours = [
        (i-1, j), (i+1, j), (i, j-1), (i, j+1),
        (i-1, j-1), (i-1, j+1), (i+1, j-1), (i+1, j+1)
    ]
    valid_neighbours = [
        (x, y) for x, y in neighbours
        if 0 <= x < fields.shape[1] and 0 <= y < fields.shape[2]
    ]
    correlations = [
        np.corrcoef(fields[:, i, j], fields[:, x, y])[0, 1]
        for x, y in valid_neighbours
    ]
    return np.sum(correlations) / len(valid_neighbours)

variance_fields = np.zeros((n_steps, density_field.resolution, density_field.resolution))
autocorr_fields = np.zeros((n_steps, density_field.resolution, density_field.resolution))
spatialcorr_fields = np.zeros((n_steps, density_field.resolution, density_field.resolution))
stepped_times = np.zeros(n_steps)

for n in range(n_steps):
    print(f'analysis-spatial.py: Calculating step {n+1}/{n_steps}', end=' '*50 + '\r')
    S = slice(n*step, n*step + window_size)
    detrended_fields = fields[S] - np.mean(fields[S], axis=0)
    variance_fields[n] = np.var(detrended_fields, axis=0)
    
    for i in range(density_field.resolution):
        for j in range(density_field.resolution):
            autocorr_fields[n, i, j] = np.corrcoef(detrended_fields[:-1, i, j], detrended_fields[1:, i, j])[0, 1]
            spatialcorr_fields[n, i, j] = spatial_correlation(detrended_fields, i, j)
    
    stepped_times[n] = specific_times[n*step]
print()


fig, ax = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle(f'{folder}/{alias}', fontsize=12)
ax = np.array(ax).flatten()

variance_series = variance_fields.sum(axis=(1, 2)) * density_field.resolution**2
autocorr_series = autocorr_fields.sum(axis=(1, 2)) * density_field.resolution**2
spatialcorr_series = spatialcorr_fields.sum(axis=(1, 2)) * density_field.resolution**2

ax[0].plot(stepped_times, variance_series, 'k--', alpha=0.3)
ax[1].plot(stepped_times, autocorr_series, 'k--', alpha=0.3)
ax[2].plot(stepped_times, spatialcorr_series, 'k--', alpha=0.3)
for a in [ax[0], ax[1], ax[2]]:
    xlim = a.get_xlim()
    ylim = a.get_ylim()
    a.set_xlim(xlim)
    a.set_ylim(ylim)
ax[0].set_title('Integrated variance', fontsize = 8)
ax[1].set_title('Integrated autocorrelation', fontsize = 8)
ax[2].set_title('Integrated spatial correlation', fontsize = 8)

# Disable tick marks outside the update function
for a in [ax[3], ax[4], ax[5]]:
    a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

def update(n):
    print(f'analysis-spatial.py: Generating frame {n+1}/{n_steps}', end=' '*50 + '\r')
    trange = (float(stepped_times[n]), float(stepped_times[n] + time_window_size))
    
    ax[0].plot(stepped_times[:n], variance_series[:n], 'k')
    ax[1].plot(stepped_times[:n], autocorr_series[:n], 'k')
    ax[2].plot(stepped_times[:n], spatialcorr_series[:n], 'k')
    
    ax[3].imshow(variance_fields[n], cmap='Greys_r')
    ax[3].set_title(f'Variance for t = {trange}', fontsize = 8)
    
    ax[4].imshow(autocorr_fields[n], cmap='Greys_r')
    ax[4].set_title(f'Autocorrelation for t = {trange}', fontsize = 8)
    
    ax[5].imshow(spatialcorr_fields[n], cmap='Greys_r')
    ax[5].set_title(f'Spatial correlation for t = {trange}', fontsize = 8)

ani = animation.FuncAnimation(fig, update, frames=n_steps, repeat=True)
if save_figs:
    ani.save(f'{folder}/figures/{alias}_spatial_analysis.mp4', writer='ffmpeg', dpi=300)
    print(f'analysis-spatial.py: Saved animation to {folder}/figures/{alias}_spatial_analysis.mp4')
plt.show()


