import numpy as np
import json
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, detrend
from matplotlib.animation import FuncAnimation

from mods.simulation import Simulation, sim_from_data, load_sim_data
from mods.buffers import StateBuffer, rewrite_state_buffer_data
from mods.fields import getDensity
from mods.postprocessing import variance, autocorrelation, spatial_correlation

np.random.seed(0)  # For reproducibility

load_folder = 'Data/temp/memory_test'
surfix = 'bs5'

sim_data = load_sim_data(load_folder, surfix)
sim = sim_from_data(sim_data,
                    times_to_load='all')


fields = sim.density_field_buffer.get_fields()
detrended_fields = detrend(fields, axis=0, type='linear')
variances = variance(fields)
autocorrelations = autocorrelation(fields, lag = 1)
spatial_correlations = spatial_correlation(fields)

fig, ax = plt.subplots(2, 3)
fig.suptitle(f'{surfix = }')

def animate(i):
    for a in ax.flatten():
        a.clear()
        a.axis('off')
    ax[0, 0].imshow(fields[i], cmap='Greys', interpolation='nearest')
    ax[0, 0].set_title(f'Time: {times[i]:.2f}')

    ax[0, 1].imshow(detrended_fields[i], cmap='Greys',interpolation='nearest')
    ax[0, 1].set_title('Detrended')

    ax[1, 0].imshow(variances, cmap='Greys', interpolation='nearest')
    ax[1, 0].set_title('Variance')

    ax[1, 1].imshow(autocorrelations, cmap='Greys', interpolation='nearest')
    ax[1, 1].set_title('Autocorrelation')

    ax[1, 2].imshow(spatial_correlations, cmap='Greys', interpolation='nearest')
    ax[1, 2].set_title('Spatial correlation')
    return ax

times = sim.density_field_buffer.times
time_step = times[1] - times[0]
ani = FuncAnimation(fig, animate, frames=len(
    times), interval=10 * time_step, repeat=True)
plt.show()
