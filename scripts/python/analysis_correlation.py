import numpy as np
import json
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.signal import correlate2d, detrend
from matplotlib.animation import FuncAnimation

from mods.simulation import Simulation, sim_from_data
from mods.buffers import StateBuffer, rewrite_state_buffer_data
from mods.fields import getDensity
from mods.postprocessing import variance, autocorrelation, spatial_correlation

np.random.seed(0)  # For reproducibility

load_folder = 'Data/temp/memory_test'
surfix = 'bs10'

# biomass_buffer_df = pd.read_csv(f'{load_folder}/biomass_{surfix}.csv', header=0)
# data_buffer_df = pd.read_csv(f'{load_folder}/data_buffer_{surfix}.csv', header=0)
density_field_buffer_df = pd.read_csv(
    f'{load_folder}/density_field_buffer_{surfix}.csv', header=None)

kwargs = json.load(open(f'{load_folder}/kwargs_{surfix}.json'))
# size_buffer_df = pd.read_csv(f'{load_folder}/size_buffer_{surfix}.csv', header=0)
state_buffer_df = pd.read_csv(
    f'{load_folder}/state_buffer_{surfix}.csv', header=None)


times = state_buffer_df.iloc[:, -1].unique()
time_step = times[1] - times[0]

sim = sim_from_data(state_buffer_df=state_buffer_df,
                    density_field_buffer_df=density_field_buffer_df,
                    times_to_load=times,
                    **kwargs)


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


ani = FuncAnimation(fig, animate, frames=len(
    times), interval=10 * time_step, repeat=True)
plt.show()
