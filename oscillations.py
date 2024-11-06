import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json
import scipy

from plant import Plant
from simulation import Simulation, StateBuffer, DataBuffer, FieldBuffer

load_folder = r'Data\lq_rc_ensemble'
sim_nums = [int(f.split('.')[0].replace('data_buffer_', ' '))
            for f in os.listdir(load_folder) if 'data_buffer_' in f]

new_sim_nums = []
for i, n in enumerate(sim_nums):
    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()

    if data_buffer_arr.shape[0] == 10000 and n not in [1031, 1045, 1059, 3010, 5029, 5043, 5057, 5071]:
        new_sim_nums.append(n)

sim_nums = new_sim_nums


def plot_fft(ax, data, color, label):
    freqs = scipy.fft.fftfreq(len(data), 1)
    periods = 1 / freqs
    amplitudes = np.abs(scipy.fft.fft(data))
    for i in range(1, int(5000/299.9)):
        ax.axvline(x=i*299.9, color='grey', linestyle='--', linewidth=0.5)
    ax.plot(periods, amplitudes, '.', color=color, label=label)
    ax.set_xlim(left=0, right=5000)
    ax.set_ylabel('Amplitude')
    ax.legend(loc='lower right')


transient_cutoff = 5000
for i, n in enumerate(sim_nums[::-1]):
    with open(os.path.join(load_folder, f'kwargs_{n}.json'), 'r') as file:
        kwargs = json.load(file)

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
    data_buffer = DataBuffer(data=data_buffer_arr)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.subplots_adjust(hspace=0.0)

    plot_fft(ax[0], data_buffer.values[transient_cutoff:, 2],
             'green', '|FFT| of biomass')
    plot_fft(ax[1], data_buffer.values[transient_cutoff:, 1],
             'teal', '|FFT| of population')

    ax[1].set_xlabel('Period (iterations)')

    fig.suptitle(f'|FFT| of biomass and population for sim {n}')
    fig.tight_layout()

plt.show()
