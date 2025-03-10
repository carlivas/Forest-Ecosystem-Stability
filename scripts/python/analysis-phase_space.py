import numpy as np
import matplotlib.pyplot as plt
import json
import os

from mods.simulation import Simulation
from scipy.interpolate import interp1d
import matplotlib.animation as animation

save_animation = True

load_folder = 'Data/dynamics' # Path to the folder containing the buffers
aliases = [f.split('-')[-1] for f in os.listdir(load_folder) if 'kwargs-' in f]
aliases = [f.replace('.json', '') for f in aliases]
aliases = [f.replace('.csv', '') for f in aliases]
print(f'{aliases = }')

for i, alias in enumerate(aliases):
    sim = Simulation(folder=load_folder, alias=alias)


    skip = 10
    window_size = 200
    key = 'Biomass'
    data = sim.data_buffer.get_data()[key]
    data_mean = np.convolve(data, np.ones(
        window_size)/window_size, mode='valid')
    data_diff = np.diff(data_mean)
    data_mean = data_mean
    data_diff = data_diff

    time_step = sim.time_step
    time = np.arange(0, len(data)*time_step, time_step)

    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    def animate(i):
        idx = i*skip
        ax.clear()

        nbins = 5
        time_bins = np.linspace(max(0,idx - 1000), idx, nbins+1, dtype=int)
        lines = []
        for j in range(nbins):
            t = (j+1)/(nbins)
            line, = ax.plot(data_mean[time_bins[j]:time_bins[j+1]],
                            data_diff[time_bins[j]:time_bins[j+1]], 'g-', lw=t, alpha=t)
            lines.append(line)

        line1, = ax.plot(data_mean[:time_bins[0]], data_diff[:time_bins[0]], 'g-', lw=1/nbins, alpha=1/nbins)
        line2, = ax.plot(data_mean[idx], data_diff[idx], 'g.')
        lines.append(line1)
        lines.append(line2)
        
        ax.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.2)
        ax.set_xlabel(key)
        ax.set_ylabel(f'np.diff({key})')
        # ax.set_aspect(aspect=w/h)
        
        # ax[1].plot(data_mean[:idx], data_diff[:idx], 'g-')
        # ax[1].plot(data_mean[idx], data_diff[idx], 'g.')
        fig.suptitle(f'({alias}) Phase space t = {time[idx]}')
        print(f'Animating frame {i+1}/{len(data_mean[::skip])-1} ({100*(i+1)/(len(data_mean[::skip])-1):.2f}%)', end='\r')
        return lines


    anim = animation.FuncAnimation(fig, animate, frames=len(
        data_mean[::skip])-1, interval=30*1000/(len(data_mean[::skip])-1))
    if save_animation:
        os.makedirs(f'{load_folder}/figures', exist_ok=True)
        anim.save(f'{load_folder}/figures/phase_space-{alias}.mp4', writer='ffmpeg', dpi=600)
    else:
        plt.show()
