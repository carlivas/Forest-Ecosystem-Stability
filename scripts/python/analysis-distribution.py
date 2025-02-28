import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import json

from mods.buffers import StateBuffer
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * x ** b

folder = 'Data/parameter_shift/L1000_shifted/maturity_size' # Path to the folder containing the buffers
save_fig = True

if not os.path.exists(folder):
    raise FileNotFoundError(f"The specified path does not exist: {folder}")

load_folder = os.path.abspath(folder)
print(f'load_folder: {load_folder}')

for root, dirs, files in os.walk(load_folder):
    aliases = [f.split('-')[-1].split('.')[0] for f in files if 'kwargs' in f]
    aliases = [s for s in aliases if 'checkpoint' not in s]
    # aliases = sorted(aliases, key=lambda x: int(x.split('-')[-1]))
    if not aliases:
        continue
    print(f'aliases: {aliases}')
    
    for i, alias in enumerate(aliases):
        kwargs = pd.read_json(f'{root}/kwargs-{alias}.json', typ='series').to_dict()
        state_buffer = StateBuffer(file_path=f'{root}/state_buffer-{alias}.csv')
        state_buffer_df = state_buffer.get_data()
        
        skip = min(100, len(state_buffer_df['t'].unique()) // 100)
        bins = 50
        times = state_buffer_df['t'].unique()[::skip]
        # times = [t for t in times if t > 300]
        timestep = times[1] - times[0]
        
        hist_list = []
        bins_list = []
        fit_params_list = []

        for i in range(len(times)):
            t = times[i]
            hist, bins = np.histogram(state_buffer_df[state_buffer_df['t'] == t]['r']*kwargs['L'], bins=100)
            hist_list.append(hist)
            bins_list.append(bins)
            
            if hist.sum() < 100:
                print(f'hist.sum() < 100 at t={t}', end='\r')
                fit_params_list.append([np.nan, np.nan])
                continue
            # Fit the histogram values with a power law function
            bin_centers = (bins[:-1] + bins[1:]) / 2
            try:
                popt, _ = curve_fit(power_law, bin_centers, hist, p0=[1, -1], maxfev=10000)
            except RuntimeError:
                popt = [np.nan, np.nan]
            
            fit_params_list.append(popt)
            print(f'Calculating histograms and fits: {i = }/{len(times)} ({i/len(times)*100:.2f}%)', end='\r')
        print()
        
        y_min = min(hist.min() for hist in hist_list[int(300/timestep):])
        y_max = max(hist.max() for hist in hist_list[int(300/timestep):])
        x_min = min(bins.min() for bins in bins_list[int(300/timestep):])
        x_max = max(bins.max() for bins in bins_list[int(300/timestep):])
        
        a_mean = np.mean([a for a, b in fit_params_list[int(300/timestep):] if not np.isnan(a)])
        a_std = np.std([a for a, b in fit_params_list[int(300/timestep):] if not np.isnan(a)])
        b_mean = np.mean([b for a, b in fit_params_list[int(300/timestep):] if not np.isnan(b)])
        b_std = np.std([b for a, b in fit_params_list[int(300/timestep):] if not np.isnan(b)])
        
        a_min = a_mean - 5*a_std
        a_max = a_mean + 5*a_std
        b_min = b_mean - 5*b_std
        b_max = b_mean + 5*b_std
        
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

        def update_hist(i):
            t = times[i]
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            H = hist_list[i]
            B = bins_list[i]
            width = B[1] - B[0]
            ax1.bar(B[:-1], H, width=width, align='edge', color='g')
            
            # Plot the power law fit
            bin_centers = (B[:-1] + B[1]) / 2
            a, b = fit_params_list[i]
            ax1.plot(bin_centers, power_law(bin_centers, a, b), 'r-', label=f'Power law fit: a={a:.2f}, b={b:.2f}')
            
            ax1.set_title(f'Size distribution at time {t}')
            ax1.set_xlabel('Size [m]')
            ax1.set_ylabel('Frequency')
            ax1.set_ylim(y_min, y_max)
            ax1.set_xlim(x_min, x_max)
            ax1.legend()
            
            # Plot a and b over time
            times_so_far = times[:i+1]
            a_values = [fit_params_list[j][0] for j in range(i+1)]
            b_values = [fit_params_list[j][1] for j in range(i+1)]
            
            ax2.plot(times_so_far, a_values, 'b-', label='a over time')
            ax2.set_title('a over time')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('a')
            ax2.set_xlim(0, times[-1])
            ax2.set_ylim(a_min, a_max)
            ax2.legend()
            
            ax3.plot(times_so_far, b_values, 'm-', label='b over time')
            ax3.set_title('b over time')
            ax3.set_xlabel('Time')
            ax3.set_ylabel('b')
            ax3.set_xlim(0, times[-1])
            ax3.set_ylim(b_min, b_max)
            ax3.legend()
            
            print(f'Animating histograms and fits: {i = }/{len(times)} ({i/len(times)*100:.2f}%)', end='\r')
        print('')
        
        ani_hist = FuncAnimation(fig, update_hist, frames=len(times), repeat=False, interval=timestep)
        if save_fig:
            ani_hist.save(f'{root}/figures/size_dist-{alias}.mp4', writer='ffmpeg')
        else:
            plt.show()