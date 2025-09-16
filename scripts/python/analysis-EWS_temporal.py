import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mods.utilities import get_unique_aliases

def load_data(input_folder):
    """Load data and filter based on precipitation difference."""
    aliases = get_unique_aliases(input_folder)
    kwargs_list, data_buffer_list, alias_list = [], [], []

    for alias in aliases:
        kwargs_path = os.path.join(input_folder, f'kwargs-{alias}.json')
        data_buffer_path = os.path.join(input_folder, f'data_buffer-{alias}.csv')

        if os.path.exists(kwargs_path) and os.path.exists(data_buffer_path):
            kwargs = pd.read_json(kwargs_path, typ='series').to_dict()
            data_buffer = pd.read_csv(data_buffer_path)
            dp = data_buffer['Precipitation'].iloc[200] - data_buffer['Precipitation'].iloc[199]
            
            if not np.isclose(dp, -4e-6, atol=1e-5):
                continue
            init_biomass = data_buffer['Biomass'].iloc[300]
            if not init_biomass < 0.4:
                continue
            if data_buffer['Biomass'].iloc[0] == 0:
                data_buffer.drop(index=0, inplace=True)
            data_buffer.reset_index(drop=True, inplace=True)
            kwargs_list.append(kwargs)
            data_buffer_list.append(data_buffer)
            alias_list.append(alias)
        
    # Order triplets in order of run number
    data = [(kwargs_list[i], data_buffer_list[i], alias_list[i]) for i in range(len(data_buffer_list))]

    return data

def filter_data(kwargs_list, data_buffer_list, alias_list, density_scheme):
    kwargs_list_filtered, data_buffer_list_filtered, alias_list_filtered = [], [], []
    for i, (kwargs, data_buffer, alias) in enumerate(zip(kwargs_list, data_buffer_list, alias_list)):
        if kwargs.get('density_scheme', None) is None:
            kwargs['density_scheme'] = 'local'
        
        if kwargs['density_scheme'] == density_scheme:
            kwargs_list_filtered.append(kwargs)
            data_buffer_list_filtered.append(data_buffer)
            alias_list_filtered.append(alias)
    return kwargs_list_filtered, data_buffer_list_filtered, alias_list_filtered

def detrend_cubic(x, y):
    """Detrend data using a cubic polynomial fit."""
    coeffs = np.polyfit(x, y, 3)
    poly = np.poly1d(coeffs)
    detrended_data = y - poly(x)
    
    # fig, ax = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
    # ax[0].plot(x, y , color='k')
    # ax[0].plot(x, poly(x), color='r', label='Cubic Fit')
    # ax[1].plot(x, detrended_data, color='k')
    # plt.show()
    return detrended_data

def estimate_tipping_point(run_data, window_size = 1500, step_size = 1):
    kwargs, db, alias = run_data
    time = db['Time'].values
    biomass = db['Biomass'].values
    dt = time[1] - time[0]
    
    biomass_diff = []
    window_starts = np.arange(500, time[-1], step_size).astype(int)
    window_centers = window_starts + window_size//2
    for start in window_starts:
        if start + window_size > time[-1]:
            break
        
        dB = abs(biomass[start+window_size] - biomass[start])
        dB /= window_size * dt
        biomass_diff.append(dB)
    biomass_diff = np.array(biomass_diff)
    
    where_max_diff = np.argmax(biomass_diff)
    tp_time = time[window_centers[where_max_diff]]
    tp_biomass = biomass[window_centers[where_max_diff]]
    
    return tp_time, tp_biomass

def plot_run_data(ax, run_data, color='k', label=None, alpha=0.1, tipping_points=False):
    kwargs, db, alias = run_data
    time = db['Time'].values
    biomass = db['Biomass'].values
    
    # plot the biomass data for each run
    ax.plot(time, biomass, color=color, alpha=alpha, label=label)
    
    # plot the tipping point time and biomass
    if tipping_points:            
        tp_time, tp_biomass = estimate_tipping_point(run_data)
        ax.plot(tp_time, tp_biomass, 'ro', label='Tipping Point')
    
    # ax.set_ylabel('Biomass')

# def plot_run_statistics(axs, run_stats):
#     # fig, ax = plt.subplots(3, 1, figsize=(8,6), sharex=True)
    
#     ax[0].plot(run_stats['variance'])
#     ax[0].set_ylabel('Variance')
    
#     ax[1].plot(run_stats['skewness'])
#     ax[1].set_ylabel('Skewness')
    
#     ax[2].plot(run_stats['autocor1'])
#     ax[2].set_ylabel('Autocorrelation (lag=1)')
#     ax[-1].set_xlabel('Time')
    

# def plot_biomass(ax, biomass_data, time_slice, color, alpha, labels=None):
#     """Plot biomass data for each run."""
#     for i, col in enumerate(biomass_data.columns):
#         if labels is not None:
#             label = labels[i]
#         else:
#             label = col
#         ax.plot(biomass_data.index[time_slice], biomass_data[col].iloc[time_slice], label=label, color=color, alpha=alpha)
#     ax.set_ylabel('Biomass')
#     ax.legend(fontsize=8)
#     ax.grid(True)
#     ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
    
    
    

def calculate_statistics(run_data, time_slice, window_size):
    run_stats = pd.DataFrame()
    
    kwargs, db, alias = run_data
    density_scheme = kwargs['density_scheme']
    if density_scheme is None:
        density_scheme = 'local'
    
    run_stats = pd.DataFrame()
    time = db['Time'].values
    biomass = db['Biomass'].values
    # define rolling windows for statistics
    window_starts = np.arange(time_slice.start, time_slice.stop, time_slice.step if time_slice.step else 1)
    window_centers = window_starts + window_size//2
        
    for start, index in zip(window_starts, window_centers):
        if start + window_size > time_slice.stop:
            break
        
        window_time = time[start:start + window_size]
        window_biomass = biomass[start:start + window_size]
        if len(window_time) == 0:
            print(f'Empty window: {start}')
            continue
        # detrend data for each run in windows using cubic spline
        detrended_window = detrend_cubic(window_time, window_biomass)
        
        
        # plot the detrended data for each run
        # fig,ax = plt.subplots(1, 1, figsize=(6, 3))
        # ax.plot(window_time, detrended_window, color='k', label='Detrended')
        # plt.show()
        
        var = pd.Series(detrended_window).var()
        skew = pd.Series(detrended_window).skew()
        ac1 = pd.Series(detrended_window).autocorr(lag=1) if var != 0 else np.nan
        # calculate the variance, skewness and autocorrelation1 for detrended window
        run_stats.at[index, f'variance_{density_scheme}_{alias}'] = var
        run_stats.at[index, f'skewness_{density_scheme}_{alias}'] = skew
        run_stats.at[index, f'autocor1_{density_scheme}_{alias}'] = ac1
        run_stats.at[index, f'biomass_{density_scheme}_{alias}'] = biomass[index]
    run_stats.index.name = 'time'
    return run_stats       
    

def aggregate_statistics(stats):
    """Aggregate statistics across runs."""
    stats_all = pd.DataFrame()
    stats_agg = pd.DataFrame()
    
    for i, run_stats in enumerate(stats):
        stats_all = pd.concat([stats_all, run_stats], axis=1)
    
    
    for density_scheme in ['local', 'global']:
        for stat in ['biomass', 'variance', 'skewness', 'autocor1']:
            like = f'{stat}_{density_scheme}'
            
            columns_filtered = stats_all.filter(like=like).columns
            stats_filtered = stats_all[columns_filtered]            
            stats_agg[f'{stat}_mean_{density_scheme}'] = stats_filtered.mean(axis=1)
            stats_agg[f'{stat}_std_{density_scheme}'] = stats_filtered.std(axis=1)
    
    stats_agg.index.name = 'time'
    return stats_agg

def plot_rolling_stats(rolling_stats, colors, zorder):
    rolling_stats = rolling_stats.apply(pd.to_numeric, errors='coerce')  # Ensure numeric types
    fig, ax = plt.subplots(4, 2, figsize=(6, 8))
    for d, density_scheme in enumerate(['local', 'global']):
        ax[0, d].set_title(f'{density_scheme} model')
        ax[-1, d].set_xlabel('Time')
        for s, stat in enumerate(['biomass', 'variance', 'skewness', 'autocor1']):
            color = colors[s]
            mean = rolling_stats[f'{stat}_mean_{density_scheme}']
            std = rolling_stats[f'{stat}_std_{density_scheme}']
            ax[s, d].plot(rolling_stats.index, mean, color=color, label=stat, zorder=zorder)
            ax[s, d].fill_between(rolling_stats.index, mean - std, mean + std, color=color, alpha=0.2, zorder=zorder)
            ax[s, d].legend(loc='lower left')
            ax[s, d].grid(True)
            ax[s, d].ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
            # ax[s, 0].set_ylabel(stat)
            
            if stat != 'biomass':
                # slic = slice(500, mean.index[-1] -500)
                slic = slice(0, rolling_stats.index[-1])
                
                ylim = (np.mean(mean[slic]) - 6 * np.std(mean[slic]),
                        np.mean(mean[slic]) + 6 * np.std(mean[slic]))
                if not np.isnan(ylim).any():
                   ax[s, d].set_ylim(ylim)           
                    
            if d > 0:
                ylims = [ax[s, d].get_ylim(), ax[s, 0].get_ylim()]
                max_ylim = max(ylims, key=lambda x: x[1] - x[0])
                ax[s, d].sharey(ax[s, 0])
                ax[s, d].set_ylim(max_ylim)
            if s > 0:
                ax[s, d].sharex(ax[0, d])
            if s < len(ax) - 1:
                ax[s, d].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            if d > 0:
                ax[s, d].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        
        # ax[i, j].set_yticks([])
        # xlims = [ax[i, j].get_xlim() for i in range(2)]
        # max_xlim = max(xlims, key=lambda x: x[1] - x[0])
        # for i in range(2):
        #     ax[i, j].set_ylim(max_xlim)
    # if i > 0:
    #     ax[i, j].sharex(ax[0, j])
    #     max_ylim = max(ylims, key=lambda x: x[1] - x[0])
    #     for j in range(2):
    #         ax[i, j].set_ylim(max_ylim)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.15, wspace=0.15)
        
    return fig, ax
                    
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.titlesize': 12})
plt.rcParams.update({'axes.labelsize': 12})
plt.rcParams.update({'xtick.labelsize': 12})
plt.rcParams.update({'ytick.labelsize': 12})
plt.rcParams.update({'legend.fontsize': 12})
plt.rcParams.update({'figure.titlesize': 12})
plt.rcParams.update({'axes.formatter.useoffset': False})
plt.rcParams.update({'axes.formatter.use_mathtext': True})
plt.rcParams.update({'axes.formatter.limits': (-3, 3)})

generate_data = False
save_figs = False
input_folder = r'C:\Users\carla\Dropbox\_CARL\UNI\KANDIDAT\PROJEKT\LaTeX\Figures\Data\lin_prec_L2000'
output_folder = r'C:\Users\carla\Dropbox\_CARL\UNI\KANDIDAT\PROJEKT\LaTeX\Figures'
window_size = 2500
step_size = 1
analysis_start = 500
time_slice = slice(analysis_start, -1, step_size)

# IMPORT THE SIMULATION DATA
sim_data = load_data(input_folder)
print(f'{len(sim_data) = }')
if generate_data:    
    print('\nGenerating data...')
    stats = []
    tipping_points = []
    biomass_data = pd.DataFrame()
    for i, run_data in enumerate(sim_data):
        print(f'Processing run {i+1}/{len(sim_data)}:', run_data[2])
        # calculate the estimated time of tipping (greatest slope)
        tp_time, tp_biomass = estimate_tipping_point(run_data)
        tipping_points.append((tp_time, tp_biomass))
                
        # CALCULATE STATISTISCS FOR EACH RUN
        time_slice_temp = slice(analysis_start, int(tp_time), step_size)
        run_stats = calculate_statistics(run_data, time_slice_temp, window_size)
        stats.append(run_stats)

    # biomass_data = pd.DataFrame(biomass_data)
    # # CALCULATE AGGREGATED STATISTICS
    # biomass_stats = aggregate_biomass(biomass_data)
    stats_agg = aggregate_statistics(stats)
    pd.DataFrame(stats_agg).to_csv(os.path.join(output_folder, 'Data\EWS_temporal.csv'), index=True)
else:
    
    stats_agg = pd.read_csv(os.path.join(output_folder, 'Data\EWS_temporal.csv'), index_col=0)

print(f'{stats_agg.index = }')
# PLOT THE AGGREGATED STATISTICS AND 
fig, axs = plot_rolling_stats(stats_agg, ['g', 'r', 'm', 'b'], zorder=2)

for run_data in sim_data:
    if run_data[0]['density_scheme'] == 'local':
        plot_run_data(axs[0, 0], run_data, color='k')
    else:
        plot_run_data(axs[0, 1], run_data, color='k')
for ax in axs.flatten():
    ylim = ax.get_ylim()
    ax.fill_betweenx(ylim, analysis_start, color='r', alpha=0.2, zorder=0, edgecolor=None)
    ax.fill_betweenx(ylim, analysis_start, analysis_start+window_size, color='k', alpha=0.2, zorder=1, edgecolor=None)
    ax.set_xlim(0, 12500)
    ax.grid(True, zorder=0)

if save_figs:
    fig.savefig(os.path.join(output_folder, 'EWS_temporal.png'), dpi=300, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(output_folder, "EWS_temporal.png")}')
else:
    plt.show()
