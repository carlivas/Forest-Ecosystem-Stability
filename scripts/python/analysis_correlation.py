import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.stats import f

def lin_func(x, a, b):
    return a * x + b

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def pow_func(x, a, b, c):
    return a * x ** b + c

def calculate_fit_and_p_value(fit_func, x, y):
    popt, _ = curve_fit(fit_func, x, y, maxfev=int(1e6))
    y_fit = fit_func(x, *popt)
    
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    df_model = len(popt) - 1
    df_resid = len(y) - len(popt)
    f_stat = (r_squared / df_model) / ((1 - r_squared) / df_resid)
    p_value = 1 - f.cdf(f_stat, df_model, df_resid)
    
    return y_fit, p_value

folder = '../../Data/linear_precipitation/L500'
key = 'Biomass'
save_fig = True

if not os.path.exists(folder):
    raise FileNotFoundError(f"The specified path does not exist: {folder}")

load_folder = os.path.abspath(folder)
print(f'load_folder: {load_folder}')

for root, dirs, files in os.walk(load_folder):
    surfixes = [f.split('_')[-1].split('.')[0] for f in files if 'kwargs' in f]
    surfixes = [s for s in surfixes if 'checkpoint' not in s]
    # surfixes = sorted(surfixes, key=lambda x: int(x.split('-')[-1]))
    if not surfixes:
        continue
    print(f'surfixes: {surfixes}')
    
    max_num_plants = 0
    kwargs_list = []
    data_buffer_list = []
    for i, n in enumerate(surfixes):
        kwargs = pd.read_json(
            f'{root}/kwargs_{n}.json', typ='series').to_dict()
        data_buffer = pd.read_csv(f'{root}/data_buffer_{n}.csv')
        data_buffer_list.append(data_buffer)
        kwargs_list.append(kwargs)
    
    time = pd.concat([data_buffer['Time'] for data_buffer in data_buffer_list], axis=0).reset_index(drop=True)
    data = pd.concat([data_buffer[key] for data_buffer in data_buffer_list], axis=0).reset_index(drop=True)
    precipitation = pd.concat([data_buffer['Precipitation'] for data_buffer in data_buffer_list], axis=0).reset_index(drop=True)
    time_prec = precipitation.index

    time_range = (1000, 27500)
    time_slice = slice(*time_range)
    
    window_size = 1000
    step = window_size//2
    
    data_analysis = data.iloc[time_slice].dropna()
    
    # # Detrend data
    # data_analysis = data_analysis - data_analysis.rolling(window=window_size, center=True).mean()
    
    time_analysis = time.iloc[data_analysis.index]
    
    data_var = data_analysis.rolling(window=window_size, step=step, center=True).var().dropna()
    time_var = time.iloc[data_var.index]
    data_var_fit, p_value_var = calculate_fit_and_p_value(lin_func, time_var, data_var)

    data_ac = data_analysis.rolling(window=window_size, step=step, center=True).apply(lambda x: x.autocorr(lag=1)).dropna()
    time_ac = time.iloc[data_ac.index]
    data_ac_fit, p_value_ac = calculate_fit_and_p_value(lin_func, time_ac, data_ac)



    fig, ax = plt.subplots(2, 2, figsize=(12, 4), sharex=True)
    ax = ax.T.flatten()
    ax[0].plot(time, data, color='tab:blue')
    ax[0].set_ylabel(key)

    ax[1].plot(time_prec, precipitation, color='tab:green')
    ax[1].set_ylabel('Precipitation')

    ax[2].plot(time_var, data_var_fit, color='black', label=f'p-value: {p_value_var:.2e}')
    ax[2].plot(time_var, data_var, color='tab:red', marker='o', markersize=5, ls='', markerfacecolor='none')
    ax[2].set_ylabel(key + ' Variance')
    ax[2].set_xlabel('Time')
    
    xlim = np.array(ax[2].get_xlim())
    ylim = np.array(ax[2].get_ylim())
    xOff = xlim.min() + (xlim.max() - xlim.min()) * 0.05
    yOff = ylim.min() + (ylim.max() - ylim.min()) * 0.8
    ax[2].errorbar(xOff, yOff, xerr=window_size, fmt='', color='black', ecolor='black', capsize=5, capthick=1.5)
    ax[2].legend()

    ax[3].plot(time_ac, data_ac_fit, color='black', label=f'p-value: {p_value_ac:.2e}')
    ax[3].plot(time_ac, data_ac, color='tab:purple', marker='o', markersize=5, ls='', markerfacecolor='none')
    ax[3].set_ylabel('Autocorrelation')
    ax[3].set_xlabel('Time')
    
    xlim = np.array(ax[3].get_xlim())
    ylim = np.array(ax[3].get_ylim())
    xOff = xlim.min() + (xlim.max() - xlim.min()) * 0.05
    yOff = ylim.min() + (ylim.max() - ylim.min()) * 0.8
    ax[3].errorbar(xOff, yOff, xerr=window_size, fmt='', color='black', ecolor='black', capsize=5, capthick=1.5)
    ax[3].legend()
    
    for a in ax:
        a.axvline(time_range[0], color='black', ls='--', alpha = 0.5)
        a.axvline(time_range[1], color='black', ls='--', alpha = 0.5)

    sim_name = root.split('/')[-1]
    
    if len(surfixes) == 1:
        sim_name += '_' + surfixes[0]
        
    fig.suptitle(f'{key} Correlation {sim_name}')
    fig.tight_layout()
    
    if save_fig:
        fig_path = f'{root}/figures/_{key.lower()}_correlation_{sim_name}.png'
        fig.savefig(fig_path, dpi=600)
plt.show()
    
