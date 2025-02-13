import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.optimize import curve_fit
from scipy.stats import f

# Exponential function for fitting
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

load_folder = 'D:/695774818_finished'

if not os.path.exists(load_folder):
    raise FileNotFoundError(f"The specified path does not exist: {load_folder}")

load_folder = os.path.abspath(load_folder)
print(f'load_folder: {load_folder}')

for root, dirs, files in os.walk(load_folder):
    surfixes = [f.split('_')[-1].split('.')[0] for f in files if 'kwargs' in f]
    surfixes = sorted(surfixes, key=lambda x: int(x.split('-')[-1]))
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
    biomass = pd.concat([data_buffer['Biomass'] for data_buffer in data_buffer_list], axis=0).reset_index(drop=True)
    # population = pd.concat([data_buffer['Population'] for data_buffer in data_buffer_list], axis=0).reset_index(drop=True)
    precipitation = pd.concat([pd.DataFrame(kwargs['precipitation'] * np.ones_like(data_buffer['Time'])) for kwargs, data_buffer in zip(kwargs_list, data_buffer_list)], axis=0).reset_index(drop=True)
    time_prec = precipitation.index
    
    time_range = (20_000, 220_000)
    time_slice = slice(*time_range)
    
    window_size = 5000
    step = window_size
    
    # Detrend biomass data
    biomass_analysis = biomass.iloc[time_slice]
    biomass_analysis = biomass_analysis - biomass_analysis.rolling(window=window_size, center=True).mean()
    biomass_analysis = biomass_analysis.dropna()
    time_analysis = time.iloc[biomass_analysis.index]

    biomass_var = biomass_analysis.rolling(window=window_size, step=step, center=True).var().dropna()
    time_var = time.iloc[biomass_var.index]
    biomass_var_fit, p_value_var = calculate_fit_and_p_value(pow_func, time_var, biomass_var)

    biomass_ac = biomass_analysis.rolling(window=window_size, step=step, center=True).apply(lambda x: x.autocorr(lag=1)).dropna()
    time_ac = time.iloc[biomass_ac.index]
    biomass_ac_fit, p_value_ac = calculate_fit_and_p_value(pow_func, time_ac, biomass_ac)



    fig, ax = plt.subplots(5, 1, figsize=(8, 9), sharex=True)
    ax[0].plot(time, biomass, color='tab:blue')
    ax[0].set_ylabel('Biomass')

    ax[1].plot(time_prec, precipitation, color='tab:green')
    ax[1].set_ylabel('Precipitation')

    ax[2].plot(time_analysis, biomass_analysis, color='tab:blue')
    ax[2].set_ylabel('Detrended Biomass')

    ax[3].plot(time_var, biomass_var_fit, color='black', label=f'p-value: {p_value_var:.2e}')
    ax[3].plot(time_var, biomass_var, color='tab:red', marker='o', markersize=5, ls='', markerfacecolor='none')
    ax[3].set_ylabel('Biomass Variance')
    ax[3].set_xlabel('Time')
    # ax[3].set_ylim(1e-6, 3e-5)
    
    xlim = np.array(ax[3].get_xlim())
    ylim = np.array(ax[3].get_ylim())
    xOff = xlim.min() + (xlim.max() - xlim.min()) * 0.05
    yOff = ylim.min() + (ylim.max() - ylim.min()) * 0.9
    ax[3].errorbar(xOff, yOff, xerr=window_size, fmt='', color='black', ecolor='black', capsize=5, capthick=2)
    ax[3].legend()

    ax[4].plot(time_ac, biomass_ac_fit, color='black', label=f'p-value: {p_value_ac:.2e}')
    ax[4].plot(time_ac, biomass_ac, color='tab:purple', marker='o', markersize=5, ls='', markerfacecolor='none')
    ax[4].set_ylabel('Autocorrelation')
    ax[4].set_xlabel('Time')
    ax[4].legend()
    
    for a in ax:
        a.axvline(time_range[0], color='black', ls='--', alpha = 0.5)
        a.axvline(time_range[1], color='black', ls='--', alpha = 0.5)
    
    plt.show()
    
