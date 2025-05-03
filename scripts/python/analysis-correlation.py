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

folder = 'Data/lin_prec_test'
key = 'Biomass'
save_fig = False

load_folder = os.path.abspath(folder)
print(f'load_folder: {load_folder}')
files = os.listdir(load_folder)
aliases = [f.split('-')[-1].split('.')[0] for f in files if 'data_buffer-' in f]
aliases = [s for s in aliases if 'checkpoint' not in s]
print(f'aliases: {aliases}')

max_num_plants = 0
kwargs_list = []
data_buffer_list = []
for i, alias in enumerate(aliases):
    kwargs = pd.read_json(
        f'{load_folder}/kwargs-{alias}.json', typ='series').to_dict()
    data_buffer = pd.read_csv(f'{load_folder}/data_buffer-{alias}.csv')

    time = data_buffer['Time']
    data = data_buffer[key]
    precipitation = data_buffer['Precipitation']
    time_prec = precipitation.index

    time_range = (1000, time_prec.max())
    time_slice = slice(*time_range)

    window_size = (time_prec.max() - time_prec.min()) // 15
    step = window_size//4

    data_analysis = data.iloc[time_slice].dropna()

    # # Detrend data
    data_analysis = data_analysis - data_analysis.rolling(window=window_size, center=True).mean()
    time_analysis = time.iloc[data_analysis.index]

    data_var = data_analysis.rolling(window=window_size, step=step, center=True).var().dropna()
    time_var = time.iloc[data_var.index]
    # data_var_fit, p_value_var = calculate_fit_and_p_value(lin_func, time_var, data_var)

    data_ac = data_analysis.rolling(window=window_size, step=step, center=True).apply(lambda x: x.autocorr(lag=1)).dropna()
    time_ac = time.iloc[data_ac.index]
    # data_ac_fit, p_value_ac = calculate_fit_and_p_value(lin_func, time_ac, data_ac)



    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax = ax.T.flatten()
    
    ax0 = ax[0]
    ax1 = ax0.twinx()
    lns1 = ax0.plot(time, data, color='tab:blue', label=key)
    ax0.set_ylabel(key)
    lns2 = ax1.plot(time_prec, precipitation, color='k', lw=2, ls='--', label='Precipitation')
    ax1.set_ylabel('Precipitation')
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax0.legend(lns, labs, loc='lower left')
    ax0.set_ylim(-data.max()*0.1, data.max()*1.1)
    ax1.set_ylim(-precipitation.max()*0.1, precipitation.max()*1.1)
    
    ax[1].plot(time_analysis, data_analysis, color='tab:blue', ls='-', markerfacecolor='white')
    ax[1].set_ylabel(f'Detrended {key}')

    # ax[2].plot(time_var, data_var_fit, color='black', label=f'p-value: {p_value_var:.2e}')
    ax[2].plot(time_var, data_var, color='tab:red', marker='o', markersize=6, ls='-', markerfacecolor='white')
    ax[2].set_ylabel('Variance')
    ax[2].set_xlabel('Time')

    xlim = np.array(ax[2].get_xlim())
    ylim = np.array(ax[2].get_ylim())
    # xOff = xlim.min() + (xlim.max() - xlim.min()) * 0.05
    yOff = ylim.min() + (ylim.max() - ylim.min()) * 0.8
    xOff = time_range[0] + window_size
    ax[2].errorbar(xOff, yOff, xerr=window_size, fmt='', color='black', ecolor='black', capsize=5, capthick=1.5)
    # ax[2].legend()

    # ax[3].plot(time_ac, data_ac_fit, color='black', label=f'p-value: {p_value_ac:.2e}')
    ax[3].plot(time_ac, data_ac, color='tab:purple', marker='o', markersize=6, ls='-', markerfacecolor='white')
    ax[3].set_ylabel('Autocorrelation')
    ax[3].set_xlabel('Time')

    xlim = np.array(ax[3].get_xlim())
    ylim = np.array(ax[3].get_ylim())
    # xOff = xlim.min() + (xlim.max() - xlim.min()) * 0.05
    yOff = ylim.min() + (ylim.max() - ylim.min()) * 0.8
    xOff = time_range[0] + window_size
    ax[3].errorbar(xOff, yOff, xerr=window_size, fmt='', color='black', ecolor='black', capsize=5, capthick=1.5)
    # ax[3].legend()

    for a in ax:
        a.axvline(time_range[0], color='black', ls='--', alpha = 0.5)
        a.axvline(time_range[1], color='black', ls='--', alpha = 0.5)

        
    fig.suptitle(f'{key.lower()}_correlation_{alias}')
    fig.subplots_adjust(wspace=0.0, hspace=0.3)
    fig.tight_layout()

    if save_fig:
        fig_path = f'{load_folder}/figures/{key.lower()}_correlation_{alias}.png'
        fig.savefig(fig_path, dpi=600)
plt.show()

