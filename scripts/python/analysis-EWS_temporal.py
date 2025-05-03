import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mods.utilities import get_unique_aliases

folder = 'Data/lin_prec_test'
alias = 'linprec_global_L2000_250430_114412'
save_fig = False

kwargs = pd.read_json(
    f'{load_folder}/kwargs-{alias}.json', typ='series').to_dict()
data_buffer = pd.read_csv(f'{load_folder}/data_buffer-{alias}.csv')


window_size = 1500  # Example window size, adjust as needed
time_range = slice(300, 15000, 1)

fig, axs = plt.subplots(4, 1, figsize=(6, 9), sharex=True)
fig.suptitle(f'{alias} - EWS_temporal')
# rolling_mean = data_buffer['Biomass'].iloc[time_range].rolling(window=window_size).mean()
rolling_vari = data_buffer['Biomass'].iloc[time_range].rolling(window=window_size, center=False).var()
rolling_skew = data_buffer['Biomass'].iloc[time_range].rolling(window=window_size, center=False).skew()
rolling_kurt = data_buffer['Biomass'].iloc[time_range].rolling(window=window_size, center=False).kurt()
# axs[0].plot(rolling_mean, label='Rolling Mean')
axs[0].plot(data_buffer['Biomass'].iloc[time_range], label='Biomass', color='k')
axs[1].plot(rolling_vari, label='Rolling Variance', color='r')
axs[2].plot(rolling_skew, label='Rolling Skewness', color='g')
axs[3].plot(rolling_kurt, label='Rolling Kurtosis', color='b')

# Add an indicator for the rolling window size
for ax in axs:
    ylim = ax.get_ylim()
    ax.fill_betweenx(
        (-10, 10),
        time_range.start,
        time_range.start + window_size,
        color='gray',
        alpha=0.3,
        label=f'Window Size ({window_size})'
    )
    ax.set_ylim(ylim)  # Reset the y-limits after filling

axs[-1].set_xlabel('Time')
for ax in axs:
    ax.legend(fontsize = 8)
    ax.grid()
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
        
fig.tight_layout()            
plt.show()