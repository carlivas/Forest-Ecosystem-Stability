import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from scipy.optimize import curve_fit
from scipy.stats import f


folder = 'D:/recovery_rates'
alias = 'rec_rate_250310_171212'

# Specify keys to analyze
keys = ['Biomass', 'Population']
perturbation_times = np.array([1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000])
save_fig = False
save_kwargs = False


# Get data
kwargs = pd.read_json(
    f'{folder}/kwargs-{alias}.json', typ='series').to_dict()
perturbation_times = np.array(kwargs.get('perturbation_times', perturbation_times))

if save_kwargs:
        kwargs['perturbation_times'] = perturbation_times
        with open(f'{folder}/kwargs-{alias}.json', 'w') as f:
            json.dump(kwargs, f, indent=4)
    
db = pd.read_csv(f'{folder}/data_buffer-{alias}.csv')
time = db['Time'].values
time_step = time[1] - time[0]



# Calculate running means
window=200
running_means = db.rolling(window=window).mean()

# Setup dataframes
vals_before_perturbation = db[db['Time'].isin(perturbation_times)][keys]
vals_after_perturbation = db[db['Time'].isin(perturbation_times+time_step)][keys]
perturbation_sizes = pd.DataFrame(vals_after_perturbation.values - vals_before_perturbation.values, columns=keys, index=perturbation_times)

means_before_perturbation = running_means[running_means.index.isin(perturbation_times)][keys]
recovery_times = pd.DataFrame(columns=keys, index=perturbation_times)
vals_after_recovery = pd.DataFrame(columns=keys, index=perturbation_times)

# Calculate recovery times and values
for key in keys:
    for t in perturbation_times:
        val = db.loc[db['Time'] == t, key].values
        condition = db.loc[db['Time'] > t, key] >= means_before_perturbation.loc[t, key]
        dt = condition.idxmax() - t if condition.any() else np.nan

        recovery_times.loc[t, key] = dt
        vals_after_recovery.loc[t, key] = db.loc[db['Time'] == t + dt, key].values
        
        
print(f'recovery_times = \n{recovery_times}\n')
print(f'perturbation_sizes = \n{perturbation_sizes}\n')
print(f'perturbation_percentages = \n{perturbation_sizes / means_before_perturbation * 100}\n')
fig, axs = plt.subplots(len(keys), 2, figsize=(16, 8), sharex='col', gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle(f'{folder}/{alias}')
axs = np.array(axs).reshape(len(keys), 2)
axs[-1, 0].set_xlabel('Time')
axs[-1, 1].set_xlabel('Precipitation before perturbation')
axs[-1, 1].invert_xaxis()
# Add a secondary y-axis for Precipitation
ax_precipitation = axs[0, 0].twinx()
ax_precipitation.plot(time, db['Precipitation'], 'b', label='Precipitation', alpha=0.5)
ax_precipitation.tick_params(axis='y', labelcolor='b')

for k, key in enumerate(keys):
    # Left column: Time vs Recovery
    axs[k, 0].plot(time, db[key], 'k', label=key, alpha=0.5)
    axs[k, 0].plot(time, running_means[key], 'k--', label=f'Mean (window={window})')
    for i, t in enumerate(perturbation_times):
        axs[k, 0].fill_betweenx([0, db[key].max()], t, t + recovery_times.loc[t, key], color='r', alpha=0.3, label='recovery interval' if i == 0 else '', edgecolor='none')
    
    # Right column: Precipitation vs Recovery time
    precipitations_before_perturbation = db[db['Time'].isin(perturbation_times)]['Precipitation']
    axs[k, 1].plot(precipitations_before_perturbation, recovery_times[key], 'k--o', label=key)
    axs[k, 1].set_ylabel('Recovery time')
    axs[k, 1].legend()

axs[-1, 0].legend(loc = 'lower left')

# Fuse legends of the twinx axes
lines, labels = axs[0, 0].get_legend_handles_labels()
lines_precip, labels_precip = ax_precipitation.get_legend_handles_labels()
axs[0, 0].legend(lines + lines_precip, labels + labels_precip, loc='lower left')

fig.tight_layout()
if save_fig:
    fig.savefig(f'{folder}/figures/recovery_rates-{alias}.png', dpi=300)

# plt.show()
