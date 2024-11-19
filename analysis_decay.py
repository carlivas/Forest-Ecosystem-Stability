import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json
import scipy

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer


def linear_regression(X, y, advanced=False):
    """Calculates the linear regression of a dataset.

    Args:
        X (ndarray): (N, 2) array where the first row is a column of ones and the second row is the independent variable.
        y (ndarray): (N, ) array of the dependent variable.

    Returns:
        tuple: A tuple containing the slope, intercept, regression line, residuals and R2 of the linear regression.
    """
    # Calculate theta using the equation
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    intercept, slope = theta

    if advanced:
        # Calculate the regression line using the calculated theta
        regression_line = X @ theta

        residuals = y - regression_line
        # Calculate the sum of squared residuals
        sum_squared_residuals = np.sum(residuals**2)
        # regression_line = slope * X[:, 1] + intercept
        # residuals = y - regression_line
        return intercept, slope, regression_line, residuals, sum_squared_residuals
    else:
        return intercept, slope


load_folder = r'Data\lq_rc_ensemble_n100'
sim_nums = [int(f.split('.')[0].replace('data_buffer_', ' '))
            for f in os.listdir(load_folder) if 'data_buffer_' in f]

new_sim_nums = []
for i, n in enumerate(sim_nums):
    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
    data_buffer_arr = data_buffer_arr[~np.isnan(data_buffer_arr).any(axis=1)]

    # if data_buffer_arr.shape[0] < 10000 and data_buffer_arr[-1][2] == 0:
    if data_buffer_arr.shape[0] > 1 and data_buffer_arr[-1][2] == 0:
        new_sim_nums.append(n)

sim_nums = new_sim_nums

errs = []
for i, n in enumerate(sim_nums[::-1]):
    with open(os.path.join(load_folder, f'kwargs_{n}.json'), 'r') as file:
        kwargs = json.load(file)

    lq = kwargs['sim_kwargs']['land_quality']
    sg = kwargs['plant_kwargs']['species_germination_chance']

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{n}.csv', header=None).to_numpy()
    data_buffer_arr = data_buffer_arr[~np.isnan(data_buffer_arr).any(axis=1)]
    data_buffer = DataBuffer(data=data_buffer_arr)

    times = data_buffer.values[:, 0]
    biomass = data_buffer.values[:, 1]
    populations = data_buffer.values[:, 2]

    X = np.vstack([np.ones_like(times), times]).T
    y = populations

    intercept, slope, regression_line, residuals, sum_squared_residuals = linear_regression(
        X, y, advanced=True)

    if sum_squared_residuals < 10000:
        errs.append([n, lq, sg, sum_squared_residuals])

errs = np.array(errs)
errs = errs[errs[:, 3].argsort()]
lin_nums = errs[:, 0].astype(int)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
for a in ax.flatten():
    a.set_facecolor('#f0f0e0')
# norm = plt.Normalize(vmin=0, vmax=max(errs[:, 3]))
norm = plt.Normalize(vmin=0, vmax=10000)
sc = ax[0, 0].scatter(errs[:, 1], errs[:, 2],
                      c=errs[:, 3], cmap='Greys_r', norm=norm)
ax[0, 0].set_xlabel('land quality')
ax[0, 0].set_ylabel('species germination chance')
plt.colorbar(sc, ax=ax[0, 0], label='Sum of Squared Residuals', norm=norm,
             orientation='horizontal', pad=0.1, aspect=30, location='top').ax.xaxis.label.set_fontsize(7)

DY = (norm.vmax - norm.vmin)*0.05
ylim = (norm.vmin - DY, norm.vmax + DY)
ax[0, 1].scatter(errs[:, 1], errs[:, 3], c=errs[:, 3],
                 cmap='Greys_r', norm=norm)
ax[0, 1].set_ylim(ylim)
ax[0, 1].set_xlabel('land quality', fontsize=7)
ax[0, 1].set_ylabel('sum of squared residuals', fontsize=7)
ax[0, 1].set_title('Error vs land quality', fontsize=8)

ax[1, 0].scatter(errs[:, 2], errs[:, 3], c=errs[:, 3],
                 cmap='Greys_r', norm=norm)
ax[1, 0].set_ylim(ylim)
ax[1, 0].set_xlabel('species germination chance', fontsize=7)
ax[1, 0].set_ylabel('sum of squared residuals', fontsize=7)
ax[1, 0].set_title('Error vs germination chance', fontsize=8)

bins = np.linspace(0, norm.vmax, 20)
ax[1, 1].hist(errs[:, 3], bins=bins, color='k')
ax[1, 1].set_xlabel('sum of squared residuals', fontsize=7)
ax[1, 1].set_ylabel('frequency', fontsize=7)
ax[1, 1].set_title('Error histogram', fontsize=8)


fig.tight_layout()
plt.show()
