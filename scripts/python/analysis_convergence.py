'''
This document makes a simple linear regression of the last part of a imulation run to evaluate whether the run has converged, and is only used as a test for the method.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import json
from scipy.signal import welch

from mods.plant import Plant
from mods.simulation import Simulation
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer


def linear_regression(x, y, advanced=False):
    """Calculates the linear regression of a dataset.

    Args:
        x (ndarray): (N, ) array of the independent variable.
        y (ndarray): (N, ) array of the dependent variable.

    Returns:
        tuple: A tuple containing the slope, intercept, regression line, residuals and R2 of the linear regression.
    """
    X = np.vstack([np.ones_like(x), x]).T

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


load_folder = r'Data\temp\dfres_2_20'

trend_window = 5000
trend_threshold = 1e-3

sim_nums = [f.split('_')[-1].split('.')[0]
            for f in os.listdir(load_folder) if 'data_buffer' in f]

for idx, i in enumerate(sim_nums[:]):
    kwargs = json.load(open(f'{load_folder}/kwargs_{i}.json'))

    data_buffer_arr = pd.read_csv(
        f'{load_folder}/data_buffer_{i}.csv')
    data_buffer = DataBuffer(data=data_buffer_arr)

    # time, population, biomass = data_buffer.get_data(
    #     keys=['time', 'population', 'biomass'])
    data_buffer.finalize()
    data = data_buffer.get_data()
    time = data[:, 0]
    biomass = data[:, 1]
    population = data[:, 2]
    print(f'{time=}')
    print(f'{population=}')
    print(f'{biomass=}')

    window = np.min([trend_window, len(time)])
    x = time[-window:]
    y_B = biomass[-window:]
    y_P = population[-window:]

    _, slope_B, regression_line_B, _, _ = linear_regression(
        x, y_B, advanced=True)
    _, slope_P, regression_line_P, _, _ = linear_regression(
        x, y_P, advanced=True)

    fig, ax = data_buffer.plot()

    rel_slope_B = slope_B/y_B.max()
    rel_slope_P = slope_P/y_P.max()

    did_converge_B = np.abs(rel_slope_B) < trend_threshold
    c_B = 'g' if did_converge_B else 'r'
    did_converge_P = np.abs(rel_slope_P) < trend_threshold
    c_P = 'g' if did_converge_P else 'r'
    did_converge = did_converge_B and did_converge_P
    c = 'g' if did_converge else 'r'

    ax[0].plot(x, regression_line_B, label=f'relative slope: {
               rel_slope_B:.5f}', linestyle='--', color=c_B)
    ax[0].set_ylabel('Biomass', fontsize=8)
    ax[0].legend(fontsize=8)

    ax[1].plot(x, regression_line_P, label=f'relative slope: {
               rel_slope_P:.5f}', linestyle='--', color=c_P)
    ax[1].set_ylabel('Population', fontsize=8)
    ax[1].set_xlabel('Time', fontsize=8)
    ax[1].legend(fontsize=8)
    fig.suptitle(f'Trend analysis for sim {
                 i}\nConverged: {did_converge}', color=c)

    plt.show()
