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
from mods.utilities import linear_regression


load_folder = 'Data\MODI\modi_ensemble_L4500\precipitation_6300e-5'
trend_windows = [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

for trend_window in trend_windows:
    print(f'Analysing trend for window: {trend_window}')
    trend_threshold = 3

    sim_nums = [f.split('_')[-1].split('.')[0]
                for f in os.listdir(load_folder) if 'data_buffer' in f]
    convergence_list = pd.DataFrame(
        columns=['sim_num', 'guess was right', 'truth', 'convergence_factor'])
    for idx, i in enumerate(sim_nums[:]):
        kwargs = json.load(open(f'{load_folder}/kwargs_{i}.json'))

        data_buffer_arr = pd.read_csv(
            f'{load_folder}/data_buffer_{i}.csv')
        data_buffer = DataBuffer(data=data_buffer_arr, keys=[
                                 'Time', 'Biomass', 'Population'])

        data_buffer.finalize()
        data = data_buffer.get_data()
        time = data[:, 0]
        biomass = data[:, 1]
        population = data[:, 2]

        window = np.min([trend_window, len(time)])
        x = time[-window:]
        y_B = biomass[-window:]

        std_B = np.std(y_B)
        if std_B == 0:
            _, slope_norm_B, regression_line_norm_B, _, _ = linear_regression(
                x, y_B, advanced=True)
        else:
            y_B_norm = (y_B - np.mean(y_B)) / np.std(y_B)
            _, slope_norm_B, regression_line_norm_B, _, _ = linear_regression(
                x, y_B_norm, advanced=True)

        regression_line_B = regression_line_norm_B * np.std(y_B) + np.mean(y_B)

        should_converge = True
        if i in ['20241226-114456',
                 '20241226-165931',
                 '20241226-174701',
                 '20241226-210538',
                 '20241227-043326',
                 '20241227-045519',
                 '20241227-123221',
                 '20241227-155101',
                 '20241227-165826',
                 '20241227-210928',
                 '20241229-143128',
                 '20250101-235727']:
            should_converge = False
        convergence_factor = np.abs(slope_norm_B) * trend_window - trend_threshold
        did_converge_B = convergence_factor < 0
        did_converge = did_converge_B
        guess_was_right = did_converge == should_converge
        convergence_list.loc[idx] = [i, guess_was_right,
                                     should_converge, convergence_factor]

        # fig, ax = data_buffer.plot(keys=['biomass', 'population'])

        # c_B = 'k' if did_converge_B else 'r'
        # c = 'k' if did_converge else 'r'

        # ax[0].plot(x, regression_line_B, label=f'Convergence factor: {
        #     convergence_factor:.2e}', linestyle='--', color=c_B)
        # ax[0].set_ylabel('Biomass', fontsize=8)
        # ax[0].legend(fontsize=8)

        # ax[1].set_ylabel('Population', fontsize=8)
        # ax[1].set_xlabel('Time', fontsize=8)
        # ax[1].legend(fontsize=8)
        # fig.suptitle(f'Trend analysis for sim {
        #     i}\nConverged: {did_converge}, threshold: {trend_threshold:.2e}', color=c)
        # plt.show()

    n_wrong_guesses = len(
        convergence_list[convergence_list['guess was right'] == False])
    n_right_guesses = len(
        convergence_list[convergence_list['guess was right'] == True])
    print(f'Number of wrong guesses: {n_wrong_guesses}')
    print(f'Number of right guesses: {n_right_guesses}')
    convergence_list.to_csv(
        f'{load_folder}/_convergence_list_{trend_window}.csv', index=False)
