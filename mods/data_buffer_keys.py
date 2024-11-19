import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy


class DataBuffer:
    def __init__(self, size=None, data=None, keys=None):
        if data is not None:
            self.import_data(data)
            return

        if keys is not None:
            self.keys = list(str(key) for key in keys)
        else:
            self.keys = ['Time', 'Biomass', 'Population']

        self.size = size
        self.values = np.full((size, len(self.keys)), np.nan)
        self.length = 0

    def add(self, data, t):
        self.values[t] = np.array([t, *data])

        if len(self.values) > self.size:
            self.values.pop(0)
        self.length = self.length + 1

    def analyze_state(self, state, t):
        biomass = sum([plant.area for plant in state])
        population = len(state)
        data = np.array([biomass, population])
        print(' '*30 + f'|\tt = {t:^5}\t|\tP = {population:^6}\t|\tB = {
              np.round(biomass, 5):^5}\t|', end='\r')
        if t % 100 == 0:
            print()
        self.add(data, t)
        return data

    def analyze_and_add(self, state, t):
        data = self.analyze_state(state, t)
        self.add(data, t)
        return data

    def finalize(self):
        self.values = self.values[:self.length-1]

    def plot(self, size=6, title='DataBuffer', keys=None):
        fig, ax = plt.subplots(len(self.keys) - 1, 1, figsize=(
            size/2 * len(self.keys) - 1, size))

        if title is not None:
            fig.suptitle(title, fontsize=10)

        fig.tight_layout(pad=3.0)
        cmap = plt.get_cmap('winter')

        for i, key in enumerate(self.keys[1:]):
            x_data = self.values[:, 0]
            y_data = self.values[:, i+1]

            ax[i].plot(self.values[:, 0], self.values[:, i+1],
                       label=key, color=cmap((i + 1)/len(self.keys)))
            ax[i].grid()
            ax[i].legend()
        ax[1].set_xlabel('Time')

        for ax_i in ax:
            ax_i.grid()
            ax_i.legend()
        return fig, ax

    def get_data(self, data_idx=None, keys=None):
        if keys is not None:
            keys_not_found = [key for key in keys if key not in self.keys]
            if len(keys_not_found) > 0:
                raise ValueError(f'Keys {keys_not_found} not in {self.keys:}')
            if isinstance(keys, str):
                keys = [keys]
            if isinstance(keys, list):
                keys_idx = [self.keys.index(key) for key in keys]
        elif keys is None:
            keys_idx = list(range(0, len(self.keys)))

        if data_idx is None:
            data_idx = list(range(0, self.length))
        elif isinstance(data_idx, int):
            data_idx = [data_idx]

        data = copy.deepcopy(self.values[np.ix_(data_idx, keys_idx)])

        return data.T if len(data_idx) > 1 else data.T

    def save(self, path):
        if not path.endswith('.csv'):
            path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the DataBuffer object to the specified path as csv with headers
        header = ','.join(self.keys)
        np.savetxt(path, self.values, delimiter=',',
                   header=header, comments='')

    def import_data(self, data, keys=None):
        if isinstance(data, np.ndarray):
            if keys is None and isinstance(data[0, 0], str):
                self.keys = list(data[0])
                data = data[1:].astype(float)
            else:
                self.keys = list(str(i) for i in range(data.shape[1]))
            self.size = data.shape[0]
            self.values = data
            self.length = data.shape[0]
        elif isinstance(data, pd.DataFrame):
            self.size = data.shape[0]
            self.values = data.to_numpy()
            self.length = data.shape[0]
            self.keys = list(str(col) for col in data.columns)

        # if isinstance(data, np.ndarray):
        #     self.size = data.shape[0]
        #     self.values = data
        #     self.length = data.shape[0]
        #     self.keys = list(str(i) for i in range(data.shape[1]))
        # elif isinstance(data, pd.DataFrame):
        #     self.size = data.shape[0]
        #     self.values = data.to_numpy()
        #     self.length = data.shape[0]
        #     self.keys = list(str(col) for col in data.columns)
