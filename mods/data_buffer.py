import numpy as np
import matplotlib.pyplot as plt
import os
import copy


class DataBuffer:
    def __init__(self, size=None, data=None):
        if data is not None:
            self.size = len(data)
            self.values = np.array(data)
            self.length = len(data)
        else:
            self.size = size
            self.values = np.full((size, 3), np.nan)
            self.length = 0

    def add(self, data, t):
        self.values[t] = np.array([t, *data])

        if len(self.values) > self.size:
            self.values.pop(0)
        self.length = self.length + 1

    def analyze_state(self, state, t):
        biomass = sum([plant.area for plant in state])
        population_size = len(state)
        data = np.array([biomass, population_size])
        print(' '*25 + f'\t|\tt = {t:^5}    |    P = {population_size:^6}    |    B = {
              np.round(biomass, 5):^5}', end='\r')
        self.add(data, t)
        return data

    def analyze_and_add(self, state, t):
        data = self.analyze_state(state, t)
        self.add(data, t)
        return data

    def finalize(self):
        self.values = self.values[:self.length-1]

    def plot(self, size=6, title='DataBuffer'):
        fig, ax = plt.subplots(2, 1, figsize=(
            size, size))

        if title is not None:
            fig.suptitle(title, fontsize=10)

        fig.tight_layout(pad=3.0)
        ax[0].plot(self.values[:, 0], self.values[:, 1],
                   label='Biomass', color='green')
        # ax[0].set_xticks([])
        ax[1].plot(self.values[:, 0], self.values[:, 2],
                   label='Population Size', color='teal')
        ax[1].set_xlabel('Time')

        for ax_i in ax:
            ax_i.grid()
            ax_i.legend()
        return fig, ax

    def get_data(self, indices=None):

        if indices is None:
            return copy.deepcopy(self.values)
        if isinstance(indices, int):
            return copy.deepcopy(self.values[indices])

        return copy.deepcopy([self.values[i] for i in indices])

    def save(self, path):

        path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the DataBuffer object to the specified path as csv
        np.savetxt(path, self.values, delimiter=',')
        # with open(path, 'wb') as f:
        #     pickle.dump(self, f)

    # def load(self, path):
    #     # Create the directory if it doesn't exist
    #     os.makedirs(os.path.dirname(path), exist_ok=True)

    #     # Load the DataBuffer object from the specified path
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)
    #     return data
