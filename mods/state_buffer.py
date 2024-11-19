import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from mods.plant import Plant


class StateBuffer:
    def __init__(self, size=100, skip=10, sim=None, preset_times=None, data=None, plant_kwargs=None):
        self.sim = sim
        self.size = size
        self.skip = skip
        self.states = []
        self.times = []
        self.preset_times = preset_times

        if data is not None:
            self.import_data(
                data=data, plant_kwargs=plant_kwargs)

        if preset_times is not None:
            preset_times = [int(t) for t in preset_times]

    def add(self, state, t):
        if len(self.times) == 0 or t not in self.times:
            if len(self.times) >= self.size:
                for i, time in enumerate(self.times):
                    if time not in self.preset_times:
                        self.states.pop(i)
                        self.times.pop(i)
                        # print(
                        #     f'StateBuffer.add(): Removed state at time {time}.')
                        break

            self.states.append(state)
            self.times.append(t)
            # print(f'StateBuffer.add(): Added state at time {t}.')
            # print(f'StateBuffer.add(): {self.times=}')
            # print(f'StateBuffer.add(): {len(self.states)=}')
            # print()

    def get(self, times=None):
        if times is None:
            return copy.deepcopy(self.states)

        indices = []
        for t in times:
            if t < self.times[0] or t > self.times[-1]:
                warnings.warn(
                    f'Time {t} is out of bounds. Start time: {self.times[0]}, End time: {self.times[-1]}', UserWarning)
                indices.append(np.nan)
            else:
                indices.append(np.where(np.array(self.times) == t)[0][0])
        return copy.deepcopy([self.states[i] for i in indices if not np.isnan(i)])

    def get_states(self):
        return copy.deepcopy(self.states)

    def get_times(self):
        return copy.deepcopy(self.times)

    def make_array(self):
        columns_per_plant = 5  # x, y, r, t, id

        L = 0

        states = self.get_states()
        times = self.get_times()

        for state in states:
            L += len(state)

        shape = (L, columns_per_plant)
        state_buffer_array = np.full(shape, np.nan)

        i = 0
        while i < L:
            for t, state in zip(times, states):
                for plant in state:
                    state_buffer_array[i, 0] = plant.pos[0]
                    state_buffer_array[i, 1] = plant.pos[1]
                    state_buffer_array[i, 2] = plant.r
                    state_buffer_array[i, 3] = t
                    state_buffer_array[i, 4] = plant.id

                    i += 1
        return state_buffer_array

    def save(self, path):
        path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_buffer_array = self.make_array()

        # Save the StateBuffer array object to the specified path as csv
        np.savetxt(path, state_buffer_array, delimiter=',')

    def import_data(self, data, plant_kwargs):
        states = []
        times = []
        for i in range(len(data)):
            x, y, r, t, id = data.loc[i]
            if np.isnan(x) or np.isnan(y) or np.isnan(r) or np.isnan(t) or np.isnan(id):
                continue
            else:
                if t not in times:
                    states.append([])
                    times.append(int(t))
                states[-1].append(
                    Plant(pos=np.array([x, y]), r=r, id=id, **plant_kwargs))

        self.states = states
        self.times = times

    def plot_state(self, state, t=None, size=2, fig=None, ax=None, half_width=0.5, half_height=0.5):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # ax.set_title('State')
        # ax.set_xlabel('Width (u)')
        # ax.set_ylabel('Height (u)')
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)
        ax.set_aspect('equal', 'box')
        for plant in state:
            if t is not None:
                ax.set_title(f't = {t}', fontsize=7)
            ax.add_artist(plt.Circle(plant.pos, plant.r,
                                     color='green', fill=True, transform=ax.transData))

        # if self.sim is not None:
        #     _m = self.sim.kwargs.get('_m')
        #     if _m is not None:
        #         x_ticks = ax.get_xticks() * _m
        #         y_ticks = ax.get_yticks() * _m
        #         ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
        #         ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot(self, size=2, title='StateBuffer'):
        states = self.get_states()
        times = self.get_times()
        T = len(times)
        if T == 0:
            print('StateBuffer.plot(): No states to plot.')
            return None, None
        if T == 1:
            n_rows = 1
            n_cols = 1
        else:
            n_rows = int(np.floor(T / np.sqrt(T)))
            n_cols = (T + 1) // n_rows + (T % n_rows > 0)
        print(f'StateBuffer.plot(): {n_rows=}, {n_cols=}')

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.05, hspace=0.05)
        fig.tight_layout()
        fig.suptitle(title, fontsize=10)
        if T == 1:
            self.plot_state(state=states[0], t=times[0], size=size, ax=ax)
        else:
            for i in range(T):
                state = states[i]
                if n_rows == 1:
                    k = i
                    self.plot_state(
                        state=state, t=times[i], size=size, fig=fig, ax=ax[k])
                else:
                    l = i//n_cols
                    k = i % n_cols
                    self.plot_state(
                        state=state, t=times[i], size=size, fig=fig, ax=ax[l, k])

        if T < n_rows*n_cols:
            for j in range(T, n_rows*n_cols):
                if n_rows == 1:
                    ax[j].axis('off')
                else:
                    l = j//n_cols
                    k = j % n_cols
                    ax[l, k].axis('off')

        return fig, ax
