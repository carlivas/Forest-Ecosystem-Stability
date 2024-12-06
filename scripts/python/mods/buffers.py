import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import warnings

from mods.plant import Plant
from matplotlib.colors import ListedColormap


class DataBuffer:
    def __init__(self, sim=None, size=None, data=None, keys=None):
        if data is not None:
            self.import_data(data)
            return

        if keys is not None:
            self.keys = list(str(key) for key in keys)
        else:
            self.keys = ['Time', 'Biomass', 'Population', 'Precipitation']

        self.sim = sim
        self.size = size+1
        self.values = np.full((size+1, len(self.keys)), np.nan)
        self.length = 0

    def add(self, t, data):
        self.values[t] = np.array([t, *data])
        self.length = self.length + 1

        if len(self.values) > self.size:
            self.values.pop(0)
            self.length = self.size

    def analyze_state(self, state, t):
        biomass = sum([plant.area for plant in state])
        population = len(state)
        precipitation = self.sim.precipitation_func(t)
        data = np.array([biomass, population, precipitation])
        print(' '*30 + f'|  t = {t:<6}  |  N = {population:<6}  |  B = {
              np.round(biomass, 4):<6}  |  P = {np.round(precipitation, 4):<6}', end='\r')
        if t % 100 == 0:
            print()
        self.add(t, data)
        return data

    def analyze_and_add(self, state, t):
        data = self.analyze_state(state, t)
        self.add(t, data)
        return data

    def finalize(self):
        self.values = self.values[:np.where(
            ~np.isnan(self.values).all(axis=1))[0][-1] + 1]
        self.length = self.values.shape[0]

    def plot(self, size=6, title='DataBuffer', keys=None):
        if keys is not None:
            self.keys = keys
        fig, ax = plt.subplots(
            len(self.keys) - 1, 1,
            figsize=(size,
                     size * (len(self.keys) - 1) / 3),
            sharex=True)

        if title is not None:
            fig.suptitle(title, fontsize=10)

        fig.tight_layout(pad=3.0, h_pad=0.0)

        cmap = ListedColormap(
            ['#012626', '#1A402A', '#4B7340', '#7CA653', '#A9D962'])

        if keys is None:
            keys = self.keys[1:]

        for i, key in enumerate(keys):
            x_data = self.values[:, 0]
            y_data = self.values[:, i+1]

            ax[i].plot(x_data, y_data,
                       label=key, color=cmap((i + 1)/len(self.keys)))
            ax[i].set_ylim(0, 1.1*np.nanmax(y_data))
            ax[i].grid()
            ax[i].legend()
        ax[-1].set_xlabel('Time')

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

        data = self.values[np.ix_(data_idx, keys_idx)]

        return data if len(data_idx) > 1 else data[0]

    def save(self, path):
        self.finalize()

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

        self.finalize()


class FieldBuffer:
    def __init__(self, sim=None, resolution=2, size=10, skip=1, preset_times=None, data=None, **sim_kwargs):
        self.sim = sim
        self.size = size    # estimated max number of fields to store
        self.resolution = resolution
        self.skip = skip
        self.times = []
        self.preset_times = preset_times
        self.sim_kwargs = sim_kwargs

        self.fields = np.full(
            (self.size, self.resolution, self.resolution), np.nan)

        if data is not None:
            fields, times = self.import_data(
                data=data)
            self.fields = fields
            self.times = times

        if preset_times is not None:
            integer_types = (np.int8, np.uint8, np.int16, np.uint16,
                             np.int32, np.uint32, np.int64, np.uint64)
            if not all(isinstance(t, integer_types) for t in preset_times):
                raise ValueError(
                    "All elements in preset_times must be integers")

    def add(self, field, t):
        if len(self.times) == 0 or t not in self.times:
            if len(self.times) >= self.size:
                for i, time in enumerate(self.times):
                    if time not in self.preset_times:
                        fields = self.get_fields()
                        B = np.roll(fields[i:], -1, axis=0)
                        fields = np.concatenate(
                            (fields[:i], B), axis=0)
                        self.fields = fields
                        self.times.pop(i)
                        # print(
                        # f'FieldBuffer.add(): Removed field at time {time}.')
                        break

            self.fields[len(self.times)] = field
            self.times.append(t)
            # print(f'FieldBuffer.add(): Added field at time {t}.')
            # print(f'FieldBuffer.add(): {self.times=}')
            # print(f'FieldBuffer.add(): {len(self.fields)=}')
            # print()

    def get_fields(self, times=None):
        if times is None:
            fields = self.fields
        else:
            # Find the the indices of FieldBuffer.times that match the input times
            indices = [np.where(np.array(self.times) == t)[
                0][0] if t in self.times else np.nan for t in times]

            fields = [self.fields[i] for i in indices if not np.isnan(i)]
        return copy.deepcopy(fields)

    def get_times(self):
        return self.times.copy()

    def import_data(self, data):
        times = data.loc[:, 0].astype(int)
        fields_arr = data.loc[:, 1:].values

        self.resolution = int(np.sqrt(data.values.shape[-1]))
        fields = fields_arr.reshape(-1, self.resolution, self.resolution)

        return fields, times

    def make_array(self):
        shape = self.fields.shape
        arr = self.get_fields().reshape(-1, shape[1]*shape[2])

        # Find the first row with NaN values
        nan_index = np.where(np.isnan(arr).any(axis=1))[0]
        if nan_index.size > 0:
            arr = arr[:nan_index[0]]

        times = np.array(self.get_times())
        return np.concatenate((times.reshape(-1, 1), arr), axis=1)

    def save(self, path):
        path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the FieldBuffer array object to the specified path as csv
        np.savetxt(path, self.make_array(), delimiter=',')

    def plot_field(self, field, t=None, size=2, fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        ax.contour(field, levels=[1.0], colors=[
                   'r'], linewidths=[1], alpha=0.5)
        ax.imshow(field, origin='lower', cmap='Greys',
                  vmin=vmin, vmax=vmax, extent=extent)

        if t is not None:
            ax.text(0.0, -0.6, f't = {t}', ha='center', fontsize=7)

        # if self.sim is not None:
        #     _m = self.sim.kwargs['_m']
        #     if _m is not None:
        #         x_ticks = ax.get_xticks() * _m
        #         y_ticks = ax.get_yticks() * _m
        #         ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
        #         ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot(self, size=2, vmin=0, vmax=None, title='FieldBuffer', extent=[-0.5, 0.5, -0.5, 0.5]):
        fields = self.get_fields()
        times = self.get_times()
        if vmax is None:
            vmax = np.nanmax(fields)
        T = len(times)

        if T == 0:
            print('FieldBuffer.plot(): No states to plot.')
            return None, None
        if T == 1:
            n_rows = 1
            n_cols = 1
        else:
            n_rows = int(np.floor(np.sqrt(T)))
            n_cols = (T + 1) // n_rows + (T % n_rows > 0)

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.05, hspace=0.05)
        fig.tight_layout()

        fig.suptitle(title, fontsize=10)

        cax = fig.add_axes([0.05, 0.05, 0.9, 0.02])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=7)

        if T == 1:
            self.plot_field(fields[0], t=times[0],
                            size=size, fig=fig, ax=ax, vmin=vmin, vmax=vmax, extent=extent)
        else:
            for i in range(T):
                field = fields[i]
                if n_rows == 1:
                    k = i
                    self.plot_field(
                        field=field, t=times[i], size=size, fig=fig, ax=ax[k], vmin=vmin, vmax=vmax, extent=extent)
                else:
                    l = i // n_cols
                    k = i % n_cols
                    self.plot_field(
                        field=field, t=times[i], size=size, fig=fig, ax=ax[l, k], vmin=vmin, vmax=vmax, extent=extent)

        if T < n_rows*n_cols:
            for j in range(T, n_rows*n_cols):
                if n_rows == 1:
                    ax[j].axis('off')
                else:
                    l = j//n_cols
                    k = j % n_cols
                    ax[l, k].axis('off')

        return fig, ax


class StateBuffer:
    def __init__(self, size=100, skip=10, sim=None, preset_times=None, data=None, plant_kwargs=None):
        self.sim = sim
        self.size = size
        self.skip = skip
        self.states = []
        self.times = []
        self.preset_times = preset_times
        self.plant_kwargs = plant_kwargs

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

    def get_states(self, times=None):
        if times is None:
            states = self.states
        else:
            # Find the the indices of StateBuffer.times that match the input times
            indices = [np.where(np.array(self.times) == t)[
                0][0] if t in self.times else np.nan for t in times]
            states = [self.states[i] for i in indices if not np.isnan(i)]
        return copy.deepcopy(states)

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
        for i in range(data.shape[0]):
            x, y, r, t, id = data.loc[i]
            if np.isnan(x) or np.isnan(y) or np.isnan(r) or np.isnan(t):
                print(
                    f'StateBuffer.import_data(): !Warning! Skipping NaN values at row {i}.')
                continue
            else:
                if t not in times:
                    states.append([])
                    times.append(int(t))
                states[-1].append(
                    Plant(pos=np.array([x, y]), r=r, id=id, **plant_kwargs))

        self.states = states
        self.times = times

    def plot_state(self, state, t=None, size=2, fig=None, ax=None, half_width=0.5, half_height=0.5, fast=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # ax.set_title('State')
        # ax.set_xlabel('Width (u)')
        # ax.set_ylabel('Height (u)')
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)
        ax.set_aspect('equal', 'box')

        for plant in state:
            if fast and plant.r > self.plant_kwargs['r_min']*100:
                ax.add_artist(plt.Circle(plant.pos, plant.r,
                                         color='green', fill=False, transform=ax.transData))
            elif not fast:
                ax.add_artist(plt.Circle(plant.pos, plant.r,
                                         color='green', fill=True, transform=ax.transData))

        if t is not None:
            ax.text(0.0, -0.6, f't = {t}', ha='center', fontsize=7)

        # if self.sim is not None:
        #     _m = self.sim.kwargs['_m']
        #     if _m is not None:
        #         x_ticks = ax.get_xticks() * _m
        #         y_ticks = ax.get_yticks() * _m
        #         ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
        #         ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot(self, size=2, title='StateBuffer', fast=False):
        if fast:
            print(
                'StateBuffer.plot(): Faster plotting is enabled, some elements might be missing in the plots.')
            title += ' (Fast)'
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
            self.plot_state(state=states[0], t=times[0],
                            size=size, ax=ax, fast=fast)
        else:
            for i in range(T):
                state = states[i]
                if n_rows == 1:
                    k = i
                    self.plot_state(
                        state=state, t=times[i], size=size, fig=fig, ax=ax[k], fast=fast)
                else:
                    l = i//n_cols
                    k = i % n_cols
                    self.plot_state(
                        state=state, t=times[i], size=size, fig=fig, ax=ax[l, k], fast=fast)

        if T < n_rows*n_cols:
            for j in range(T, n_rows*n_cols):
                if n_rows == 1:
                    ax[j].axis('off')
                else:
                    l = j//n_cols
                    k = j % n_cols
                    ax[l, k].axis('off')

        return fig, ax
