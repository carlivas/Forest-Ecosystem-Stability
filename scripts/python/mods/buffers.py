import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import warnings
import json

from mods.plant import Plant
from matplotlib.colors import ListedColormap
from matplotlib import animation

path_kwargs = 'default_kwargs.json'
default_kwargs = None
with open(path_kwargs, 'r') as file:
    default_kwargs = json.load(file)

def rewrite_state_buffer_data(state_buffer_df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in state_buffer_df.keys():
        print('State buffer already rewritten')
        return state_buffer_df
    
    state_buffer_df = pd.DataFrame(state_buffer_df.values, columns=['x', 'y', 'r', 't'])
    # generate id column and assign it to the dataframe
    id = np.arange(len(state_buffer_df))
    state_buffer_df = state_buffer_df.assign(id=id)
    state_buffer_df = state_buffer_df[['id', 'x', 'y', 'r', 't']]
    return state_buffer_df

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

    def add(self, data, t):
        self.values[self.length] = np.array([t, *data])
        self.length = self.length + 1

        if len(self.values) > self.size:
            print(f'\nDataBuffer.add(): !Warning! DataBuffer is full, previous data will be overwritten.')
            self.values.pop(0)
            self.length = self.size

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
            if data_idx < 0:
                data_idx = self.length + data_idx
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


class StateBuffer:
    def __init__(self, size=100, skip=10, sim=None, preset_times=None, data=None, **kwargs):
        self.sim = sim
        self.size = size
        self.skip = skip
        self.states = []
        self.times = []
        self.preset_times = preset_times

        if data is not None:
            self.import_data(
                data=data, kwargs=kwargs)

    def add(self, state, t):
        if len(self.times) < self.size:
            self.states.append(state)
            self.times.append(t)
        elif len(self.times) == self.size:
            removable_indices = [i for i, time in enumerate(
                self.times) if not any(np.isclose(time, preset_time) for preset_time in self.preset_times)]
            if removable_indices:
                self.states.pop(removable_indices[0])
                self.times.pop(removable_indices[0])

            self.states.append(state)
            self.times.append(t)
        else:
            print(
                f'\nStateBuffer.add(): !Warning! StateBuffer is full, previous states will be overwritten.')
            self.states.pop(0)
            self.times.pop(0)

            self.states.append(state)
            self.times.append(t)

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
        columns_per_plant = 4  # x, y, r, t

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

                    i += 1
        return state_buffer_array

    def save(self, path):
        path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_buffer_array = self.make_array()

        # Save the StateBuffer array object to the specified path as csv
        np.savetxt(path, state_buffer_array, delimiter=',')
        
    def import_data(self, data, kwargs):

        states = []
        times = []

        required_keys = ['L', 'time_step', 'r_min', 'r_max', 'growth_rate', 'dispersal_range']
        for key in required_keys:
            if key not in kwargs:
                kwargs[key] = default_kwargs[key]
                warnings.warn(f"StateBuffer.import_data(): Key '{key}' not found in kwargs. Using default value of {default_kwargs[key]} from default_kwargs.")

        _m = 1/kwargs['L']
        time_step = kwargs['time_step']
        r_min = kwargs['r_min'] * _m
        r_max = kwargs['r_max'] * _m
        growth_rate = kwargs['growth_rate'] * _m * time_step
        dispersal_range = kwargs['dispersal_range'] * _m

        # data = data.reset_index(drop=True)
        
        for i in range(data.shape[0]):
            id, x, y, r, t = data.loc[i]
            if any(np.isnan([id, x, y, r, t])):
                print(f'StateBuffer.import_data(): Skipping NaN(s) at row {i}.')
                continue
            if t not in times:
                states.append([])
                times.append(t)

            states[-1].append(
                Plant(
                    pos=np.array([x, y]),
                    r=r,
                    id=id,
                    r_min=r_min,
                    r_max=r_max,
                    growth_rate=growth_rate,
                    dispersal_range=dispersal_range
                )
            )

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
            if fast and plant.r > 0.005:
                ax.add_artist(plt.Circle(plant.pos, plant.r,
                                         color='green', fill=False, transform=ax.transData))
            elif not fast:
                ax.add_artist(plt.Circle(plant.pos, plant.r,
                                         color='green', fill=True, transform=ax.transData))

        if t is not None:
            t = float(round(t, 2))
            ax.text(0.0, -0.6, f'{t = }', ha='center', fontsize=7)

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
        elif T <= 40:
            n_rows = int(np.floor(T / np.sqrt(T)))
            n_cols = (T + 1) // n_rows + (T % n_rows > 0)
        else:
            self.animate(fast=fast)
            return None, None

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.05, hspace=0.05)
        fig.tight_layout()
        fig.suptitle(title, fontsize=8)
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

    def animate(self, size=6, title=None, fast=False):
        print('StateBuffer.animation(): Animating StateBuffer...')
        states = self.get_states()
        times = self.get_times()
        time_step = times[1] - times[0]
        T = len(times)

        if title is None:
            title = 'StateBuffer Animation'
        fig, ax = plt.subplots(1, 1, figsize=(size, size))
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal', 'box')
        ax.set_xticks([])
        ax.set_yticks([])

        circles = [plt.Circle(plant.pos, plant.r, color='green',
                              fill=not fast) for plant in states[0]]
        for circle in circles:
            ax.add_artist(circle)

        def animate(i):
            t = float(round(times[i], 2))
            ax.set_title(f'{t = }', fontsize = 8)
            for circle, plant in zip(circles, states[i]):
                circle.center = plant.pos
                circle.radius = plant.r
            return ax

        ani = animation.FuncAnimation(
            fig, animate, frames=T, interval= 10 * time_step, repeat=True)
        return ani


class FieldBuffer:
    def __init__(self, sim=None, resolution=10, size=20, skip=200, preset_times=None, data=None):
        self.sim = sim
        self.size = size    # estimated max number of fields to store
        self.resolution = resolution
        self.skip = skip
        self.times = []
        self.preset_times = preset_times

        self.fields = np.full(
            (self.size, self.resolution, self.resolution), np.nan)

        if data is not None:
            fields, times = self.import_data(
                data=data)
            self.fields = fields
            self.times = times.tolist()

    def add(self, field, t):
        if len(self.times) < self.size:
            self.fields[len(self.times)] = field
            self.times.append(t)
        elif len(self.times) == self.size:
            for i, time in enumerate(self.times):
                if not any(np.isclose(time, preset_time) for preset_time in self.preset_times):
                    fields = self.get_fields()
                    B = np.roll(fields[i:], -1, axis=0)
                    fields = np.concatenate(
                        (fields[:i], B), axis=0)
                    self.fields = fields
                    self.times.pop(i)
                    break
            self.fields[-1] = field
            self.times.append(t)
        else:
            print(
                f'\nFieldBuffer.add(): !Warning! FieldBuffer is full, previous fields will be overwritten.')
            self.fields = np.roll(self.fields, -1, axis=0)
            self.fields[-1] = field
            self.times.pop(0)
            self.times.append(t)

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
        times = data.loc[:, 0].values
        fields_arr = data.loc[:, 1:].values

        self.resolution = int(np.sqrt(data.values.shape[-1]))
        fields = fields_arr.reshape(-1, self.resolution, self.resolution)

        return fields, times

    def make_array(self):
        shape = self.fields.shape
        arr = self.get_fields().reshape(-1, shape[1]*shape[2])
        times = np.array(self.get_times())

        # Find the first row with NaN values
        nan_index = np.where(np.isnan(arr).any(axis=1))[0]
        if nan_index.size > 0:
            arr = arr[:nan_index[0]]

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
            t = float(round(t, 2))
            ax.text(0.0, -0.6, f'{t = }', ha='center', fontsize=7)

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
        elif T <= 40:
            n_rows = int(np.floor(T / np.sqrt(T)))
            n_cols = (T + 1) // n_rows + (T % n_rows > 0)
        else:
            self.animate(vmin=vmin, vmax=vmax, extent=extent)
            return None, None

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

    def animate(self, size=6, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        print('FieldBuffer.animation(): Animating FieldBuffer...')
        fields = self.get_fields()
        times = self.get_times()
        time_step = times[1] - times[0]
        T = len(times)

        fig, ax = plt.subplots(1, 1, figsize=(size, size))
        fig.suptitle('FieldBuffer Animation', fontsize=10)
        fig.tight_layout()
        cax = fig.add_axes([0.05, 0.05, 0.9, 0.02])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap='Greys', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize=7)

        def animate(i):
            ax.clear()
            t = float(round(times[i], 2))
            ax.set_title(f'{t = }')
            ax.imshow(fields[i], origin='lower', cmap='Greys',
                      vmin=vmin, vmax=vmax, extent=extent)
            ax.set_xticks([])
            ax.set_yticks([])
            return ax

        ani = animation.FuncAnimation(
            fig, animate, frames=T, interval= 10 * time_step, repeat=True)
        plt.show()
        return ani


class HistogramBuffer:
    def __init__(self, size=10, bins=25, start=0, end=1, title='HistogramBuffer', data=None):
        if data is not None:
            self.import_data(data)
            return
        self.bins = bins
        self.bin_vals = np.linspace(start, end, bins+1)
        self.values = np.full((size, bins), np.nan)
        self.times = np.full(size, np.nan)
        self.start = start
        self.end = end
        self.title = title
        self.length = 0

    def add(self, data, t):
        hist, _ = np.histogram(data, bins=self.bin_vals)
        self.values[self.length] = hist
        self.times[self.length] = t
        self.length = self.length + 1
        
        size = self.values.shape[0]
        if len(self.values) > size:
            print(f'\nHistogramBuffer.add(): !Warning! HistogramBuffer is full, previous data will be overwritten.')
            self.values.pop(0)
            self.length = size

    def get_values(self, t=None):
        values = self.values[~np.isnan(self.values).all(axis=1)].copy()
        if t is None:
            return values
        return values[t]
    
    def get_times(self):
        return self.times[~np.isnan(self.times)].copy()

    def save(self, path):
        path = path + '.csv'
        values = self.get_values()
        times = self.get_times()
        data = np.concatenate((times.reshape(-1, 1), values), axis=1)
        
        df = pd.DataFrame(data, columns=['t'] + [str(
            self.bin_vals[i]) for i in range(self.bins)])
        df.to_csv(path, index=False)

    def import_data(self, data):
        self.times = data.values[:, 0]
        self.values = data.values[:, 1:]
        self.bins = data.values[:, 1:].shape[1]
        self.bin_vals = np.array([float(col) for col in data.columns[1:]])
        self.start = self.bin_vals[0]
        self.end = self.bin_vals[-1]
        
        print(f'HistogramBuffer.import_data(): Imported data with')
        print(f'{self.values.shape = }')
        print(f'{self.times = }')
        print(f'{self.bins = }')
        print(f'{self.bin_vals = }')

    def plot(self, size=2, t=None, nplots=20, title=None, density=False, xscale=1, xlabel='x', ylabel='frequency'):
        values = self.get_values()
        start = self.start * xscale
        end = self.end * xscale
        xx = np.linspace(start, end, self.bins)
        if density:
            values = values / \
                np.sum(values, axis=1)[:, np.newaxis]
            ylabel = 'density'
        ymax = np.nanmax(values)
        if title is None:
            title = self.title
        if t is None:
            tt = self.get_times()
        elif isinstance(t, (int, float)):
            tt = [t]

        nrows = 5
        ncols = nplots//nrows + bool(nplots % nrows)
        fig, ax = plt.subplots(nrows, ncols, figsize=(size*nrows, size*ncols))
        fig.suptitle(title, fontsize=10)
        fig.subplots_adjust(hspace=0.5)
        if nplots == 1:
            ax = [ax]

        if len(tt) > nplots:
            tt = np.linspace(tt[0], tt[-1], nplots, dtype=int)

        for i, ax_i in enumerate(ax.flatten()):
            if i >= len(tt):
                ax_i.axis('off')
                continue
            t = tt[i]
            
            ax_i.bar(xx, values[t], width=(
                end - start) / self.bins, color='black', alpha=0.5)

            xticks = (xx[0], xx[-1])
            xticklabels = [f'{xt:.2e}' if isinstance(
                xt, float) else str(xt) for xt in xticks]
            ax_i.set_xticks(xticks)
            ax_i.set_xticklabels(xticklabels)
            ax_i.tick_params(axis='both', which='major', labelsize=7)
            
            ax_i.set_xlabel(xlabel, fontsize=7, labelpad=-8)
            ax_i.set_ylim(0, ymax*1.1)
            ax_i.set_ylabel(ylabel, fontsize=7)

            t = float(round(t, 2))
            ax_i.set_title(f'{t = }', fontsize=8)
        fig.tight_layout()
        return fig, ax

    def animate(self, size=6, title=None, density=False, xscale=1, xlabel='x', ylabel='frequency'):
        print('HistogramBuffer.animation(): Animating HistogramBuffer...')
        tt = self.get_times()
        time_step = tt[1] - tt[0]
        values = self.get_values()
        start = self.start * xscale
        end = self.end * xscale
        xx = np.linspace(start, end, self.bins)
        if density:
            values = values / \
                np.sum(values, axis=1)[:, np.newaxis]
            ylabel = 'density'
        ymax = np.nanmax(values)
        
        if title is None:
            title = self.title + ' Animation'
        
        fig, ax = plt.subplots(1, 1, figsize=(size, 2*size//3))
        # fig.tight_layout()

        def animate(i):
            ax.clear()
            ax.bar(xx, values[i], color='black', alpha=0.5, width=(
                end - start) / self.bins)
            ax.set_ylim(0, ymax*1.1)
            t = tt[i]
            t = float(round(t, 2))
            ax.set_title(title + f' at {t = }', fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return ax

        ani = animation.FuncAnimation(
            fig, animate, frames=len(tt), interval=10 * time_step, repeat=True)
        return ani
