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

path_kwargs = '../../default_kwargs.json'
with open(path_kwargs, 'r') as file:
    default_kwargs = json.load(file)


def rewrite_state_buffer_data(state_buffer_df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in state_buffer_df.keys():
        print('State buffer already rewritten')
        return state_buffer_df

    input('State buffer needs to be rewritten. Press Enter to continue... (or Ctrl+C to cancel)')
    state_buffer_df = pd.DataFrame(
        state_buffer_df.values, columns=['x', 'y', 'r', 't'])
    # generate id column and assign it to the dataframe
    ids = np.arange(state_buffer_df.shape[0])
    state_buffer_df = state_buffer_df.assign(id=ids)
    state_buffer_df = state_buffer_df[['id', 'x', 'y', 'r', 't']]
    return state_buffer_df


def rewrite_hist_buffer_data(hist_buffer_df: pd.DataFrame) -> pd.DataFrame:
    if hist_buffer_df.iloc[0, 0] == 'bins':
        hist_buffer_df = hist_buffer_df.drop(0)

    bin_in_keys = np.array(
        ['bin' in key for key in hist_buffer_df.iloc[0].values])
    if bin_in_keys.any():
        return hist_buffer_df

    N_bins = int(hist_buffer_df.shape[1] - 1)
    bin_range = (float(hist_buffer_df.iloc[0, 1]), float(
        hist_buffer_df.iloc[0, -1]))
    note = f"{N_bins =}, {bin_range =} "
    hist_buffer_df = hist_buffer_df.drop(1)

    keys = ['t'] + ['bin_' + str(i) for i in range(N_bins)]
    hist_buffer_df.columns = keys
    hist_buffer_df = hist_buffer_df.reset_index(drop=True)
    return hist_buffer_df, note


def rewrite_density_field_buffer_data(density_field_buffer_df: pd.DataFrame) -> pd.DataFrame:
    if 't' in density_field_buffer_df.keys():
        print('Density field buffer already rewritten')
        return density_field_buffer_df
    
    input('State buffer needs to be rewritten. Press Enter to continue... (or Ctrl+C to cancel)')
    old_keys = [float(d) for d in density_field_buffer_df.columns.tolist()]
    density_field_buffer_df.loc[-1] = old_keys  # Add old keys as the first line of data
    density_field_buffer_df.index = density_field_buffer_df.index + 1  # Shift index
    density_field_buffer_df = density_field_buffer_df.sort_index()  # Sort by index to place the old keys at the top

    keys = ['t'] + ['cell_' + str(i) for i in range(len(density_field_buffer_df.columns) - 1)]
    density_field_buffer_df.columns = keys
    return density_field_buffer_df


class DataBuffer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.batch_size = 1000

        if os.path.exists(self.file_path):
            print(f'DataBuffer.__init__(): Loading data from already existing file {self.file_path}.')
            self.columns = list(pd.read_csv(self.file_path).keys())
        else:
            self.columns = ['Time', 'Biomass', 'Population']
            self._initialize_file()

        self.buffer = pd.DataFrame(columns=self.columns, dtype=np.float64)

    def _initialize_file(self):
        # Check if the file path ends with '.csv', if not, add it
        if not self.file_path.endswith('.csv'):
            self.file_path += '.csv'

        # Raise an error if the file already exists, as to not override it
        if os.path.exists(self.file_path):
            raise FileExistsError(f'DataBuffer._initialize_file(): File {
                                  self.file_path} already exists.')

        # If the file does not exist, create it and write the column names to it
        else:
            print(f"DataBuffer._initialize_file(): Creating file {self.file_path}...")
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            empty_df = pd.DataFrame(columns=self.columns)
            with open(self.file_path, 'w') as f:
                empty_df.to_csv(f, index=False, float_format='%.18e')
            

    def add(self, data):
        if list(data.keys()) != list(self.columns):
            raise ValueError(f'DataBuffer.add(): Data keys {list(
                data.keys())} do not match the expected keys: {self.columns}.')
        if self.buffer.empty:
            self.buffer = data
        else:
            self.buffer = pd.concat([self.buffer, data], ignore_index=True)
        # print(f'DataBuffer.add(): Added data to buffer. {self.buffer=}')
        if self.buffer.shape[0] >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self):
        new_rows = pd.DataFrame(self.buffer, columns=self.columns, dtype=np.float64)
        print(f'\nDataBuffer._flush_buffer(): Flushing {new_rows.shape[0]} rows to file.', end='\n')

        with open(self.file_path, 'a', newline='') as f:
            new_rows.to_csv(f, header=False, index=False, float_format='%.18e')

        self.buffer = pd.DataFrame(columns=self.columns)
    
    def finalize(self):
        if not self.buffer.empty:
            self._flush_buffer()

    def get_data(self):
        return pd.read_csv(self.file_path)

    def plot(self, size=6, title='DataBuffer', keys=None):
        data = self.get_data()
        if len(data) < 2:
            print(f'DataBuffer.plot(): Not enough data to plot ({len(data) = }).')
            return
        # Specify which keys need to be plotted
        if keys is None:
            keys = [key for key in self.columns if key != 'Time']
        elif not all(key in self.columns for key in keys):
            raise ValueError(
                f'DataBuffer.plot(): Keys {keys} not found in {self.columns}.')
        
        # Create the figure and axes
        fig, ax = plt.subplots(
            len(keys), 1,
            figsize=(size, size * len(keys) / 3),
            sharex=True)

        # Set the title of the figure
        if title is not None:
            fig.suptitle(title, fontsize=10)

        fig.tight_layout(pad=3.0, h_pad=0.0)

        cmap = ListedColormap(
            ['#012626', '#1A402A', '#4B7340', '#7CA653', '#A9D962'])

        if isinstance(ax, plt.Axes):
            ax = np.array([ax])
        for i, key in enumerate(keys):
            x_data = data['Time']
            y_data = data[key]

            ax[i].plot(x_data, y_data,
                       label=key, color=cmap((i + 1) / len(keys)))
            y_max = np.nanmax(y_data)
            if y_max != 0:
                ax[i].set_ylim(-0.1 * y_max, 1.1 * y_max)
            ax[i].grid()
            ax[i].legend()
        ax[-1].set_xlabel('Time')

        for ax_i in ax:
            ax_i.grid()
            ax_i.legend()
        return fig, ax


class StateBuffer:
    def __init__(self, file_path, skip=1):
        self.file_path = file_path
        self.skip = skip
        self.batch_size = 100_000

        self.columns=['id', 'x', 'y', 'r', 't']
        if os.path.exists(self.file_path):
            print(f'StateBuffer.__init__(): Loading data from already existing file {self.file_path}.')
        else:
            self._initialize_file()
        
        self.buffer = pd.DataFrame(columns=self.columns, dtype=np.float64)

    def _initialize_file(self):
        # Check if the file path ends with '.csv', if not, add it
        if not self.file_path.endswith('.csv'):
            self.file_path += '.csv'
        
        # Raise an error if the file already exists, as to not override it
        if os.path.exists(self.file_path):
            raise FileExistsError(f'StateBuffer._initialize_file(): File {self.file_path} already exists.')
        # If the file does not exist, create it and write the column names to it
        else:
            print(f"StateBuffer._initialize_file(): Creating file {self.file_path}...")
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            empty_df = pd.DataFrame(columns=self.columns)
            with open(self.file_path, 'w') as f:
                empty_df.to_csv(f, index=False, float_format='%.18e')


    def add(self, plants, t):
        new_data = pd.DataFrame(
            [[plant.id, plant.x, plant.y, plant.r, t] for plant in plants],
            columns=['id', 'x', 'y', 'r', 't']
        )
        if new_data.empty:
            return
        
        if self.buffer.empty:
            self.buffer = new_data
        else:
            self.buffer = pd.concat([self.buffer, new_data], ignore_index=True)
        
        if self.buffer.shape[0] >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self):
        new_rows = pd.DataFrame(self.buffer, columns=self.columns, dtype=np.float64)
        print(f'\nStateBuffer._flush_buffer(): Flushing {new_rows.shape[0]} rows to file.', end='\n')
        
        with open(self.file_path, 'a', newline='') as f:
            new_rows.to_csv(f, header=False, index=False, float_format='%.18e')
        self.buffer = pd.DataFrame(columns=self.columns, dtype=np.float64)
    
    def finalize(self):
        if not self.buffer.empty:
            self._flush_buffer()

    def get_data(self):
        data = pd.read_csv(self.file_path)
        if 'id' not in data.keys():
            data = rewrite_state_buffer_data(data)
            self.override_data(data)
        return data
    
    def get_last_state(self):
        data = self.get_data()
        if data.empty:
            print('StateBuffer.get_last_state(): No data to return.')
            return pd.DataFrame(columns=self.columns)
        last_t = data['t'].unique()[-1]
        last_state = data[data['t'] == last_t]
        return last_state
    
    def override_data(self, data):
        with open(self.file_path, 'w', newline='') as f:
            data.to_csv(f, index=False, float_format='%.18e', header=True)
    
    def save(self, file_path):
        if not file_path.endswith('.csv'):
            file_path = file_path + '.csv'
        
        # Raise an error if the file already exists, as to not override it
        if os.path.exists(file_path):
            raise FileExistsError(f'StateBuffer.save(): File {file_path} already exists.')
        
        data = self.get_data()
        data.to_csv(file_path, index=False, float_format='%.18e')

    @staticmethod
    def plot_state(state, size=2, fig=None, ax=None, half_width=0.5, half_height=0.5, fast=False):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-half_height, half_height)
        ax.set_aspect('equal', 'box')
        
        ## JUST FOR DEBUGGING, REMOVE LATER
        if state['t'].nunique() > 1:
            warnings.warn('StateBuffer.plot_state(): More than one unique time in state.')
        
        t = state['t'].iloc[0]

        for id, x, y, r in state[['id', 'x', 'y', 'r']].values:
            if fast and r > 0.005:
                ax.add_artist(plt.Circle((x, y), r,
                                         color='green', fill=False, transform=ax.transData))
            elif not fast:
                ax.add_artist(plt.Circle((x, y), r,
                                         color='green', fill=True, transform=ax.transData))

        if t is not None:
            t = float(round(t, 2))
            ax.text(0.0, -0.6, f'{t = }', ha='center', fontsize=7)

        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot(self, size=2, title='StateBuffer', fast=False, n_plots=20):
        data = self.get_data()
        if data.empty:
            print('StateBuffer.plot(): No data to plot.')
            return
        if fast:
            print(
                'StateBuffer.plot(): Faster plotting is enabled, some elements might be missing in the plots.')
            title += ' (Fast)'
        times_unique = data['t'].unique()
        b = np.linspace(times_unique.min(), times_unique.max(), min(len(times_unique), n_plots))
        times = [times_unique[np.abs(times_unique - t).argmin()] for t in b]
        T = len(times)
        n_cols = int(np.ceil(np.sqrt(T)))
        n_rows = int(np.ceil(T / n_cols))

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
        fig.tight_layout()
        fig.suptitle(title, fontsize=8)
        if isinstance(ax, plt.Axes):
            ax = np.array([ax])
        for i, a in enumerate(ax.flatten()):
            if i >= T:
                a.axis('off')
                continue
            state = data[data['t'] == times[i]]
            # plot_state(state=state, size=size, ax=a, fast=fast)
            self.plot_state(state=state, size=size, ax=a, fast=fast)
        return fig, ax

    def animate(self, size=6, title=None, fast=False):
        print('StateBuffer.animation(): Animating StateBuffer...')
        data = self.get_data()
        times = data['t'].unique()
        states = [data[data['t'] == t] for t in times]
        time_step = times[1] - times[0]
        T = len(times)
        print(f'StateBuffer.animation(): {T = }')

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

        circles = [plt.Circle((x, y), r, color='green',
                              fill=not fast) for x, y, r in states[0][['x', 'y', 'r']].values]
        for circle in circles:
            ax.add_artist(circle)

        def animate(i):
            t = float(round(times[i], 2))
            ax.set_title(f'{t = }', fontsize = 8)
            for circle, (x, y, r) in zip(circles, states[i][['x', 'y', 'r']].values):
                circle.center = (x, y)
                circle.radius = r
            return ax

        ani = animation.FuncAnimation(
            fig, animate, frames=T, interval= 10 * time_step, repeat=True)
        return ani


class FieldBuffer:
    def __init__(self, file_path, resolution=100, skip=100):
        self.file_path = file_path
        self.resolution = resolution
        self.skip = skip
        self.batch_size = 100
        self.columns = ['t'] + [f'cell_{i}' for i in range(resolution * resolution)]

        if os.path.exists(self.file_path):
            print(f'FieldBuffer.__init__(): Loading data from already existing file {self.file_path}.')
        else:
            self._initialize_file()

        self.buffer = pd.DataFrame(columns=self.columns, dtype=np.float64)

    def _initialize_file(self):
        # Check if the file path ends with '.csv', if not, add it
        if not self.file_path.endswith('.csv'):
            self.file_path += '.csv'
        # Raise an error if the file already exists, as to not override it
        if os.path.exists(self.file_path):
            raise FileExistsError(f'FieldBuffer._initialize_file(): File {self.file_path} already exists.')
        # If the file does not exist, create it and write the column names to it
        else:
            print(f"FieldBuffer._initialize_file(): Creating file {self.file_path}...")
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            empty_df = pd.DataFrame(columns=self.columns)
            with open(self.file_path, 'w') as f:
                empty_df.to_csv(f, index=False, float_format='%.18e')

    def add(self, field, t):
        new_data = pd.DataFrame(
            np.hstack(([t], field.flatten())).reshape(1, -1),
            columns=self.columns
        )
        if self.buffer.empty:
            self.buffer = new_data
        else:
            self.buffer = pd.concat([self.buffer, new_data], ignore_index=True)
        if len(self.buffer) >= self.batch_size:
            self._flush_buffer()

    def _flush_buffer(self):
        new_rows = pd.DataFrame(self.buffer, columns=self.columns)
        print(f'\nFieldBuffer._flush_buffer(): Flushing {new_rows.shape[0]} rows to file.', end='\n')
        
        with open(self.file_path, 'a', newline='') as f:
            new_rows.to_csv(f, header=False, index=False, float_format='%.18e')
        self.buffer = pd.DataFrame(columns=self.columns)
    
    def finalize(self):
        if not self.buffer.empty:
            self._flush_buffer()

    def get_data(self):
        data = pd.read_csv(self.file_path)
        if 't' not in data.keys():
            data = rewrite_density_field_buffer_data(data)
            self.override_data(data)
        return data

    def get_fields(self):
        data = self.get_data()
        times = data['t'].unique()
        fields = [data[data['t'] == t].iloc[:, 1:].values.reshape(self.resolution, self.resolution) for t in times]
        return fields

    def override_data(self, data):
        with open(self.file_path, 'w', newline='') as f:
            data.to_csv(f, index=False, float_format='%.18e', header=True)

    def save(self, path):
        if not path.endswith('.csv'):
            path = path + '.csv'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = self.get_data()
        data.to_csv(path, index=False, float_format='%.18e')

    @staticmethod
    def plot_field(field, t=None, size=2, fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        ax.contour(field, levels=[1.0], colors=['r'], linewidths=[1], alpha=0.5)
        ax.imshow(field, origin='lower', cmap='Greys', vmin=vmin, vmax=vmax, extent=extent)
        if t is not None:
            t = float(round(t, 2))
            ax.text(0.0, -0.6, f'{t=}', ha='center', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot(self, size=2, n_plots=20, vmin=0, vmax=None, title='FieldBuffer', extent=[-0.5, 0.5, -0.5, 0.5]):
        data = self.get_data()
        if data.empty:
            print('FieldBuffer.plot(): No data to plot.')
            return
        times_unique = data['t'].unique()
        b = np.linspace(times_unique.min(), times_unique.max(), min(len(times_unique), n_plots))
        times = [times_unique[np.abs(times_unique - t).argmin()] for t in b]
        fields = np.array([data[data['t'] == t].iloc[:, 1:].values.reshape(self.resolution, self.resolution) for t in times])
        if vmax is None:
            vmax = np.nanmax(fields)
        T = len(times)
        n_cols = int(np.ceil(np.sqrt(T)))
        n_rows = int(np.ceil(T / n_cols))
        
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(size * n_cols, size * n_rows))
        fig.tight_layout()
        fig.suptitle(title, fontsize=8)
        if isinstance(ax, plt.Axes):
            ax = np.array([ax])
        for i, a in enumerate(ax.flatten()):
            if i >= T:
                a.axis('off')
                continue
            field = fields[i]
            t = times[i]
            # plot_field(field=field, t=t, size=size, ax=a, vmin=vmin, vmax=vmax, extent=extent)
            self.plot_field(field=field, t=t, size=size, ax=a, vmin=vmin, vmax=vmax, extent=extent)
        return fig, ax

    def animate(self, size=6, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        print('FieldBuffer.animation(): Animating FieldBuffer...')
        data = self.get_data()
        times = data['t'].unique()
        fields = [data[data['t'] == t].iloc[:, 1:].values.reshape(self.resolution, self.resolution) for t in times]
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
            ax.set_title(f'{t=}')
            ax.imshow(fields[i], origin='lower', cmap='Greys', vmin=vmin, vmax=vmax, extent=extent)
            ax.set_xticks([])
            ax.set_yticks([])
            return ax

        ani = animation.FuncAnimation(fig, animate, frames=T, interval=10 * time_step, repeat=True)
        plt.show()
        return ani


class HistogramBuffer:
    def __init__(self, size=10, bins=25, start=0, end=1, title='HistogramBuffer', data=None):
        self.title = title
        self.bins = bins
        self.bin_vals = np.linspace(start, end, bins+1)

        if data is not None:
            self.import_data(data, self.bin_vals)
            return

        self.values = np.full((size+1, bins), np.nan)
        self.times = np.full(size+1, np.nan)
        self.start = start
        self.end = end
        self.length = 0

    def add(self, data, t):
        hist, _ = np.histogram(data, bins=self.bin_vals)
        self.values[self.length] = hist
        self.times[self.length] = t
        self.length = self.length + 1

        size = self.values.shape[0]
        if len(self.values) > size:
            print(f'\nHistogramBuffer.add(): !Warning! HistogramBuffer is full, previous data will be overridten.')
            self.values.pop(0)
            self.length = size

    def extend(self, n):
        self.values = np.concatenate(
            (self.values, np.full((n, self.bins), np.nan)), axis=0)
        self.times = np.concatenate((self.times, np.full(n, np.nan)), axis=0)

    def get_values(self, t=None):
        values = self.values[~np.isnan(self.values).all(axis=1)].copy()
        if t is not None:
            return values[t]
        return values

    def get_times(self):
        return self.times[~np.isnan(self.times)].copy()

    def make_dataframe(self):
        values = self.get_values()
        times = self.get_times()
        data = np.concatenate((times.reshape(-1, 1), values), axis=1)

        hist_buffer_df = pd.DataFrame(data, columns=['t'] + [str(
            self.bin_vals[i]) for i in range(self.bins)])
        return hist_buffer_df

    def save(self, path):
        if not path.endswith('.csv'):
            path = path + '.csv'

        hist_buffer_df = self.make_dataframe()
        hist_buffer_df.to_csv(path, index=False)

    def import_data(self, data, bin_vals):
        self.times = data.values[:, 0]
        self.values = data.values[:, 1:]
        self.bins = data.values[:, 1:].shape[1]
        self.bin_vals = bin_vals
        self.start = self.bin_vals[0]
        self.end = self.bin_vals[-1]
        self.length = self.values.shape[0]

    def plot(self, size=2, t=None, nplots=20, title=None, density=False, xscale=1, xlabel='', ylabel='frequency'):
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
        ii = np.linspace(0, len(tt)-1, nplots, dtype=int)
        ii = np.unique(ii)

        if isinstance(ax, plt.Axes):
            ax = np.array([ax])
        for i, ax_i in enumerate(ax.flatten()):
            if i >= len(tt):
                ax_i.axis('off')
                continue
            idx = ii[i]
            t = tt[idx]

            ax_i.bar(xx, values[idx], width=(
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
            ax_i.set_title(f'{t=}', fontsize=8)
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
            ax.set_title(title + f' at {t=}', fontsize=12)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return ax

        ani = animation.FuncAnimation(
            fig, animate, frames=len(tt), interval=10 * time_step, repeat=True)
        return ani
