import copy
import os
import numpy as np
import matplotlib.pyplot as plt


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
                data=data, sim_kwargs=sim_kwargs)
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

    def get(self, times=None):
        if times is None:
            return copy.deepcopy(self.fields)
        else:
            indices = []
            for t in times:
                if t not in self.times:
                    print(
                        f'!Warning! FieldBuffer.get(): Time {t} not in field buffer. Times: {self.times}')
                    indices.append(np.nan)
                else:
                    indices.append(np.where(np.array(self.times) == t)[0][0])
            return copy.deepcopy([self.fields[i] for i in indices if not np.isnan(i)])

    def get_fields(self):
        return copy.deepcopy(self.fields)

    def get_times(self):
        return copy.deepcopy(self.times)

    def import_data(self, data, sim_kwargs):
        times = data[:, 0].astype(int)
        fields_arr = data[:, 1:]

        self.resolution = int(np.sqrt(data.shape[-1]))
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

    def plot_field(self, field, time, size=2, fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        ax.contour(field, levels=[1.0], colors=[
                   'r'], linewidths=[1], alpha=0.5)
        ax.imshow(field, origin='lower', cmap='Greys',
                  vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(f't = {time}', fontsize=7)

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
            self.plot_field(fields[0], time=times[0],
                            size=size, fig=fig, ax=ax, vmin=vmin, vmax=vmax, extent=extent)
        else:
            for i in range(T):
                field = fields[i]
                if n_rows == 1:
                    k = i
                    self.plot_field(
                        field=field, time=times[i], size=size, fig=fig, ax=ax[k], vmin=vmin, vmax=vmax, extent=extent)
                else:
                    l = i // n_cols
                    k = i % n_cols
                    self.plot_field(
                        field=field, time=times[i], size=size, fig=fig, ax=ax[l, k], vmin=vmin, vmax=vmax, extent=extent)

        if T < n_rows*n_cols:
            for j in range(T, n_rows*n_cols):
                if n_rows == 1:
                    ax[j].axis('off')
                else:
                    l = j//n_cols
                    k = j % n_cols
                    ax[l, k].axis('off')

        return fig, ax
