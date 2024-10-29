import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
import copy
import os
import pickle

from plant import Plant
from density_field import DensityField


def check_pos_collision(pos, plant):
    return np.sum((pos - plant.pos) ** 2) < plant.r ** 2


def check_collision(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2) < (p1.r + p2.r) ** 2


class Simulation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        self.t = 0
        self.plants = []
        self.land_quality = kwargs.get('land_quality')

        self.half_width = kwargs.get('half_width')
        self.half_height = kwargs.get('half_height', self.half_width)
        self.kt_leafsize = kwargs.get('kt_leafsize')
        self.kt = None

        self.state_buffer = StateBuffer(
            size=kwargs.get('state_buffer_size', 20),
            skip=kwargs.get('state_buffer_skip', 1),
            preset_times=kwargs.get('state_buffer_preset_times', None)
        )

        self.data_buffer = DataBuffer(
            size=kwargs.get('n_iter'),
        )

        self.density_field = DensityField(
            self.half_width,
            self.half_height,
            kwargs.get('density_check_radius'),
            kwargs.get('density_field_resolution')
        )

        self.density_field_buffer = FieldBuffer(
            sim=self,
            resolution=kwargs.get('density_field_resolution'),
            size=kwargs.get('density_field_buffer_size'),
            skip=kwargs.get('density_field_buffer_skip', 1),
            preset_times=kwargs.get('density_field_buffer_preset_times', None)
        )

    def add(self, plant):

        if isinstance(plant, Plant):
            self.plants.append(plant)

        elif isinstance(plant, (list, np.ndarray)):
            for p in plant:
                if isinstance(p, Plant):
                    self.plants.append(p)
                else:
                    raise ValueError(
                        "All elements in the array must be Plant objects")
        else:
            raise ValueError(
                "Input must be a Plant object or an array_like of Plant objects")

    def update_kdtree(self):
        if len(self.plants) == 0:
            self.kt = None
        else:
            self.kt = KDTree(
                [plant.pos for plant in self.plants], leafsize=self.kt_leafsize)

    def update_density_field(self):
        self.density_field.update(self)

    def step(self):
        # First Phase: Update all plants
        for plant in self.plants:
            plant.update(self)

        # Second Phase: Collect non-dead plants and add them to the new state, and make sure all new plants get a unique id
        new_plants = []
        plant_ids = [plant.id for plant in self.plants]

        for plant in self.plants:
            if not plant.is_dead:
                if plant.id is None:
                    plant.id = max(
                        [id for id in plant_ids if id is not None]) + 1
                new_plants.append(plant)

        self.plants = new_plants

        self.t += 1

        # Update nessessary data structures
        self.update_kdtree()
        self.update_density_field()

        self.data_buffer.analyze_and_add(self.get_state(), t=self.t)

        if self.t % self.state_buffer.skip == 0 or self.t in self.state_buffer.preset_times:
            self.state_buffer.add(state=self.get_state(), t=self.t)

        if self.t % self.density_field_buffer.skip == 0 or self.t in self.density_field_buffer.preset_times:
            self.density_field_buffer.add(
                field=self.density_field.get_values(), t=self.t)

    def run(self, n_iter=None):
        import time
        if n_iter is None:
            n_iter = self.kwargs.get('n_iter')
        start_time = time.time()
        try:

            for _ in range(1, n_iter):

                self.step()

                # if no plants are left or if the number of plants exceeds 100 times the number of plants in the initial state, stop the simulation
                l = len(self.plants)
                if l == 0 or l > self.kwargs.get('num_plants') * 100:

                    break

                elapsed_time = time.time() - start_time

                if _ % 3 == 0:
                    dots = '.  '
                elif _ % 3 == 1:
                    dots = '.. '
                else:
                    dots = '...'

                print(f'{dots} Elapsed time: {elapsed_time:.2f}s', end='\r')

        except KeyboardInterrupt:

            print('\nInterrupted by user...')

    def get_collisions(self, plant):
        if self.kt is None:
            return []
        plant.is_colliding = False
        collisions = []
        indices = self.kt.query_ball_point(
            x=plant.pos, r=plant.d, workers=-1)
        for i in indices:
            other_plant = self.plants[i]
            if other_plant != plant:
                if check_collision(plant, other_plant):
                    plant.is_colliding = True
                    other_plant.is_colliding = True
                    collisions.append(other_plant)
        return collisions

    def site_quality(self, pos):
        # if position is in bounds, return the density at that position
        if np.abs(pos[0]) > self.half_width or np.abs(pos[1]) > self.half_height:
            return 0
        else:
            density_nearby = self.density_field.query(pos)
            return density_nearby + self.land_quality

    def get_state(self):
        return copy.deepcopy(self.plants)

    def get_density_field(self):
        return copy.deepcopy(self.density_field)

    def initiate(self):
        self.update_kdtree()
        self.update_density_field()

        self.data_buffer.analyze_and_add(state=self.get_state(), t=0)
        self.state_buffer.add(state=self.get_state(), t=0)
        self.density_field_buffer.add(
            field=self.density_field.get_values(), t=0)

    def initiate_uniform_lifetimes(self, n, t_min, t_max, **plant_kwargs):
        growth_rate = plant_kwargs['growth_rate']
        plant_kwargs['r_max'] = t_max * growth_rate
        plants = [
            Plant(
                pos=np.random.uniform(-self.half_width, self.half_width, 2),
                r=np.random.uniform(t_min, t_max) * growth_rate, id=i,
                **plant_kwargs
            )
            for i in range(n)
        ]
        self.add(plants)
        self.initiate()

    def plot(self, size=2, t=None, highlight=None):
        if t is None:
            t = self.t
        fig, ax = self.plot_state(
            self.get_state(), t=t, size=size, highlight=highlight)
        return fig, ax

    def plot_state(self, state, t=None, size=2, fig=None, ax=None, highlight=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # ax.set_title('State')
        # ax.set_xlabel('Width (u)')
        # ax.set_ylabel('Height (u)')
        ax.set_xlim(-self.half_width, self.half_width)
        ax.set_ylim(-self.half_height, self.half_height)
        ax.set_aspect('equal', 'box')
        if t is not None:
            ax.set_title(f'{t=}', fontsize=7)
        else:
            ax.set_title('')
        for plant in state:
            if highlight is not None and plant.id in highlight:
                color = 'red'
            else:
                color = 'green'
            density = self.density_field.query(plant.pos)
            ax.add_artist(plt.Circle(plant.pos, plant.r,
                          color=color, fill=True, transform=ax.transData))

            sm = plt.cm.ScalarMappable(
                norm=Normalize(vmin=0, vmax=self.density_field.values.max()), cmap='Greys')
            color = sm.to_rgba(density)
            ax.add_artist(plt.Circle(plant.pos, plant.r, fill=True,
                          color=color, alpha=1, transform=ax.transData))

        _m = self.kwargs.get('_m')
        print(f'Simulation.plot_state(): {ax.get_xticks()=}')
        print(f'Simulation.plot_state(): {_m=}')
        x_ticks = ax.get_xticks() * _m
        y_ticks = ax.get_yticks() * _m
        ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
        ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
        return fig, ax

    def plot_states(self, states, times=None, size=2):
        l = len(states)
        n_rows = int(np.floor(l / np.sqrt(l)))
        n_cols = (l + 1) // n_rows + (l % n_rows > 0)
        print(f'simulation.plot_states(): {n_rows=}, {n_cols=}')

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.05, hspace=0.05)
        fig.tight_layout()
        if len(states) == 1:
            self.plot_state(states[0], t=times[0], size=size, ax=ax)
        else:
            i = 0
            # for i, state in enumerate(states):
            while i < len(states):
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

                i += 1
        if len(states) < n_rows*n_cols:
            for j in range(len(states), n_rows*n_cols):
                if n_rows == 1:
                    ax[j].axis('off')
                else:
                    l = j//n_cols
                    k = j % n_cols
                    ax[l, k].axis('off')

        return fig, ax


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

    # def add(self, field, t):
    #     fields = self.get_fields()
    #     if len(self.times) == self.size:

    #         for i, time in enumerate(self.times):

    #             if time not in self.preset_times:
    #                 self.fields[i] = field
    #                 self.times.append(t)
    #                 print(f'\n    FieldBuffer.add(): Added field at time {t}.')

    #                 self.fields = np.concatenate(
    #                     (fields[:i], np.roll(fields[i:], -1, axis=0)), axis=0)
    #                 self.times.pop(i)

    #                 print(
    #                     f'\n    FieldBuffer.add(): Removed field at time {time}.')
    #                 break
    #     else:
    #         self.fields[len(self.times)] = field
    #         self.times.append(t)
    #         print(f'\n    FieldBuffer.add(): Added field at time {t}.')

    def add(self, field, t):
        if len(self.times) == 0 or t != self.times[-1]:
            if len(self.times) >= self.size:
                for i, time in enumerate(self.times):
                    if time not in self.preset_times:
                        fields = self.get_fields()
                        B = np.roll(fields[i:], -1, axis=0)
                        fields = np.concatenate(
                            (fields[:i], B), axis=0)
                        self.fields = fields
                        self.times.pop(i)
                        print(
                            f'\n    FieldBuffer.add(): Removed field at time {time}.')
                        break

            self.fields[len(self.times)] = field
            self.times.append(t)
            print(
                f'\n    FieldBuffer.add(): Added field at time {t}.')

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
        ax.imshow(field.T, origin='lower', cmap='Greys',
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
        print(f'FieldBuffer.plot(): {T=}')

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


class StateBuffer:
    def __init__(self, size=100, skip=10, sim=None, preset_times=None, data=None, plant_kwargs=None):
        self.sim = sim
        self.size = size
        self.skip = skip
        self.states = []
        self.times = []

        if data is not None:
            self.import_data(
                data=data, plant_kwargs=plant_kwargs)

        self.preset_times = preset_times

    def add(self, state, t):
        if len(self.times) == 0 or t != self.times[-1]:
            if len(self.times) >= self.size:
                for i, time in enumerate(self.times):
                    if time not in self.preset_times:
                        self.states.pop(i)
                        self.times.pop(i)
                        print(
                            f'\n    StateBuffer.add(): Removed state at time {time}.')
                        break

            self.states.append(state)
            self.times.append(t)
            print(
                f'\n    StateBuffer.add(): Added state at time {t}.')

    def get(self, times=None):
        if times is None:
            return copy.deepcopy(self.states)

        indices = []
        for t in times:
            if t < self.times[0] or t > self.times[-1]:
                print(
                    f'!Warning! StateBuffer.get(): Time {t} is out of bounds. Start time: {self.times[0]}, End time: {self.times[-1]}')
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
            x, y, r, t, id = data[i]
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


class DataBuffer:
    def __init__(self, size=None, data=None):
        if data is not None:
            self.values = data
            self.length = len(data)
            self.size = len(data)
        else:
            self.size = size
            self.values = np.full((size, 3), np.nan)
            self.length = 0

    def add(self, data, t):
        self.values[t] = [t, *data]

        if len(self.values) > self.size:
            self.values.pop(0)
        self.length = self.length + 1

    def analyze_state(self, state, t):
        biomass = sum([plant.area for plant in state])
        population_size = len(state)
        data = np.array([biomass, population_size])
        print(' '*45 +
              f'|\tt = {t:^5}    |    P = {population_size:^6}    |    B = {np.round(biomass, 5):^5}', end='\r')
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
