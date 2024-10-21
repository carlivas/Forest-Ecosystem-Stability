import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator
import copy

from plant import Plant
from density_field import DensityField


def check_pos_collision(pos, plant):
    return np.sum((pos - plant.pos) ** 2) < plant.r ** 2


def check_collision(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2) < (p1.r + p2.r) ** 2


class Simulation:
    def __init__(self, **kwargs):
        self.t = 0
        self.plants = []
        self.land_quality = kwargs.get('land_quality')

        self.half_width = kwargs.get('half_width')
        self.half_height = kwargs.get('half_height', self.half_width)
        self.density_field = DensityField(
            self.half_width,
            self.half_height,
            kwargs.get('density_check_radius'),
            kwargs.get('density_check_resolution')
        )

        self.kt_leafsize = kwargs.get('kt_leafsize')
        self.kt = None

        self.state_buffer = StateBuffer(
            size=kwargs.get('state_buffer_size', 100),
            skip=kwargs.get('state_buffer_skip', 1),
            preset_times=kwargs.get('state_buffer_preset_times', None)
        )

        self.data_buffer = DataBuffer(
            size=kwargs.get('n_iter'),
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

    def get_collisions(self, plant):
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
        self.state_buffer.add(state=self.get_state(), t=0)
        self.data_buffer.analyze_and_add(self.get_state(), t=0)

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

    def plot(self, size=6, t=None, highlight=None):
        if t is None:
            t = self.t
        fig, ax = self.plot_state(
            self.get_state(), t=t, size=size, highlight=highlight)
        return fig, ax

    def plot_state(self, state, t=None, size=6, fig=None, ax=None, highlight=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # ax.set_title('State')
        # ax.set_xlabel('Width (u)')
        # ax.set_ylabel('Height (u)')
        ax.set_xlim(-self.half_width, self.half_width)
        ax.set_ylim(-self.half_height, self.half_height)
        ax.set_aspect('equal', 'box')
        if t is not None:
            ax.set_title(f'{t = }', fontsize=7)
        else:
            ax.set_title('')
        for plant in state:
            if highlight is not None and plant.id in highlight:
                color = 'red'
            else:
                color = 'green'
            ax.add_artist(plt.Circle(plant.pos, plant.r,
                          color=color, fill=True, transform=ax.transData))

        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot_states(self, states, times=None, size=6):
        l = len(states)
        n_rows = int(np.floor(l / np.sqrt(l)))
        n_cols = (l + 1) // n_rows + (l % n_rows > 0)
        print(f'simulation.plot_states(): {n_rows = }, {n_cols = }')

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
    def __init__(self, init_field, size, skip=1, preset_times=None, data=None, **sim_kwargs):
        self.size = size    # estimated max number of fields to store
        self.skip = skip
        self.field = np.full((size, init_field.shape), np.nan)
        self.times = []

        if data is not None:
            self.import_data(
                data=data, plant_kwargs=plant_kwargs)

        self.preset_times = preset_times

    # def add(self, field, t):
    #     if len(self.fields) == 0 or field != self.fields[-1]:
    #         self.fields.append(field)
    #         self.times.append(t)
    #         if len(self.fields) > self.size:

    #             for i, time in enumerate(self.times):
    #                 if time not in self.preset_times:
    #                     self.fields.pop(i)
    #                     self.times.pop(i)
    #                     break

    #         print(
    #             f'\nFieldBuffer.add(): Added field at time {t}.')

    # def get(self, times=None):
    #     if times is None:
    #         return copy.deepcopy(self.fields)

    #     indices = []
    #     for t in times:
    #         if t < self.times[0] or t > self.times[-1]:
    #             print(
    #                 f'!Warning! FieldBuffer.get(): Time {t} is out of bounds. Start time: {self.times[0]}, End time: {self.times[-1]}')
    #             indices.append(np.nan)
    #         else:
    #             indices.append(np.where(np.array(self.times) == t)[0][0])
    #     return copy.deepcopy([self.fields[i] for i in indices if not np.isnan(i)])

    # def get_fields(self):
    #     return copy.deepcopy(self.fields)

    # def get_times(self):
    #     return copy.deepcopy(self.times)

    # def make_array(self):
    #     columns_per_plant = 3  # x, y, r

    #     L = 0

    #     for field in self.fields:
    #         for plant in field:
    #             if plant.id > L:
    #                 L = plant.id

    #     shape = (self.times[-1] + 1, (L + 1) * columns_per_plant)
    #     field_buffer_array = np.full(shape, np.nan)

    #     times = [t for t in self.times if t != np.nan]
    #     fields = self.get(times=times)
    #     for i, t in enumerate(times):
    #         field = fields[i]
    #         for plant in field:
    #             j = plant.id
    #             field_buffer_array[t, j *
    #                                columns_per_plant] =


class StateBuffer:
    def __init__(self, size=100, skip=10, preset_times=None, data=None, plant_kwargs=None):
        self.size = size
        self.skip = skip
        self.states = []
        self.times = []

        if data is not None:
            self.import_data(
                data=data, plant_kwargs=plant_kwargs)

        self.preset_times = preset_times

    def add(self, state, t):
        if len(self.states) == 0 or state != self.states[-1]:
            self.states.append(state)
            self.times.append(t)
            if len(self.states) > self.size:

                for i, time in enumerate(self.times):
                    if time not in self.preset_times:
                        self.states.pop(i)
                        self.times.pop(i)
                        break

            print(
                f'\nStateBuffer.add(): Added state at time {t}.')

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
        columns_per_plant = 3  # x, y, r

        L = 0

        for state in self.states:
            for plant in state:
                if plant.id > L:
                    L = plant.id

        shape = (self.times[-1] + 1, (L + 1) * columns_per_plant)
        state_buffer_array = np.full(shape, np.nan)

        times = [t for t in self.times if t != np.nan]
        states = self.get(times=times)
        for i, t in enumerate(times):
            state = states[i]
            for plant in state:
                j = plant.id
                state_buffer_array[t, j *
                                   columns_per_plant] = plant.pos[0]
                state_buffer_array[t, j *
                                   columns_per_plant + 1] = plant.pos[1]
                state_buffer_array[t, j*columns_per_plant + 2] = plant.r

        return state_buffer_array

    def save(self, path):
        import os
        import pickle
        # path = path + '.pkl'

        # # Create the directory if it doesn't exist
        # os.makedirs(os.path.dirname(path), exist_ok=True)

        # # Save the StateBuffer object to the specified path as pickle
        # with open(path, 'wb') as f:
        #     pickle.dump(self, f)

        path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_buffer_array = self.make_array()

        # Save the StateBuffer array object to the specified path as csv
        np.savetxt(path, state_buffer_array, delimiter=',')

    def import_data(self, data, plant_kwargs):
        states = []
        times = []
        for t in range(len(data)):
            if np.isnan(data[t]).all():
                continue
            else:
                times.append(t)
                state = []
                for i in range(0, len(data[t]), 3):
                    x, y, r = data[t, i:i+3]
                    if np.isnan(x) or np.isnan(y) or np.isnan(r):
                        continue
                    state.append(
                        Plant(pos=np.array([x, y]), r=r, **plant_kwargs))

                states.append(state)
        self.states = states
        self.times = times

    def plot_state(self, state, t=None, size=6, fig=None, ax=None, half_width=0.5, half_height=0.5):
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

        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot(self, size=6):
        states = self.get_states()
        times = self.get_times()
        l = len(states)
        n_rows = int(np.floor(l / np.sqrt(l)))
        n_cols = (l + 1) // n_rows + (l % n_rows > 0)
        print(f'simulation: {n_rows = }, {n_cols = }')

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.05, hspace=0.05)
        fig.tight_layout()
        if len(states) == 1:
            self.plot_state(state=states[0], t=times[0], size=size, ax=ax)
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


class DataBuffer:
    def __init__(self, size=None, data=None):
        if data is not None:
            self.buffer = data
            self.length = len(data)
            self.size = len(data)
        else:
            self.size = size
            self.buffer = np.full((size, 3), np.nan)
            self.length = 0

    def add(self, data, t):
        self.buffer[t] = [t, *data]

        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        self.length = self.length + 1

    def analyze_state(self, state, t):
        biomass = sum([plant.area for plant in state])
        population_size = len(state)
        data = np.array([biomass, population_size])
        print(' '*36 +
              f'DataBuffer.analyze_state(): {t = }     |     P = {population_size}     |     B = {np.round(biomass, 5)}', end='\r')
        self.add(data, t)
        return data

    def analyze_and_add(self, state, t):
        data = self.analyze_state(state, t)
        self.add(data, t)
        return data

    def finalize(self):
        self.buffer = self.buffer[:self.length-1]

    def plot(self, size=6):
        fig, ax = plt.subplots(2, 1, figsize=(
            size, size))
        fig.tight_layout(pad=3.0)
        ax[0].plot(self.buffer[:, 0], self.buffer[:, 1],
                   label='Biomass', color='green')
        # ax[0].set_xticks([])
        ax[1].plot(self.buffer[:, 0], self.buffer[:, 2],
                   label='Population Size', color='teal')
        ax[1].set_xlabel('Time')

        for ax_i in ax:
            ax_i.grid()
            ax_i.legend()
        return fig, ax

    def get_data(self, indices=None):

        if indices is None:
            return copy.deepcopy(self.buffer)
        return copy.deepcopy([self.buffer[i] for i in indices])

    def save(self, path):
        import os
        import pickle
        path = path + '.csv'

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the DataBuffer object to the specified path as csv
        np.savetxt(path, self.buffer, delimiter=',')
        # with open(path, 'wb') as f:
        #     pickle.dump(self, f)

    # def load(self, path):
    #     import os
    #     import pickle

    #     # Create the directory if it doesn't exist
    #     os.makedirs(os.path.dirname(path), exist_ok=True)

    #     # Load the DataBuffer object from the specified path
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)
    #     return data
