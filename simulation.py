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

        # Second Phase: Collect non-dead plants and add them to the new state
        new_plants = []
        for plant in self.plants:
            if not plant.is_dead:
                new_plants.append(plant)
        self.plants = new_plants

        # Update nessessary data structures
        self.update_kdtree()
        self.update_density_field()

        self.data_buffer.analyze_and_add(self.get_state(), self.t)

        self.t += 1
        if self.t % self.state_buffer.skip == 0 or self.t in self.state_buffer.preset_times:
            self.state_buffer.add(self.get_state(), t=self.t)

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

    def initiate_uniform_lifetimes(self, n, t_min, t_max, **plant_kwargs):
        growth_rate = plant_kwargs['growth_rate']
        plant_kwargs['r_max'] = t_max * growth_rate
        plants = [
            Plant(
                pos=np.random.uniform(-self.half_width, self.half_width, 2),
                r=np.random.uniform(t_min, t_max) * growth_rate,
                **plant_kwargs
            )
            for _ in range(n)
        ]
        self.add(plants)
        self.update_kdtree()
        self.update_density_field()
        # self.state_buffer.append(self.get_state())
        self.state_buffer.add(self.get_state(), t=0)

    # def initiate(self, n, **plant_kwargs):
    #     d = 2
    #     dist_min = plant_kwargs['r_min']**d
    #     dist_max = plant_kwargs['r_max']**d
    #     plants = [
    #         Plant(
    #             pos=np.random.uniform(-half_width, half_width, 2),
    #             r=np.random.uniform(dist_min, dist_max)**(1/d),
    #             **plant_kwargs
    #         )
    #         for _ in range(n)
    #     ]

    def plot(self, size=6):
        plot_state(self.get_state(), size)
        return fig, ax

    def plot_state(self, state, t=None, size=6, fig=None, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        # ax.set_title('State')
        # ax.set_xlabel('Width (u)')
        # ax.set_ylabel('Height (u)')
        ax.set_xlim(-self.half_width, self.half_width)
        ax.set_ylim(-self.half_height, self.half_height)
        ax.set_aspect('equal', 'box')
        for plant in state:
            if t is not None:
                ax.set_title(f't = {t}', fontsize=7)
            ax.add_artist(plt.Circle(plant.pos, plant.r,
                          color='green', fill=True, transform=ax.transData))

        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def plot_states(self, states, times=None, size=6):
        l = len(states)
        n_rows = int(np.floor(l / np.sqrt(l)))
        n_cols = (l + 1) // n_rows + (l % n_rows > 0)
        print(f'simulation: {n_rows = }, {n_cols = }')

        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
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


class StateBuffer:
    def __init__(self, size, skip=1, preset_times=None):
        self.size = size
        self.skip = skip
        self.states = []
        self.times = []
        self.length = 0

        self.start_time = 0
        self.preset_times = preset_times

    def add(self, state, t):
        if len(self.states) == 0 or state != self.states[-1]:
            self.states.append(state)
            self.times.append(t)
            if len(self.states) > self.size:
                self.states.pop(0)
                self.start_time = self.start_time + self.skip
            print(
                f'\nStateBuffer: Added state at time {t}.')
            self.length = len(self.states)

    def get(self, times=None):
        if times is None:
            return copy.deepcopy(self.states)

        indices = []
        for t in times:
            if t < self.start_time or t >= self.start_time + self.length:
                print(
                    f'!Warning! StateBuffer: Time {t} is out of bounds. Start time: {self.start_time}, End time: {self.start_time + self.length}')
                indices.append(np.nan)
            else:
                indices.append(t - self.start_time)

        print('StateBuffer: buffer_start_time = ', self.start_time)
        print(f'{indices = }')

        return copy.deepcopy([self.states[i] for i in indices if not np.isnan(i)])

    def get_states(self):
        return copy.deepcopy(self.states)

    def get_times(self):
        return copy.deepcopy(self.times)


class DataBuffer:
    def __init__(self, size):
        self.size = size
        # self.times = np.arange(size)
        # self.buffer = np.zeros((size, 2))
        self.buffer = np.full((size, 2), np.nan)
        self.times = np.full(size, np.nan)
        self.length = 0

    def add(self, data, time):
        self.buffer[time] = data
        self.times[time] = time

        if len(self.buffer) > self.size:
            self.buffer.pop(0)
        self.length = self.length + 1

    def analyze_state(self, state, time):
        biomass = sum([plant.area for plant in state])
        population_size = len(state)
        data = np.array([biomass, population_size])
        print(' '*40 +
              f'DataBuffer: {time = }     |     P = {population_size}     |     B = {np.round(biomass, 5)}', end='\r')
        self.add(data, time)
        return data

    def analyze_and_add(self, state, time):
        data = self.analyze_state(state, time)
        self.add(data, time)
        return data

    def finalize(self):
        print(self.length)
        self.buffer = self.buffer[:self.length]
        self.times = self.times[:self.length]

    def get_data(self, indices=None):

        if indices is None:
            return copy.deepcopy(self.buffer)
        return copy.deepcopy([self.buffer[i] for i in indices])

    def plot(self, size=6):
        fig, ax = plt.subplots(2, 1, figsize=(size, size))
        fig.tight_layout()
        ax[0].plot(self.times, self.buffer[:, 0],
                   label='Biomass', color='green')
        # ax[0].set_xticks([])
        ax[1].plot(self.times, self.buffer[:, 1],
                   label='Population Size', color='teal')
        ax[1].set_xlabel('Time')

        for ax_i in ax:
            ax_i.grid()
            ax_i.legend()
        return fig, ax
