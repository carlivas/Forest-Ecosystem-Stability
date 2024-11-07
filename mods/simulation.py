import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.colors import Normalize

from mods.plant import Plant
from mods.density_field import DensityField
from mods.data_buffer import DataBuffer
from mods.field_buffer import FieldBuffer
from mods.state_buffer import StateBuffer


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

        # density_field_resolution = np.ceil((2*self.half_width) /
        #                                    (np.sqrt(2) * kwargs.get('density_check_radius'))).astype(int)
        # density_field_resolution = max(25, density_field_resolution)
        # print(f'Simulation.__init__(): {density_field_resolution=}')
        self.density_field = DensityField(
            half_width=self.half_width,
            half_height=self.half_height,
            check_radius=kwargs.get('density_check_radius'),
            resolution=kwargs.get('density_field_resolution'),
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

    def quality_nearby(self, pos):
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
