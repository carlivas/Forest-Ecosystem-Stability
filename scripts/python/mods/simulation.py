from typing import List, Union, Optional, Any
import time
import copy
from typing import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from matplotlib.colors import Normalize
from scipy.stats import gaussian_kde

from mods.plant import Plant
from mods.fields import DensityFieldSPH as DensityField
from mods.buffers import DataBuffer, FieldBuffer, StateBuffer


def check_collision(p1: np.ndarray, p2: np.ndarray, r1: float, r2: float) -> bool:
    return np.sum((p1 - p2) ** 2) < (r1 + r2) ** 2


def dbh_to_crown_radius(dbh: float) -> float:
    # everything in m
    d = 1.42 + 28.17*dbh - 11.26*dbh**2
    return d/2


def _m_from_m2pp(m2pp, num_plants, A_bound=1) -> float:
    return np.sqrt(A_bound/(m2pp*num_plants))


def _m_from_domain_sides(L, S_bound=1) -> float:
    return S_bound / L


class Simulation:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.spinning_up = False
        self.verbose = kwargs.get('verbose', True)

        self.t = 0
        self.max_population = 10000

        self.state = np.full(self.max_population, np.nan, dtype=[
            ('pos', float, 2), ('r', float), ('area', float), ('is_dead', bool)])
        self.state['is_dead'] = 1
        self.num_plants = 0
        self.land_quality = kwargs['land_quality']
        self.spawn_rate = kwargs.get('spawn_rate', 0)

        self.species_germination_chance = 1

        precipitation = kwargs['precipitation']
        if isinstance(precipitation, float):
            self.precipitation = lambda t: precipitation
        elif isinstance(precipitation, int):
            self.precipitation = lambda t: precipitation
        elif callable(precipitation):
            self.precipitation = precipitation
        else:
            raise ValueError('Precipitation must be a float, int or callable')

        self.half_width = kwargs.get('half_width', 0.5)
        self.half_height = kwargs.get('half_height', self.half_width)

        self._m = 2*self.half_width / kwargs['L']
        kwargs['_m'] = self._m

        r_min = kwargs.get('r_min', 0.1)
        r_max = kwargs.get('r_max', 30)
        growth_rate = kwargs.get('growth_rate', 0.1)

        kwargs['r_min'] = r_min * self._m
        kwargs['r_max'] = r_max * self._m
        self.r_max = r_max * self._m
        kwargs['growth_rate'] = growth_rate * self._m
        self.growth_rate = growth_rate * self._m
        kwargs['dispersal_range'] = kwargs['dispersal_range'] * self._m

        self.kt = None

        buffer_size = kwargs.get('buffer_size', 500)
        buffer_skip = kwargs.get('buffer_skip', 20)
        buffer_preset_times = kwargs.get(
            'buffer_preset_times', (np.arange(buffer_size) * buffer_skip).astype(int))

        self.data_buffer = DataBuffer(
            sim=self, size=kwargs.get('n_iter', 10000))
        self.state_buffer = StateBuffer(size=kwargs.get('state_buffer_size', buffer_size), skip=kwargs.get(
            'state_buffer_skip', buffer_skip), preset_times=kwargs.get('state_buffer_preset_times', buffer_preset_times))
        self.density_field_buffer = FieldBuffer(sim=self, resolution=kwargs.get('density_field_resolution', 100), size=kwargs.get('density_field_buffer_size', buffer_size), skip=kwargs.get(
            'density_field_buffer_skip', buffer_skip), preset_times=kwargs.get('density_field_buffer_preset_times', buffer_preset_times))

        self.density_field = DensityField(half_width=self.half_width, half_height=self.half_height, check_radius=kwargs.get(
            'density_check_radius', 100 * self._m), resolution=kwargs.get('density_field_resolution', 100), simulation=self)

    def add(self, plants: np.ndarray) -> None:
        if self.num_plants + len(plants) >= self.max_population:
            self._resize_state()
        self.state[self.num_plants:self.num_plants + len(plants)] = plants
        self.num_plants += len(plants)

    def _resize_state(self):
        self.max_population *= 2
        new_state = np.full(self.max_population, np.nan,
                            dtype=self.state.dtype)
        new_state[:self.num_plants] = self.state
        self.state = new_state

    def update_kdtree(self) -> None:
        if self.num_plants == 0:
            self.kt = None
        else:
            self.kt = KDTree(self.state['pos'][:self.num_plants], leafsize=10)

    def update_plants(self) -> None:
        # Growth
        self.state['r'][:self.num_plants] += self.growth_rate

        # Collisions
        pairs = np.array(list(self.kt.query_pairs(
            r=self.r_max*2)))
        if pairs.size > 0:
            p1 = self.state['pos'][pairs[:, 0]]
            p2 = self.state['pos'][pairs[:, 1]]
            r1 = self.state['r'][pairs[:, 0]]
            r2 = self.state['r'][pairs[:, 1]]
            collisions = check_collision(p1, p2, r1, r2)
            self.state['is_dead'][pairs[collisions, 0]
                                  ] = r1[collisions] < r2[collisions]
            self.state['is_dead'][pairs[collisions, 1]
                                  ] = r1[collisions] >= r2[collisions]

        # Dispersal
        alive_indices = np.where(~self.state['is_dead'][:self.num_plants])[0]
        new_positions = self.state['pos'][alive_indices] + np.random.normal(
            0, self.kwargs['dispersal_range'], size=(len(alive_indices), 2))
        dispersal_chances = np.maximum(self.land_quality, self.local_density(
            new_positions) * self.precipitation(self.t) * self.species_germination_chance)
        spawn_indices = np.where(
            dispersal_chances > np.random.uniform(0, 1, len(alive_indices)))[0]

        new_plants = np.zeros(len(spawn_indices), dtype=self.state.dtype)
        new_plants['pos'] = new_positions[spawn_indices]
        new_plants['r'] = self.kwargs['r_min']
        new_plants['area'] = np.pi * self.kwargs['r_min']**2
        new_plants['is_dead'] = False

        # Mortality
        mortality_indices = np.where(
            self.state['r'][:self.num_plants] > self.kwargs['r_max'])[0]
        self.state['is_dead'][mortality_indices] = True

        # Remove dead plants efficiently
        alive_indices = np.where(~self.state['is_dead'][:self.num_plants])[0]
        self.state[:len(alive_indices)] = self.state[alive_indices]
        self.state[len(alive_indices):] = np.nan
        self.num_plants = len(alive_indices)

        self.add(new_plants)

    def step(self) -> None:
        n = self.spawn_rate
        self.attempt_spawn(n=n, **self.kwargs)

        self.update_plants()

        self.t += 1

        self.update_kdtree()
        self.density_field.update(self.state)

        population = self.num_plants
        biomass = np.sum(self.state['area'][:self.num_plants])
        precipitation = self.precipitation(self.t)
        data = np.array([biomass, population, precipitation])
        self.data_buffer.add(data=data, t=self.t)

        if self.verbose:
            print(' '*30 + f'|  t = {self.t:<6}  |  N = {population:<6}  |  B = {
                  np.round(biomass, 4):<6}  |  P = {np.round(precipitation, 4):<6}', end='\r')

        do_save_state = self.t % self.state_buffer.skip == 0 or self.t in self.state_buffer.preset_times
        do_save_density_field = self.t % self.density_field_buffer.skip == 0 or self.t in self.density_field_buffer.preset_times

        if do_save_state:
            self.state_buffer.add(state=self.get_state(), t=self.t)
        if do_save_density_field:
            self.density_field_buffer.add(
                field=self.density_field.get_values(), t=self.t)

    def run(self, n_iter: int) -> None:
        start_time = time.time()
        try:
            for _ in range(1, n_iter):
                self.step()

                if (self.num_plants > self.max_population):
                    break

                elapsed_time = time.time() - start_time
                hours, rem = divmod(elapsed_time, 3600)
                minutes, seconds = divmod(rem, 60)
                elapsed_time_str = f"{str(int(hours))}".rjust(
                    2, '0') + ":" + f"{str(int(minutes))}".rjust(2, '0') + ":" + f"{str(int(seconds))}".rjust(2, '0')

                if _ % 3 == 0:
                    dots = '.  '
                elif _ % 3 == 1:
                    dots = '.. '
                else:
                    dots = '...'

                print(f'{dots} Elapsed time: {elapsed_time_str}', end='\r')

        except KeyboardInterrupt:
            print('\nInterrupted by user...')

        print()
        print(f'Simulation.run(): Done. Elapsed time: {elapsed_time_str}')
        print()

    def attempt_spawn(self, n: int, **kwargs: Any) -> None:
        new_positions = np.random.uniform(-self.half_width,
                                          self.half_width, (n, 2))
        densities = self.density_field.query(new_positions)
        random_values = np.random.uniform(0, 1, n)
        probabilities = np.clip(
            densities * self.precipitation(self.t), self.land_quality, 1)
        spawn_indices = np.where(probabilities > random_values)[0]
        new_plants = np.full(n, np.nan, dtype=[
            ('pos', float, 2), ('r', float), ('area', float), ('is_dead', bool)])
        if len(spawn_indices) > 0:
            new_plants['pos'] = new_positions[spawn_indices]
            new_plants['r'] = kwargs['r_min']
            new_plants['area'] = np.pi * kwargs['r_min']**2
            new_plants['is_dead'] = False

            self.add(new_plants)

    def pos_in_box(self, pos: np.ndarray) -> np.ndarray:
        return (np.abs(pos[:, 0]) < self.half_width) & (np.abs(pos[:, 1]) < self.half_height)

    def local_density(self, pos: np.ndarray) -> np.ndarray:
        in_box = self.pos_in_box(pos)
        densities = np.zeros(pos.shape[0])
        densities[in_box] = self.density_field.query(pos[in_box]).flatten()
        return densities

    def get_state(self):
        return copy.deepcopy(self.state[:self.num_plants])

    def initiate(self) -> None:
        self.update_kdtree()
        self.density_field.update(self.state)

        biomass = np.sum(self.state['area'][:self.num_plants])
        population = self.num_plants
        precipitation = self.precipitation(0)
        data = np.array([biomass, population, precipitation])

        self.data_buffer.add(data=data, t=0)
        self.state_buffer.add(state=self.get_state(), t=0)
        self.density_field_buffer.add(
            field=self.density_field.get_values(), t=0)

    def initiate_from_state(self, state: List[Plant]) -> None:
        self.state = np.array([(p.pos, p.r, p.area, p.is_dead)
                              for p in state], dtype=self.state.dtype)
        self.num_plants = len(state)
        self.initiate()

    def initiate_uniform_lifetimes(self, n: int, t_min: float, t_max: float, growth_rate: float, **kwargs: Any) -> None:
        plants = [Plant(pos=np.random.uniform(-self.half_width, self.half_width, 2),
                        r=np.random.uniform(t_min, t_max) * growth_rate, id=i, **kwargs) for i in range(n)]
        self.add(plants)
        self.initiate()

    def initiate_uniform_radii(self, n: int, r_min: float, r_max: float) -> None:
        kwargs = self.kwargs
        r_min = r_min * self._m
        r_max = r_max * self._m

        new_plants = np.zeros(n, dtype=[
                              ('pos', float, 2), ('r', float), ('area', float), ('is_dead', bool)])
        new_plants['pos'] = np.random.uniform(
            -self.half_width, self.half_width, (n, 2))
        new_plants['r'] = np.random.uniform(r_min, r_max, n)
        new_plants['area'] = np.pi * new_plants['r']**2
        new_plants['is_dead'] = False

        self.add(new_plants)
        self.initiate()

    def initiate_dense_distribution(self, n: int, **kwargs: Any) -> None:
        _m = self.kwargs['_m']
        mean_dbhs = np.array([0.05, 0.2, 0.4, 0.6, 0.85, 1.50])
        mean_rs = dbh_to_crown_radius(mean_dbhs) * _m
        freqs = np.array([5800, 378.8, 50.98, 13.42, 5.62, 0.73])
        log_mean_rs = np.log(mean_rs)
        kde = gaussian_kde(log_mean_rs, weights=freqs, bw_method='silverman')
        log_samples = kde.resample(n)[0]
        samples = np.exp(log_samples)
        plants = [Plant(pos=np.random.uniform(-self.half_width, self.half_width,
                        2), r=r, id=i, **kwargs) for i, r in enumerate(samples)]
        self.add(plants)
        self.initiate()

    def initiate_packed_distribution(self, r: float, **kwargs):
        plants = []
        dx = 2 * r
        dy = np.sqrt(3) * r
        for i, x in enumerate(np.arange(-self.half_width + r, self.half_width - r, dx)):
            for j, y in enumerate(np.arange(-self.half_height + r, self.half_height - r, dy)):
                if j % 2 == 0:
                    x += r
                if j % 2 == 1:
                    x -= r
                pos = np.array([x, y])
                plants.append(Plant(pos=pos, r=r, **kwargs))
        self.add(plants)
        self.initiate()

    def plot_state(self, state: List[Plant], title: Optional = None, t: Optional[int] = None, size: int = 2, fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None, highlight: Optional[List[int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a simulation state.

        Parameters:
        -----------
        state : List[Plant]
            The state of the simulation to plot. This parameter is required.
        t : Optional[int]
            The time step to display in the plot title. Defaults to None.
        size : int
            The size of the plot. Defaults to 2.
        fig : Optional[plt.Figure]
            The figure to plot on. If None, a new figure is created. Defaults to None.
        ax : Optional[plt.Axes]
            The axes to plot on. If None, new axes are created. Defaults to None.
        highlight : Optional[List[int]]
            A list of plant IDs to highlight in the plot. Defaults to None.

        Returns:
        --------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        ax.set_xlim(-self.half_width, self.half_width)
        ax.set_ylim(-self.half_height, self.half_height)
        ax.set_aspect('equal', 'box')
        if t is not None:
            ax.set_title(f'{t=}', fontsize=7)
        else:
            ax.set_title('')
        alive_indices = np.where(~state['is_dead'])[0]
        for i in alive_indices:
            circle = plt.Circle(
                state['pos'][i], state['r'][i], fill=True, color='green')
            ax.add_artist(circle)

        # _m = self.kwargs['_m']
        # x_ticks = ax.get_xticks() * _m
        # y_ticks = ax.get_yticks() * _m
        # ax.set_xticklabels([f'{x:.1f}' for x in x_ticks])
        # ax.set_yticklabels([f'{y:.1f}' for y in y_ticks])
        ax.set_xticks([])
        ax.set_yticks([])
        if title is not None:
            ax.set_title(title, fontsize=7)
        return fig, ax

    def plot(self, size: int = 2, title: Optional = None, t: Optional = None, highlight: Optional[List[int]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the current state of the simulation.

        Parameters:
        -----------
        size : int
            The size of the plot. Defaults to 2.
        t : int
            The time step to display in the plot title. Defaults to self.t.
        highlight : Optional[List[int]]
            A list of plant IDs to highlight in the plot. Defaults to None.

        Returns:
        --------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        if t is None:
            t = self.t
        fig, ax = self.plot_state(
            self.get_state(), title=title, t=t, size=size, highlight=highlight)
        return fig, ax

    def plot_states(self, states: List[List[Plant]], times: Optional[List[int]] = None, size: int = 2) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot multiple states of the simulation.

        Parameters:
        -----------
        states : List[List[Plant]]
            A list of states to plot.
        times : Optional[List[int]]
            A list of time steps corresponding to the states. Defaults to None.
        size : int
            The size of the plot. Defaults to 2.

        Returns:
        --------
        Tuple[plt.Figure, plt.Axes]
            The figure and axes of the plot.
        """
        l = len(states)
        n_rows = int(np.floor(l / np.sqrt(l)))
        n_cols = (l + 1) // n_rows + (l % n_rows > 0)
        print(f'simulation.plot_states(): {n_rows=}, {n_cols=}')

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(
            size * n_cols, size * n_rows))
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
                            top=0.95, wspace=0.05, hspace=0.05)
        fig.tight_layout()
        if len(states) == 1:
            self.plot_state(states[0], t=times[0], size=size, ax=ax)
        else:
            i = 0
            while i < len(states):
                state = states[i]
                if n_rows == 1:
                    k = i
                    self.plot_state(
                        state=state, t=times[i], size=size, fig=fig, ax=ax[k])
                else:
                    l = i // n_cols
                    k = i % n_cols
                    self.plot_state(
                        state=state, t=times[i], size=size, fig=fig, ax=ax[l, k])
                i += 1
        if len(states) < n_rows * n_cols:
            for j in range(len(states), n_rows * n_cols):
                if n_rows == 1:
                    ax[j].axis('off')
                else:
                    l = j // n_cols
                    k = j % n_cols
                    ax[l, k].axis('off')

        return fig, ax
