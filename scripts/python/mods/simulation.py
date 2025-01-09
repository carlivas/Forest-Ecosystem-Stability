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
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable


def check_pos_collision(pos: np.ndarray, plant: Plant) -> bool:
    return np.sum((pos - plant.pos) ** 2) < plant.r ** 2


def check_collision(p1: Plant, p2: Plant) -> bool:
    return np.sum((p1.pos - p2.pos) ** 2) < (p1.r + p2.r) ** 2


def dbh_to_crown_radius(dbh: float) -> float:
    # everything in m
    d = 1.42 + 28.17*dbh - 11.26*dbh**2
    return d/2


def _m_from_m2pp(m2pp, num_plants, A_bound=1) -> float:
    return np.sqrt(A_bound/(m2pp*num_plants))


def _m_from_domain_sides(L, S_bound=1) -> float:
    return S_bound / L


default_kwargs = {
    'dispersal_range': 90,
    'spawn_rate': 1,
    'growth_rate': 0.1,
    'time_step': 1,
    'density_field_resolution': 100,
    'density_check_radius': 100,
    'buffer_size': 20,
    'buffer_skip': 20,
    'r_min': 0.1,
    'r_max': 30,
    'verbose': True,
}


class Simulation:
    def __init__(self, **kwargs):
        self.__dict__.update(default_kwargs)
        self.__dict__.update(kwargs)


        self.t = 0
        self.state = []
        self.land_quality = 0.001
        self.spawn_rate = self.spawn_rate * self.time_step

        precipitation = self.precipitation
        if isinstance(precipitation, float):
            self.precipitation = lambda t: precipitation
        elif isinstance(precipitation, int):
            self.precipitation = lambda t: precipitation
        elif callable(precipitation):
            self.precipitation = precipitation
        else:
            raise ValueError('Precipitation must be a float, int or callable')

        self.half_width = 0.5
        self.half_height = 0.5

        self._m = 1 / self.L
        self.r_min = self.r_min * self._m
        self.r_max = self.r_max * self._m
        self.dispersal_range = self.dispersal_range * self._m
        self.growth_rate = self.growth_rate * self._m * self.time_step
        self.density_field_check_radius = self.density_check_radius * self._m

        self.kt = None

        if 'buffer_preset_times' in self.__dict__:
            self.buffer_preset_times = self.buffer_preset_times
        else:
            # self.buffer_preset_times = np.linspace(1, kwargs.get('n_iter', 10000), buffer_size).astype(int)
            self.buffer_preset_times = np.arange(
                self.buffer_size) * self.buffer_skip

        self.data_buffer = DataBuffer(sim=self,
                                      size=int(
                                          np.floor(self.T / self.time_step) + 1)
                                      )

        self.state_buffer = StateBuffer(
            size=self.buffer_size,
            skip=self.buffer_skip,
            preset_times=self.buffer_preset_times
        )

        self.density_field_buffer = FieldBuffer(
            sim=self,
            resolution=self.density_field_resolution,
            size=self.buffer_size,
            skip=self.buffer_skip,
            preset_times=self.buffer_preset_times,
        )

        self.density_field = DensityField(
            half_width=self.half_width,
            half_height=self.half_height,
            density_radius=self.density_check_radius,
            resolution=self.density_field_resolution,
            simulation=self
        )

    def add(self, plant: Union[Plant, List[Plant], np.ndarray]) -> None:
        if isinstance(plant, Plant):
            self.state.append(plant)
        elif isinstance(plant, (list, np.ndarray)):
            for p in plant:
                if isinstance(p, Plant):
                    self.state.append(p)
                else:
                    raise ValueError(
                        "All elements in the array must be Plant objects")
        else:
            raise ValueError(
                "Input must be a Plant object or an array_like of Plant objects")

    def update_kdtree(self) -> None:
        if len(self.state) == 0:
            self.kt = None
        else:
            self.kt = KDTree(
                [plant.pos for plant in self.state], leafsize=10)

    def step(self) -> None:
        n = self.spawn_rate
        self.attempt_spawn(n=n)

        n0 = len(self.state)
        # First Phase: Update all plants based on the current state of the simulation
        for plant in self.state:
            plant.update(self)

        n1 = len(self.state)
        n_spawned = n1 - n0

        # Second Phase: Collect non-dead plants and add them to the new state, and make sure all new plants get a unique id
        new_plants = []
        for plant in self.state:
            if not plant.is_dead:
                new_plants.append(plant)

        self.state = new_plants
        n2 = len(self.state)
        n_dead = n1 - n2
        # print(f'step(): {n_spawned=} {n_dead=}', end='\n')

        prev_t = self.t
        self.t += self.time_step

        # Update necessary data structures
        self.density_field.update()

        if n_spawned > 0 or n_dead > 0:
            self.update_kdtree()

        population = len(self.state)
        biomass = sum([plant.area for plant in self.state])
        precipitation = self.precipitation(self.t)
        data = np.array([biomass, population, precipitation])
        self.data_buffer.add(data=data, t=self.t)

        prev_mod_state = prev_t % self.state_buffer.skip
        mod_state = self.t % self.state_buffer.skip
        do_save_state = prev_mod_state >= mod_state
        prev_mod_density_field = prev_t % self.density_field_buffer.skip
        mod_density_field = self.t % self.density_field_buffer.skip
        do_save_density_field = prev_mod_density_field >= mod_density_field

        if do_save_state:
            self.state_buffer.add(state=self.get_state(), t=self.t)
        if do_save_density_field:
            self.density_field_buffer.add(
                field=self.density_field.get_values(), t=self.t)

    def run(self, T: float, max_population: Optional[int] = 15_000) -> None:
        """
        Run the simulation for a given number of iterations.

        Parameters:
        -----------
        n_iter : Optional[int]
            The number of iterations to run the simulation. If None, the value from kwargs is used.
        """

        start_time = time.time()
        n_iter = int(T / self.time_step)
        try:
            for _ in range(1, n_iter):
                self.step()

                # if the population exceeds the maximum allowed, stop the simulation
                l = len(self.state)
                if (max_population is not None and l > max_population):
                    break

                if self.verbose:
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

                    data = self.data_buffer.get_data(data_idx=-1)
                    t, biomass, population, precipitation = data
                    t = float(round(t, 2))

                    print(f'{dots} Elapsed time: {elapsed_time_str}' + ' '*5 + f'|  {t=:^8}  |  N = {
                          population:<6}  |  B = {np.round(biomass, 4):<6}  |  P = {np.round(precipitation, 4):<6}', end='\r')

        except KeyboardInterrupt:
            print('\nInterrupted by user...')

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time_str = f"{str(int(hours))}".rjust(
            2, '0') + ":" + f"{str(int(minutes))}".rjust(2, '0') + ":" + f"{str(int(seconds))}".rjust(2, '0')
        print()
        print(f'Simulation.run(): Done. Elapsed time: {elapsed_time_str}')
        print()

    # def spin_up(self, target_population: int, max_time: int = 1000, speed: float = 0.01) -> None:
    #     """
    #     Run the simulation until the population reaches a given maximum value.

    #     Parameters:
    #     -----------
    #     target_population : int
    #         The maximum population size to run the simulation until.
    #     """
    #     self.spinning_up = True
    #     start_time = time.time()
    #     land_quality = self.land_quality
    #     spawn_rate = self.spawn_rate
    #     if self.verbose:
    #         print(f'Simulation.spin_up(): Simulating until population reaches {
    #             target_population}...')
    #     self.land_quality = 1
    #     print(f'Simulation.spin_up(): !Warning! setting land_quality to {
    #           self.land_quality}')
    #     try:
    #         while len(self.state) < target_population and self.t < max_time:

    #             drive = target_population*1.1 - len(self.state)
    #             speed_temp = self.t / max_time
    #             spin_up_rate = int(drive * speed_temp)
    #             self.spawn_rate = max(
    #                 1, spin_up_rate)  # * self.dt
    #             print()
    #             print(f'Simulation.spin_up(): {drive=} {speed_temp=}, {
    #                   spin_up_rate=},{self.spawn_rate=} ')
    #             self.step()

    #             if self.verbose:
    #                 elapsed_time = time.time() - start_time
    #                 hours, rem = divmod(elapsed_time, 3600)
    #                 minutes, seconds = divmod(rem, 60)
    #                 elapsed_time_str = f"{str(int(hours))}".rjust(
    #                     2, '0') + ":" + f"{str(int(minutes))}".rjust(2, '0') + ":" + f"{str(int(seconds))}".rjust(2, '0')

    #                 print(f'Elapsed time: {elapsed_time_str}', end='\r')

    #     except KeyboardInterrupt:
    #         print('\nInterrupted by user...')

    #     self.land_quality = land_quality
    #     self.spawn_rate = spawn_rate
    #     self.spinning_up = False
    #     print(f'Simulation.spin_up(): Done. Setting land_quality back to {
    #           land_quality} and spawn_rate back to {spawn_rate}')

    def attempt_spawn(self, n: int) -> None:

        # Take care of the decimal part of n
        decimal_part = n - int(n)
        n = int(n)
        if np.random.rand() < decimal_part:
            n += 1

        new_positions = np.random.uniform(-self.half_width,
                                          self.half_width, (n, 2))

        densities = self.density_field.query(new_positions)

        random_values = np.random.uniform(0, 1, n)

        probabilities = np.clip(
            densities * self.precipitation(self.t), self.land_quality, 1)
        # print(f'Simulation.attempt_spawn(): {np.mean(probabilities)=}')

        spawn_indices = np.where(
            probabilities > random_values)[0]

        new_plants = [
            Plant(
                pos=new_positions[i],
                r=self.r_min,
                r_min=self.r_min,
                r_max=self.r_max,
                growth_rate=self.growth_rate,
                dispersal_range=self.dispersal_range
            )
            for i in spawn_indices
        ]

        self.add(new_plants)

    def get_collisions(self, plant: Plant) -> List[Plant]:
        """
        Get a list of plants that collide with the given plant.

        Parameters:
        -----------
        plant : Plant
            The plant to check for collisions.

        Returns:
        --------
        List[Plant]
            A list of plants that collide with the given plant.
        """
        plant.is_colliding = False
        collisions = []
        if self.kt is not None:
            indices = self.kt.query_ball_point(
                x=plant.pos, r=plant.d, workers=-1)
            for i in indices:
                other_plant = self.state[i]
                if other_plant != plant:
                    if check_collision(plant, other_plant):
                        plant.is_colliding = True
                        other_plant.is_colliding = True
                        collisions.append(other_plant)
        return collisions

    def pos_in_box(self, pos: np.ndarray) -> bool:
        pos_in_box = np.abs(pos[0]) < self.half_width and np.abs(
            pos[1]) < self.half_height
        return pos_in_box

    def local_density(self, pos: np.ndarray) -> float:
        """
        Get the quality of the land near a given position.

        Parameters:
        -----------
        pos : np.ndarray
            The position to check.

        Returns:
        --------
        float
            The quality of the land near the given position.
        """
        if self.pos_in_box(pos):
            return self.density_field.query(pos)
        else:
            return 0

    def get_state(self):
        return copy.deepcopy(self.state)

    def initiate(self) -> None:
        """
        Initialize the simulation by updating necessary data structures.

        This method updates the KDTree, density field, and buffers with the initial state of the simulation.
        """
        self.update_kdtree()
        self.density_field.update()

        biomass = sum([plant.area for plant in self.state])
        population = len(self.state)
        precipitation = self.precipitation(0)
        data = np.array([biomass, population, precipitation])

        self.data_buffer.add(data=data, t=0)
        self.state_buffer.add(state=self.get_state(), t=0)
        self.density_field_buffer.add(
            field=self.density_field.get_values(), t=0)

    def initiate_uniform_lifetimes(self, n: int, t_min: float, t_max: float, growth_rate: float) -> None:
        """
        Initialize the simulation with plants having uniform lifetimes. This is only valid for the case where the growth rate is constant.

        Parameters:
        -----------
        n : int
            The number of plants to initialize.
        t_min : float
            The minimum lifetime of the plants.
        t_max : float
            The maximum lifetime of the plants.
        """
        r_min = t_min * growth_rate
        r_max = t_max * growth_rate
        plants = [
            Plant(
                pos=np.random.uniform(-self.half_width, self.half_width, 2),
                r=np.random.uniform(r_min, r_max),
                r_min=r_min,
                r_max=r_max,
                growth_rate=growth_rate,
                dispersal_range=self.dispersal_range
            )
            for i in range(n)
        ]
        self.add(plants)
        self.initiate()

    def initiate_uniform_radii(self, n: int, r_min: float, r_max: float) -> None:
        """
        Initialize the simulation with plants having uniform radii. This is only valid for the case where the growth rate is constant.

        Parameters:
        -----------
        n : int
            The number of plants to initialize.
        r_min : float
            The minimum radius of the plants.
        r_max : float
            The maximum radius of the plants.
        """

        r_min = r_min * self._m
        r_max = r_max * self._m

        new_plants = [
            Plant(
                pos=np.random.uniform(-self.half_width, self.half_width, 2),
                r=np.random.uniform(r_min, r_max),
                r_min=r_min,
                r_max=r_max,
                growth_rate=self.growth_rate,
                dispersal_range=self.dispersal_range
            )
            for i in range(n)
        ]
        self.add(new_plants)
        self.initiate()

    def initiate_dense_distribution(self, n: int) -> None:
        """
        Initialize the simulation with a dense distribution of plants. The distribution is based on empirical data taken from Cummings et al. (2002).

        Parameters:
        -----------
        n : int
            The number of plants to initialize.
        """
        mean_dbhs = np.array([0.05, 0.2, 0.4, 0.6, 0.85, 1.50])  # in m
        mean_rs = dbh_to_crown_radius(mean_dbhs) * self._m
        freqs = np.array([5800, 378.8, 50.98, 13.42, 5.62, 0.73])

        # Apply log transformation to the data
        log_mean_rs = np.log(mean_rs)

        # Calculate the KDE of the log-transformed data
        kde = gaussian_kde(log_mean_rs, weights=freqs, bw_method='silverman')

        # Resample from the KDE and transform back to the original space
        log_samples = kde.resample(n)[0]
        samples = np.exp(log_samples)

        # Create the plants
        plants = [
            Plant(
                pos=np.random.uniform(-self.half_width, self.half_width, 2),
                r=r,
                r_min=self.r_min,
                r_max=self.r_max,
                growth_rate=self.growth_rate,
                dispersal_range=self.dispersal_range
            )
            for i, r in enumerate(samples)
        ]
        self.add(plants)
        self.initiate()

    def initiate_packed_distribution(self, r: float):
        """
        Initialize the simulation with a packed hexagonal grid of plants.

        Parameters:
        -----------
        r : float
            The radius of each plant.
        """
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
                plants.append(
                    Plant(
                        pos=pos,
                        r=r,
                        r_min=self.r_min,
                        r_max=self.r_max,
                        growth_rate=self.growth_rate,
                        dispersal_range=self.dispersal_range
                    )
                )

        self.add(plants)
        self.initiate()

    def save_dict(self, path: str, exclude: Optional[List[str]] = ['t', 'state', 'kt', 'state_buffer', 'data_buffer', 'density_field_buffer', 'density_field', 'buffer_preset_times', 'verbose', 'spinning_up', 'half_width', 'half_height', 'precipitation']) -> None:
        """
        Save the keyword arguments of the simulation to a JSON file.

        Parameters:
        -----------
        path : str
            The file path to save the JSON file to.
        exclude : Optional[List[str]]
            A list of keys to exclude from the saved dictionary. Defaults to None.
        """
        save_kwargs(self.__dict__, path, exclude=exclude)

    def print_dict(self, exclude: Optional[List[str]] = ['t', 'state', 'kt', 'state_buffer', 'data_buffer', 'density_field_buffer', 'density_field', 'buffer_preset_times', 'verbose', 'spinning_up', 'half_width', 'half_height', 'precipitation']) -> None:
        """
        Print the keyword arguments of the simulation.
        """
        print_nested_dict(self.__dict__, exclude=exclude)

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
            t = round(t, 2)
            ax.set_title(f'{t=}', fontsize=7)
        else:
            ax.set_title('')
        for plant in state:
            color = 'green'
            density = self.density_field.query(plant.pos)
            ax.add_artist(plt.Circle(plant.pos, plant.r,
                          color=color, fill=True, transform=ax.transData))

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
