import copy
import json
import time
import warnings
import os
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

from mods.buffers import (DataBuffer, FieldBuffer, HistogramBuffer, StateBuffer,
                          rewrite_density_field_buffer_data,
                          rewrite_hist_buffer_data, rewrite_state_buffer_data)
from mods.fields import DensityFieldSPH as DensityField
from mods.plant import Plant
from mods.utilities import print_nested_dict, save_kwargs, convert_to_serializable, linear_regression


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


def save_simulation_results(
    sim,
    save_folder,
    surfix='test',
    kwargs=False,
    data_buffer=False,
    size_buffer=False,
    biomass_buffer=False,
    state_buffer=False,
    density_field_buffer=False,
    save_all_buffers=True
):
    if save_all_buffers:
        biomass_buffer = True
        data_buffer = True
        density_field_buffer = True
        size_buffer = True
        state_buffer = True
        kwargs = True
        
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    if kwargs:
        sim.save_dict(path=f'{save_folder}/kwargs_{surfix}')
    if data_buffer:
        sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
    if size_buffer:
        sim.size_buffer.save(f'{save_folder}/size_buffer_{surfix}')
    if biomass_buffer:
        sim.biomass_buffer.save(f'{save_folder}/biomass_buffer_{surfix}')
    if state_buffer:
        sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
    if density_field_buffer:
        sim.density_field_buffer.save(
            f'{save_folder}/density_field_buffer_{surfix}')
    print('Data saved in folder:', save_folder)


def plot_simulation_results(
    sim,
    title=None,
    convergence=True,
    state_buffer=False,
    density_field_buffer=False,
    data_buffer=False,
    biomass_buffer=False,
    size_buffer=False,
    kwargs=False,
    plot_all_buffers=True
):

    if plot_all_buffers:
        biomass_buffer = True
        data_buffer = True
        density_field_buffer = True
        size_buffer = True
        state_buffer = True
        kwargs = True

    print('Plotting...')
    L = sim.L
    N0 = len(sim.state_buffer.states[0])
    if title is None:
        title = f'{L =}, $N_0$ = {N0}'
    else:
        title = title + '  ' + f'{L =}, $N_0$ = {N0}'
    if kwargs:
        sim.print_dict()
    if state_buffer:
        sim.state_buffer.plot(title=f'{title}')
    if density_field_buffer:
        sim.density_field_buffer.plot(title=f'{title}')
    if data_buffer:
        fig, ax = sim.data_buffer.plot(title=f'{title}')
        if convergence & (sim.data_buffer.size > 1):
            is_converged, convergence_factor, regression_line = sim.convergence_check()
            time = sim.data_buffer.get_data(
                keys=['Time'])[-len(regression_line):]
            slope = (regression_line[-1] -
                     regression_line[0]) / (time[-1] - time[0])
            slope = slope[0]
            color = 'g' if is_converged else 'r'
            label = 'Converged' if is_converged else 'Not converged'
            label += f' ({convergence_factor:.3e})'
            ax[0].plot(time, regression_line, label=label,
                       color=color, linestyle='--')
            ax[0].legend()

    if biomass_buffer:
        sim.biomass_buffer.plot(title=f'Biomass')
    if size_buffer:
        sim.size_buffer.plot(title=f'Sizes')
    plt.show()


def load_sim_data(
    load_folder,
    surfix,
    state_buffer=False,
    density_field_buffer=False,
    data_buffer=False,
    biomass_buffer=False,
    size_buffer=False,
    kwargs=False,
    load_all_buffers=True
):
    biomass_buffer_df = pd.DataFrame([])
    data_buffer_df = pd.DataFrame([])
    density_field_buffer_df = pd.DataFrame([])
    size_buffer_df = pd.DataFrame([])
    state_buffer_df = pd.DataFrame([])
    kwargs_dict = {}

    if load_all_buffers:
        biomass_buffer = True
        data_buffer = True
        density_field_buffer = True
        size_buffer = True
        state_buffer = True
        kwargs = True

    if biomass_buffer:
        biomass_buffer_df = pd.read_csv(
            f'{load_folder}/biomass_buffer_{surfix}.csv', header=None, low_memory=False, comment='#')
        if 't' not in biomass_buffer_df.iloc[0, :].values:
            biomass_buffer_df, note = rewrite_hist_buffer_data(
                biomass_buffer_df)
            save_permission = input(f"Biomass buffer data at '{load_folder}' with surfix '{
                                    surfix}' needs rewriting. Do you want to rewrite and save it? (Y/n): ")
            if save_permission.lower() == 'y':
                biomass_buffer_df.to_csv(
                    f'{load_folder}/biomass_buffer_{surfix}.csv', index=False, header=True)
                with open(f'{load_folder}/biomass_buffer_{surfix}.csv', 'w') as file:
                    file.write(f"# {note.replace('\n', '\n# ')}\n")
                    biomass_buffer_df.to_csv(file, index=False)

        if biomass_buffer_df.iloc[0, 0] == 't':
            biomass_buffer_df = pd.read_csv(
                f'{load_folder}/biomass_buffer_{surfix}.csv', header=0, comment='#')

    if data_buffer:
        data_buffer_df = pd.read_csv(
            f'{load_folder}/data_buffer_{surfix}.csv', header=0, comment='#')

    if density_field_buffer:
        density_field_buffer_df = pd.read_csv(
            f'{load_folder}/density_field_buffer_{surfix}.csv', header=None, low_memory=False, comment='#')
        if 't' not in density_field_buffer_df.iloc[0, :].values:
            density_field_buffer_df = rewrite_density_field_buffer_data(
                density_field_buffer_df)
            save_permission = input(f"Density field buffer data at '{load_folder}' with surfix '{
                                    surfix}' is missing 'bins' keys. Do you want to rewrite and save it? (Y/n): ")
            if save_permission.lower() == 'y':
                density_field_buffer_df.to_csv(
                    f'{load_folder}/density_field_buffer_{surfix}.csv', index=False, header=True)
        if density_field_buffer_df.iloc[0, 0] == 't':
            density_field_buffer_df = pd.read_csv(
                f'{load_folder}/density_field_buffer_{surfix}.csv', header=0, comment='#')

    if kwargs:
        with open(f'{load_folder}/kwargs_{surfix}.json', 'r') as file:
            kwargs_dict = json.load(file)

    if size_buffer:
        size_buffer_df = pd.read_csv(
            f'{load_folder}/size_buffer_{surfix}.csv', header=None, low_memory=False, comment='#')
        if 't' not in size_buffer_df.iloc[0, :].values:
            size_buffer_df, note = rewrite_hist_buffer_data(size_buffer_df)
            save_permission = input(f"Size buffer data at '{load_folder}' with surfix '{
                                    surfix}' is missing 'bins' keys. Do you want to rewrite and save it? (Y/n): ")
            if save_permission.lower() == 'y':
                size_buffer_df.to_csv(
                    f'{load_folder}/size_buffer_{surfix}.csv', index=False, header=True)
                with open(f'{load_folder}/size_buffer_{surfix}.csv', 'w') as file:
                    file.write(f"# {note.replace('\n', '\n# ')}\n")
                    size_buffer_df.to_csv(file, index=False)
        if size_buffer_df.iloc[0, 0] == 't':
            size_buffer_df = pd.read_csv(
                f'{load_folder}/size_buffer_{surfix}.csv', header=0, comment='#')

    if state_buffer:
        state_buffer_df = pd.read_csv(
            f'{load_folder}/state_buffer_{surfix}.csv', header=None, low_memory=False, comment='#')
        if 'id' != state_buffer_df.iloc[0, 0]:
            state_buffer_df = rewrite_state_buffer_data(state_buffer_df)
            save_permission = input(f"State buffer data at '{load_folder}' with surfix '{
                                    surfix}' is missing 'id'. Do you want to rewrite and save it? (Y/n): ")
            if save_permission.lower() == 'y':
                state_buffer_df.to_csv(
                    f'{load_folder}/state_buffer_{surfix}.csv', index=False, header=True)
        if state_buffer_df.loc[0, 0] == 'id':
            state_buffer_df = pd.read_csv(
                f'{load_folder}/state_buffer_{surfix}.csv', header=0, comment='#')

    return state_buffer_df, density_field_buffer_df, data_buffer_df, biomass_buffer_df, size_buffer_df, kwargs_dict


def sim_from_data(sim_data, times_to_load='last'):
    state_buffer_df, density_field_buffer_df, data_buffer_df, biomass_buffer_df, size_buffer_df, kwargs = sim_data
    times = state_buffer_df['t'].unique()
    if times_to_load == 'last':
        times_to_load = [times[-1]]
    elif times_to_load == 'all':
        times_to_load = times
    elif isinstance(times_to_load, (int, list, np.ndarray)):
        times_to_load = [t for t in times if t in times_to_load]
    elif isinstance(times_to_load, slice):
        start = times_to_load.start if times_to_load.start is not None else 0
        stop = times_to_load.stop if times_to_load.stop is not None else len(times)
        step = times_to_load.step if times_to_load.step is not None else 1
        times_to_load = times[start:stop:step]
    else:
        raise ValueError(
            "times_to_load must be 'last', 'all', an integer, or a list of values")
    sim = Simulation(**kwargs)
    
    if state_buffer_df is not None:
        state_buffer_df = state_buffer_df[state_buffer_df['t'].isin(times_to_load)]
        sim.state_buffer = StateBuffer(data=state_buffer_df, **kwargs)
        sim.state = sim.state_buffer.states[-1]
        sim.t = sim.state_buffer.times[-1]

    if density_field_buffer_df is not None:
        density_field_buffer_df = density_field_buffer_df[density_field_buffer_df['t'].isin(
            times_to_load)]
        sim.density_field_buffer = FieldBuffer(data=density_field_buffer_df)
        sim.density_field = DensityField(half_height=sim.half_height, half_width=sim.half_width,
                                         density_radius=sim.density_check_radius, resolution=sim.density_field_resolution)
        
    if data_buffer_df is not None:
        data_buffer_df = data_buffer_df[data_buffer_df['Time'].isin(times_to_load)]
        sim.data_buffer = DataBuffer(data=data_buffer_df)
        
    if biomass_buffer_df is not None:
        biomass_buffer_df = biomass_buffer_df[biomass_buffer_df['t'].isin(
            times_to_load)]
        sim.biomass_buffer = HistogramBuffer(
            data=biomass_buffer_df, start=0, end=sim.r_max**2 * np.pi, title='Biomass')
        
    if size_buffer_df is not None:
        size_buffer_df = size_buffer_df[size_buffer_df['t'].isin(times_to_load)]
        sim.size_buffer = HistogramBuffer(
            data=size_buffer_df, start=sim.r_min, end=sim.r_max, title='Sizes')

    sim.initiate()
    print(f'Simulation loaded at t = {sim.t}')
    return sim


path_kwargs = 'default_kwargs.json'
with open(path_kwargs, 'r') as file:
    default_kwargs = json.load(file)


class IDGenerator:
    def __init__(self, start=0):
        self.current_id = start

    def get_next_id(self):
        self.current_id += 1
        return self.current_id


class Simulation:
    def __init__(self, **kwargs):
        self.__dict__.update(default_kwargs)
        self.__dict__.update(kwargs)
        self.spinning_up = False

        self.t = 0
        self.state = []
        self.land_quality = 0.001

        self.half_width = 0.5
        self.half_height = 0.5
        
        self._m = 1 / self.L
        self.r_min = self.r_min * self._m
        self.r_max = self.r_max * self._m
        self.dispersal_range = self.dispersal_range * self._m
        self.spawn_rate = self.spawn_rate * self.time_step
        self.growth_rate = self.growth_rate * self._m * self.time_step
        self.density_check_radius = self.density_check_radius * self._m
        self.kt = None

        self.id_generator = IDGenerator()

        self.density_field = DensityField(
            half_width=self.half_width,
            half_height=self.half_height,
            density_radius=self.density_check_radius,
            resolution=self.density_field_resolution,
        )

        if 'buffer_preset_times' in self.__dict__:
            self.buffer_preset_times = self.buffer_preset_times
        else:
            self.buffer_preset_times = np.arange(
                self.buffer_size) * self.buffer_skip

        self.data_buffer = DataBuffer(sim=self,
                                      size=int(
                                          np.floor(self.T / self.time_step) + 1),
                                      keys=['Time', 'Biomass', 'Population']
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

        self.biomass_buffer = HistogramBuffer(
            size=int(np.floor(self.T / self.time_step)) + 1, start=0, end=self.r_max**2 * np.pi, title='Biomass'
        )
        self.size_buffer = HistogramBuffer(
            size=int(np.floor(self.T / self.time_step)) + 1, start=self.r_min, end=self.r_max, title='Sizes'
        )

    def add(self, plant: Union[Plant, List[Plant], np.ndarray]):
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

    def update_kdtree(self, state):
        if len(state) == 0:
            self.kt = None
        else:
            self.kt = KDTree(
                [plant.pos for plant in state])

    def step(self):
        n = self.spawn_rate
        self.attempt_spawn(n=n)

        n0 = len(self.state)
        # First Phase: Update all plants based on the current state of the simulation
        for plant in self.state:
            plant.update(self)

        # Second Phase: Collect non-dead plants and add them to the new state, and make sure all new plants get a unique id
        new_plants = []
        for plant in self.state:
            if not plant.is_dead:
                new_plants.append(plant)

        self.state = new_plants

        prev_t = self.t
        self.t += self.time_step

        # Update necessary data structures
        self.density_field.update(self.state)
        self.update_kdtree(self.state)

        self.collect_data()

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

    def run(self, T, max_population=None, transient_period=None):
        
        start_time = time.time()
        sim_start_time = self.t
        n_iter = int(np.ceil(T / self.time_step))
        self.extend_buffers(n_iter)
        print(f'Simulation.run(): Running simulation for {
              n_iter} iterations...')
        try:
            for _ in range(0, n_iter):
                self.step()

                if transient_period is not None and self.t - sim_start_time < transient_period:
                    is_converged = False
                else:
                    is_converged = self.convergence_check()[0]

                # if the population exceeds the maximum allowed, stop the simulation
                l = len(self.state)
                if (max_population is not None and l > max_population):
                    print(
                        f'Simulation.run(): Population exceeded {max_population}. Stopping simulation...')
                    break
                elif is_converged:
                    print(
                        f'Simulation.run(): Convergence reached at t = {self.t}. Stopping simulation...')
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
                    t, biomass, population = data
                    t = float(round(t, 2))

                    print(f'{dots} Elapsed time: {elapsed_time_str}' + ' '*5 + f'|  {t=:^8}  |  N = {
                          population:<6}  |  B = {np.round(biomass, 4):<6}', end='\r')

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

    def convergence_check(self, trend_window=6000, trend_threshold=1):
        data = self.data_buffer.get_data(keys=['Time', 'Biomass'])
        time, biomass = data[:, 0], data[:, 1]
        time_step = time[1] - time[0]
        window = np.min([int(trend_window//time_step), len(time)])
        x = time[-window:]
        y = biomass[-window:]
        σy = np.std(y)
        μy = np.mean(y)
        if σy == 0:
            _, slope_norm, regression_line_norm, _, _ = linear_regression(
                x, y, advanced=True)
            regression_line = regression_line_norm
        else:
            y_norm = (y - μy) / σy
            _, slope_norm, regression_line_norm, _, _ = linear_regression(
                x, y_norm, advanced=True)
            regression_line = regression_line_norm * σy + μy

        convergence_factor = np.abs(slope_norm) * \
            trend_window - trend_threshold
        is_converged = bool(convergence_factor < 0) & bool(
            self.t > trend_window)
        return is_converged, convergence_factor, regression_line

    # def spin_up(self, target_population: int, max_time: int = 1000, speed: float = 0.01):
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

    def attempt_spawn(self, n: int):

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
            densities * self.precipitation, self.land_quality, 1)

        spawn_indices = np.where(
            probabilities > random_values)[0]

        new_plants = [
            Plant(
                id=self.id_generator.get_next_id(),
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

    def initiate(self):
        """
        Initialize the simulation by updating necessary data structures.

        This method updates the KDTree, density field, and buffers with the initial state of the simulation.
        """
        self.update_kdtree(self.state)
        self.density_field.update(self.state)
        print(f'\nSimulation.initiate(). Time: {time.strftime("%H:%M:%S")}')

    def collect_data(self):
        sizes = np.array([plant.r for plant in self.state])
        biomass = np.array([plant.area for plant in self.state])
        biomass_total = sum(biomass)
        population = len(self.state)
        data = np.array([biomass_total, population])

        self.data_buffer.add(data=data, t=self.t)
        self.biomass_buffer.add(data=biomass, t=self.t)
        self.size_buffer.add(data=sizes, t=self.t)
        self.state_buffer.add(state=self.get_state(), t=self.t)
        self.density_field_buffer.add(
            field=self.density_field.get_values(), t=self.t)

    def initiate_uniform_lifetimes(self, n: int, t_min: float, t_max: float, growth_rate: float):
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
                id=self.id_generator.get_next_id(),
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
        self.collect_data()

    def extend_buffers(self, n):
        self.data_buffer.extend(n)
        self.size_buffer.extend(n)
        self.biomass_buffer.extend(n)
        self.density_field_buffer.extend(n)
        self.state_buffer.extend(n)

    def initiate_uniform_radii(self, n: int, r_min: float, r_max: float):
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
                id=self.id_generator.get_next_id(),
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
        self.collect_data()

    def initiate_dense_distribution(self, n: int):
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
                id=self.id_generator.get_next_id(),
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
        self.collect_data()

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
                        id=self.id_generator.get_next_id(),
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

    exclude_default = ['state',
                       'kt',
                       'state_buffer',
                       'data_buffer',
                       'density_field_buffer',
                       'density_field',
                       'biomass_buffer',
                       'size_buffer',
                       'buffer_preset_times',
                       'verbose',
                       'spinning_up',
                       'half_width',
                       'half_height',
                       'id_generator']

    def save_dict(self, path: str, exclude: Optional[List[str]] = exclude_default):
        """
        Save the keyword arguments of the simulation to a JSON file.

        Parameters:
        -----------
        path : str
            The file path to save the JSON file to.
        exclude : Optional[List[str]]
            A list of keys to exclude from the saved dictionary. Defaults to None.
        """
        self.r_min = self.r_min / self._m
        self.r_max = self.r_max / self._m
        self.dispersal_range = self.dispersal_range / self._m
        self.spawn_rate = self.spawn_rate / self.time_step
        self.growth_rate = self.growth_rate / \
            self._m / self.time_step
        self.density_check_radius = self.density_check_radius / self._m

        save_kwargs(self.__dict__, path, exclude=exclude)

    def print_dict(self, exclude = exclude_default):
        """
        Print the keyword arguments of the simulation.
        """
        conversion_factors = {
            'r_min': self._m,
            'r_max': self._m,
            'dispersal_range': self._m,
            'spawn_rate': self.time_step,
            'growth_rate': self._m * self.time_step,
            'density_check_radius': self._m
        }

        dict_temp = {key: (value / conversion_factors[key] if key in conversion_factors.keys() else value)
                     for key, value in self.__dict__.items()}

        print_nested_dict(dict_temp, exclude=exclude)

    def cleanup(self):
        self.state = []
        self.kt = None
        self.state_buffer = None
        self.data_buffer = None
        self.density_field_buffer = None
        self.density_field = None
        self.biomass_buffer = None
        self.size_buffer = None

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
