import copy
import json
import time
import warnings
import os
from typing import *
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde

from mods.buffers import (DataBuffer, FieldBuffer, HistogramBuffer, StateBuffer,
                          rewrite_density_field_buffer_data,
                          rewrite_hist_buffer_data, rewrite_state_buffer_data)
from mods.fields import DensityFieldSPH, DensityFieldCustom
from mods.plant import Plant
from mods.utilities import print_nested_dict, save_kwargs, convert_to_serializable, linear_regression, dbh_to_crown_radius


modi_path_kwargs = '../../default_kwargs.json'
local_path_kwargs = 'default_kwargs.json'
path_kwargs = modi_path_kwargs if os.path.exists(modi_path_kwargs) else local_path_kwargs

with open(path_kwargs, 'r') as file:
    default_kwargs = json.load(file)


class IDGenerator:
    def __init__(self, start=0):
        self.current_id = start

    def get_next_id(self):
        self.current_id += 1
        return self.current_id


class Simulation:
    def __init__(self, folder, alias=None, override=False, **kwargs):
        kwargs_path = f'{folder}/kwargs_{alias}.json'
        data_buffer_path = f'{folder}/data_buffer_{alias}.csv'
        state_buffer_path = f'{folder}/state_buffer_{alias}.csv'
        density_field_buffer_path = f'{folder}/density_field_buffer_{alias}.csv'
        figure_folder = f'{folder}/figures'

        if override and os.path.exists(kwargs_path):
            do_override = input(
                f'Simulation.__init__(): OVERRIDE existing files in folder {folder} with alias {alias}? (Y/n):')
            if do_override.lower() != 'y':
                raise ValueError('Simulation.__init__(): Aborted by user...')
            else:
                for path in [kwargs_path, data_buffer_path, state_buffer_path, density_field_buffer_path]:
                    if os.path.exists(path):
                        os.remove(path)
                if os.path.exists(figure_folder):
                    for file in os.listdir(figure_folder):
                        if alias in file:
                            os.remove(os.path.join(figure_folder, file))

        os.makedirs(folder + '/figures', exist_ok=True)
        if os.path.exists(kwargs_path):
            print(f'Simulation.__init__(): Loading kwargs from {kwargs_path}')
            with open(kwargs_path, 'r') as file:
                kwargs = json.load(file)

        self.__dict__.update(default_kwargs)
        self.__dict__.update(kwargs)
        self.folder = folder

        if alias is None:
            alias = 'test'
        self.alias = alias

        self.t = 0
        self.plants = []
        self.land_quality = 0.001

        self.half_width = 0.5
        self.half_height = 0.5

        self._m = 1 / self.L
        self.r_min = self.r_min * self._m
        self.r_max = self.r_max * self._m
        self.maturity_size = self.maturity_size * self._m
        self.dispersal_range = self.dispersal_range * self._m
        self.spawn_rate = self.spawn_rate * self.time_step
        self.growth_rate = self.growth_rate * self._m * self.time_step
        self.density_check_radius = self.density_check_radius * self._m
        self.kt = None

        self.id_generator = IDGenerator()

        self.save_dict(path=kwargs_path)
        self.data_buffer = DataBuffer(file_path=data_buffer_path)
        self.state_buffer = StateBuffer(file_path=state_buffer_path)
        self.density_field_buffer = FieldBuffer(file_path=density_field_buffer_path, resolution=self.density_field_resolution)
        # self.density_field = DensityFieldSPH(
        #     half_width=self.half_width,
        #     half_height=self.half_height,
        #     density_radius=self.density_check_radius,
        #     resolution=self.density_field_resolution,
        # )
        self.density_field = DensityFieldCustom(
            half_width=self.half_width,
            half_height=self.half_height,
            bandwidth_factor=self.density_bandwidth_factor,
            query_radius_factor=self.density_query_radius_factor,
            resolution=self.density_field_resolution,
        )

        last_state_df = self.state_buffer.get_last_state()
        if not last_state_df.empty:
            self.plants = [Plant(id=id, x=x, y=y, r=r, **self.__dict__) for (id, x, y, r)
                           in last_state_df[['id', 'x', 'y', 'r']].values]
            self.t = last_state_df['t'].values[-1]
        self.initiate()

    def initiate(self):
        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)
        print(f'\nSimulation.initiate(): Time: {time.strftime("%Y-%m-%d %H:%M:%S")}')

    def add(self, plant: Union[Plant, List[Plant], np.ndarray]):
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

    def update_kdtree(self, state):
        if len(state) == 0:
            self.kt = None
        else:
            self.kt = KDTree(
                [(plant.x, plant.y) for plant in state])

    def step(self):
        n = self.spawn_rate
        self.attempt_spawn(n=n)

        n0 = len(self.plants)
        # First Phase: Update all plants based on the current state of the simulation
        for plant in self.plants:
            plant.update(self)
        
        # Check for and resolve collisions
        self.resolve_collisions(self.get_collisions())

        # Second Phase: Collect non-dead plants and add them to the new state, and make sure all new plants get a unique id
        new_plants = []
        for plant in self.plants:
            if not plant.is_dead:
                new_plants.append(plant)

        self.plants = new_plants

        prev_t = self.t
        self.t += self.time_step

        # Update necessary data structures
        self.density_field.update(self.plants)
        self.update_kdtree(self.plants)

        self.data_buffer.add(self.collect_data())

        prev_mod_state = prev_t % self.state_buffer.skip
        mod_state = self.t % self.state_buffer.skip
        do_save_state = prev_mod_state >= mod_state
        prev_mod_density_field = prev_t % self.density_field_buffer.skip
        mod_density_field = self.t % self.density_field_buffer.skip
        do_save_density_field = prev_mod_density_field >= mod_density_field

        if do_save_state:
            self.state_buffer.add(plants=self.plants, t=self.t)
        if do_save_density_field:
            self.density_field_buffer.add(
                field=self.density_field.values, t=self.t)

    def run(self, T, max_population=None, transient_period=2, delta_p=0, convergence_stop=False):

        start_time = time.time()
        n_iter = int(np.ceil(T / self.time_step))
        print(f'Simulation.run(): Running simulation for {n_iter} iterations...')
        try:
            for _ in range(0, n_iter):
                if _ > transient_period:
                    self.precipitation = max(0, self.precipitation + delta_p)
                self.step()
                is_converged, convergence_factor = self.convergence_check()[:2]
                
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

                    data = self.collect_data()
                    t, biomass, population, precipitation = data[[
                        'Time', 'Biomass', 'Population', 'Precipitation']].values.reshape(-1)
                    t = float(round(t, 2))

                    print(f'{dots} Elapsed time: {elapsed_time_str}' + ' '*5 + f'|  {t=:^8}  |  N = {
                          population:<6}  |  B = {np.round(biomass, 4):<6}  |  conv = {np.round(convergence_factor, 8):<10}', end='\r')
                # if the population exceeds the maximum allowed, stop the simulation
                l = len(self.plants)
                if (max_population is not None and l > max_population):
                    print(
                        f'\nSimulation.run(): Population exceeded {max_population}. Stopping simulation...')
                    break
                elif convergence_stop and is_converged:
                    print(
                        f'\nSimulation.run(): Convergence reached at t = {self.t}. Stopping simulation...')
                    break


        except KeyboardInterrupt:
            print('\nInterrupted by user...')

        self.finalize()

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time_str = f"{str(int(hours))}".rjust(
            2, '0') + ":" + f"{str(int(minutes))}".rjust(2, '0') + ":" + f"{str(int(seconds))}".rjust(2, '0')
        print()
        print(f'Simulation.run(): Done. Elapsed time: {elapsed_time_str}')
        print()

    def set_path(self, folder, alias=None, override=False):
        self.folder = folder
        self.alias = alias
        data_buffer_path = f'{folder}/data_buffer_{alias}.csv'
        state_buffer_path = f'{folder}/state_buffer_{alias}.csv'
        density_field_buffer_path = f'{folder}/density_field_buffer_{alias}.csv'
        if override and os.path.exists(data_buffer_path):
            do_override = input(
                f'Simulation.set_path(): OVERRIDE existing files in folder {folder} with alias {alias}? (Y/n):')
            if do_override.lower() != 'y':
                for path in [kwargs_path, data_buffer_path, state_buffer_path, density_field_buffer_path]:
                    if os.path.exists(path):
                        os.remove(path)
                if os.path.exists(figure_folder):
                    shutil.rmtree(figure_folder)
                else:
                    raise ValueError('Simulation.set_path(): Aborted by user...')             
        self.data_buffer = DataBuffer(file_path=data_buffer_path)
        self.state_buffer = StateBuffer(file_path=state_buffer_path)
        self.density_field_buffer = FieldBuffer(file_path=density_field_buffer_path, resolution=self.density_field_resolution)
        
        self.data_buffer.add(self.collect_data())
        self.data_buffer._flush_buffer()
        self.state_buffer.add(plants=self.plants, t=self.t)
        self.state_buffer._flush_buffer()
        self.density_field_buffer.add(
                        field=self.density_field.values, t=self.t)
        self.density_field_buffer._flush_buffer()

    def finalize(self):
        self.data_buffer.finalize()
        self.state_buffer.finalize()
        self.density_field_buffer.finalize()
        self.save_dict(path=f'{self.folder}/kwargs_{self.alias}')
        print(f'Simulation.finalize(): Time: {time.strftime("%H:%M:%S")}')

    def convergence_check(self, trend_window=6000, trend_threshold=1):
        data = self.data_buffer.get_data()[['Time', 'Biomass']]
        if data.shape[0] < 2:
            return False, -1, None
        time = data['Time'].values
        biomass = data['Biomass'].values
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
            (time[-1] - time[0]) > trend_window)
        return is_converged, convergence_factor, regression_line

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
                x=new_positions[i][0],
                y=new_positions[i][1],
                r=self.r_min,
                r_min=self.r_min,
                r_max=self.r_max,
                growth_rate=self.growth_rate,
                dispersal_range=self.dispersal_range,
                maturity_size=self.maturity_size
            )
            for i in spawn_indices
        ]

        self.add(new_plants)

    def get_collisions(self):
        collisions = []
        if len(self.plants) > 1:
            radii = np.array([plant.r for plant in self.plants])
            sparse_dist_matrix = self.kt.sparse_distance_matrix(self.kt, max_distance=2*np.max(radii))
            # Filter pairs that are within collision distance
            collision_dist_matrix = radii[:, None] + radii[None, :]
            collisions = [(i, j) for i, j in zip(*sparse_dist_matrix.nonzero()) if sparse_dist_matrix[i, j] < collision_dist_matrix[i, j]]
        return collisions
    
    def resolve_collisions(self, collisions):
        for i, j in collisions:
            self.plants[i].compete(self.plants[j])

    def pos_in_box(self, pos: np.ndarray) -> bool:
        pos_in_box = np.abs(pos[0]) < self.half_width and np.abs(
            pos[1]) < self.half_height
        return pos_in_box

    def local_density(self, pos: np.ndarray) -> float:
        if self.pos_in_box(pos):
            return self.density_field.query(pos)
        else:
            return 0

    def collect_data(self):
        sizes = np.array([plant.r for plant in self.plants])
        biomass = np.array([plant.area for plant in self.plants])
        biomass_total = sum(biomass)
        population = len(self.plants)
        data = pd.DataFrame(
            data={
                'Time': [self.t],
                'Biomass': [biomass_total],
                'Population': [population],
                'Precipitation': [self.precipitation],
            }
        )
        return data

    def initiate_uniform_radii(self, n: int, r_min: float, r_max: float):
        if self.plants != []:
            raise ValueError(
                "Simulation.initiate_uniform_radii(): The simulation is not empty. Please initialize an empty simulation.")
        r_min = r_min * self._m
        r_max = r_max * self._m

        new_plants = [
            Plant(
                id=self.id_generator.get_next_id(),
                x=np.random.uniform(-self.half_width, self.half_width),
                y=np.random.uniform(-self.half_height, self.half_height),
                r=np.random.uniform(r_min, r_max),
                r_min=r_min,
                r_max=r_max,
                growth_rate=self.growth_rate,
                dispersal_range=self.dispersal_range,
                maturity_size=self.maturity_size
            )
            for i in range(n)
        ]
        self.add(new_plants)
        self.initiate()
        self.data_buffer.add(data=self.collect_data())
        self.state_buffer.add(plants=self.plants, t=self.t)

    # def initiate_dense_distribution(self, n: int):
        # if self.plants != []:
        #     raise ValueError(
        #         "Simulation.initiate_dense_distribution(): The simulation is not empty. Please initialize an empty simulation.")
    #     mean_dbhs = np.array([0.05, 0.2, 0.4, 0.6, 0.85, 1.50])  # in m
    #     mean_rs = dbh_to_crown_radius(mean_dbhs) * self._m
    #     freqs = np.array([5800, 378.8, 50.98, 13.42, 5.62, 0.73])

    #     # Apply log transformation to the data
    #     log_mean_rs = np.log(mean_rs)

    #     # Calculate the KDE of the log-transformed data
    #     kde = gaussian_kde(log_mean_rs, weights=freqs, bw_method='silverman')

    #     # Resample from the KDE and transform back to the original space
    #     log_samples = kde.resample(n)[0]
    #     samples = np.exp(log_samples)

    #     # Create the plants
    #     plants = [
    #         Plant(
    #             id=self.id_generator.get_next_id(),
    #               x = np.random.uniform(-self.half_width, self.half_width),
    #               y = np.random.uniform(-self.half_height, self.half_height),
    #             r=r,
    #             r_min=self.r_min,
    #             r_max=self.r_max,
    #             growth_rate=self.growth_rate,
    #             dispersal_range=self.dispersal_range,
    #             maturity_size=self.maturity_size
    #         )
    #         for i, r in enumerate(samples)
    #     ]
    #     self.add(plants)
    #     self.initiate()
    #     self.data_buffer.add(data=self.collect_data())
    #     self.state_buffer.add(plants=state=self.plants, t=self.t)

    # def initiate_packed_distribution(self, r: float):
        # if self.plants != []:
        #     raise ValueError(
        #         "Simulation.initiate_packed_distribution(): The simulation is not empty. Please initialize an empty simulation.")
    #     plants = []
    #     dx = 2 * r
    #     dy = np.sqrt(3) * r

    #     for i, x in enumerate(np.arange(-self.half_width + r, self.half_width - r, dx)):
    #         for j, y in enumerate(np.arange(-self.half_height + r, self.half_height - r, dy)):

    #             if j % 2 == 0:
    #                 x += r
    #             if j % 2 == 1:
    #                 x -= r
    #             pos = np.array([x, y])
    #             plants.append(
    #                 Plant(
    #                     id=self.id_generator.get_next_id(),
    #                     pos=pos,
    #                     r=r,
    #                     r_min=self.r_min,
    #                     r_max=self.r_max,
    #                     growth_rate=self.growth_rate,
    #                     dispersal_range=self.dispersal_range,
    #                     maturity_size=self.maturity_size 
    #                 )
    #             )

    #     self.add(plants)
    #     self.initiate()
    exclude_default = [
        'alias',
        'biomass_buffer',
        'buffer_preset_times',
        'data_buffer',
        'density_field',
        'density_field_buffer',
        'folder',
        'half_height',
        'half_width',
        'id_generator',
        'kt',
        'size_buffer',
        'plants',
        'state_buffer',
        'verbose'
    ]

    def save_dict(self, path: str, exclude: Optional[List[str]] = exclude_default):
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

        save_kwargs(dict_temp, path, exclude=exclude)

    def print_dict(self, exclude=exclude_default):
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
        self.plants = []
        self.kt = None
        self.state_buffer = None
        self.data_buffer = None
        self.density_field_buffer = None
        self.density_field = None

    def plot_buffers(self, title=None, convergence=True, n_plots=20, fast=False):
        db_fig, db_ax, db_title = DataBuffer.plot(data = self.data_buffer.get_data(), title=title)
        sb_fig, sb_ax, sb_title = StateBuffer.plot(data = self.state_buffer.get_data(), title=title, n_plots=n_plots, fast=fast)
        dfb_fig, dfb_ax, dfb_title = FieldBuffer.plot(data = self.density_field_buffer.get_data(), title=title, n_plots=n_plots)
        figs = [db_fig, sb_fig, dfb_fig]
        axs = [db_ax, sb_ax, dfb_ax]
        titles = [db_title, sb_title, dfb_title]
        return figs, axs, titles

    def plot(self):
        state = pd.DataFrame([[p.id, p.x, p.y, p.r, self.t]
                             for p in self.plants], columns=['id', 'x', 'y', 'r', 't'])
        field = self.density_field.values

        sb_fig, sb_ax, sb_title = StateBuffer.plot_state(size=6, state=state)
        fb_fig, fb_ax, fb_title = FieldBuffer.plot_field(size=6, field=field)

        figs = [sb_fig, fb_fig]
        axs = [sb_ax, fb_ax]
        titles = [sb_title, fb_title]
        return figs, axs, titles