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

from mods.buffers import *
from mods.fields import *
from mods.plant import *
from mods.utilities import *
from mods.spatial import *


modi_path_kwargs = '../../default_kwargs.json'
local_path_kwargs = 'default_kwargs.json'
path_kwargs = modi_path_kwargs if os.path.exists(
    modi_path_kwargs) else local_path_kwargs

with open(path_kwargs, 'r') as file:
    default_kwargs = json.load(file)


class IDGenerator:
    def __init__(self, start=0):
        self.current_id = start

    def get_next_id(self):
        self.current_id += 1
        return self.current_id


class Simulation:
    def __init__(self, folder, alias='alias', **kwargs):
        kwargs_path = f'{folder}/kwargs-{alias}.json'
        data_buffer_path = f'{folder}/data_buffer-{alias}.csv'
        state_buffer_path = f'{folder}/state_buffer-{alias}.csv'
        density_field_buffer_path = f'{folder}/density_field_buffer-{alias}.csv'
        figure_folder = f'{folder}/figures'

        override = kwargs.get('override', False)
        override_force = kwargs.get('override_force', False)
        if (override or override_force) and os.path.exists(kwargs_path):
            if not override_force:
                override_input = input(
                    f'Simulation.__init__(): OVERRIDE existing files in folder "{folder}" with alias "{alias}"? (Y/n):')
            else:
                print()
                print("##################################################")
                print("##### WARNING: FORCED OVERRIDING IS ENABLED. #####")
                print("##################################################")
                print()
                override_input = 'y'
            if override_input.lower() != 'y':
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

        self.species_list = kwargs.get('species_list', [])
        if os.path.exists(kwargs_path):
            print(f'Simulation.__init__(): Loading kwargs from {kwargs_path}')
            with open(kwargs_path, 'r') as file:
                kwargs = json.load(file)

            species_files = [f for f in os.listdir(folder) if f.startswith(
                'species') and f.endswith(f'-{alias}.json')]
            species_list = []
            for species_file in species_files:
                with open(os.path.join(folder, species_file), 'r') as sf:
                    species_kwargs = json.load(sf)
                    species = PlantSpecies(**species_kwargs)
                    species_list.append(species)
            self.species_list = sorted(
                species_list, key=lambda x: x.species_id)
            print(
                f'Simulation.__init__(): Loaded {len(self.species_list)} species.')

        self.t = 0
        self.box = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        self.__dict__.update(default_kwargs)
        self.__dict__.update(kwargs)
        self.folder = folder
        self.alias = alias

        self.plants = []
        
        _m = 1 / self.L
        self.conversion_factors_default = {
            'r_min': _m,
            'r_max': _m,
            'dispersal_range': _m,
            'spawn_rate': self.time_step,
            'growth_rate': _m * self.time_step,
            'density_range': _m,
            'maturity_size': _m,
        }


        if self.species_list == []:
            self.species_list = [PlantSpecies()]            
        for s in self.species_list:
            species_kwargs_path=f'{folder}/species{str(s.species_id).replace('-', '_')}-{alias}.json'
            
            if kwargs.get('convert_kwargs', True) == True:
                save_dict(path = species_kwargs_path, d=s.__dict__)
                converted_dict = convert_dict(
                    d=s.__dict__, conversion_factors=self.conversion_factors_default, reverse=False)
                s.__dict__.update(converted_dict)
            else:
                converted_dict = convert_dict(
                    d=s.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
                save_dict(path = species_kwargs_path, d=converted_dict)

        self.maximum_plant_size = max(
            [s.r_max for s in self.species_list])
        self.kt = None

        self.id_generator = IDGenerator()

        self.data_buffer = DataBuffer(file_path=data_buffer_path)
        self.state_buffer = StateBuffer(file_path=state_buffer_path)
        self.density_field_buffer = FieldBuffer(
            file_path=density_field_buffer_path, resolution=self.density_field_resolution)
        self.density_field = DensityFieldCustom(box=self.box,
            resolution=self.density_field_resolution,
        )
        
        exisiting_data = self.data_buffer.get_data()
        if not exisiting_data.empty:
            self.precipitation = exisiting_data['Precipitation'].iloc[-1]
            self.t = exisiting_data['Time'].iloc[-1]

        if kwargs.get('state', None) is not None:
            self.set_state(kwargs['state'])
            self.__dict__.pop('state')
        else:
            state_df = self.state_buffer.get_last_state()
            self.set_state(state_df)


        if kwargs.get('convert_kwargs', True) == True:
            save_dict(path = kwargs_path, d=self.__dict__, exclude=self.exclude_default)
            converted_dict = convert_dict(
                d=self.__dict__, conversion_factors=self.conversion_factors_default, reverse=False)
            self.__dict__.update(converted_dict)
        else:
            converted_dict = convert_dict(
                d=self.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
            save_dict(path = kwargs_path, d=self.__dict__, exclude=self.exclude_default)
            
        if 'convert_kwargs' in self.__dict__:
            self.__dict__.pop('convert_kwargs')

        print(f'Simulation.__init__(): Time: {time.strftime("%H:%M:%S")}')
        print(f'Simulation.__init__(): Folder: {folder}, Alias: {alias}')

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
        # Update all plants based on the current state of the simulation
        dispersed_positions = np.empty((0, 2))
        parent_species = []
        for plant in self.plants:
            plant.grow()
            pos = plant.disperse(self)
            if len(pos) > 0:
                dispersed_positions = np.vstack((dispersed_positions, pos))
                parent_species.extend([PlantSpecies(**plant.__dict__)] * len(pos))
            plant.mortality()

        # Spawn new plants
        self.attempt_germination(dispersed_positions, parent_species)
        self.attempt_spawn(n=self.spawn_rate, species_list=self.species_list)
        
        # Check for and resolve collisions
        # self.resolve_collisions(collision_indices)
        positions = np.array([[plant.x, plant.y] for plant in self.plants if not plant.is_dead])
        if len(positions) > 0:
            radii = np.array([plant.r for plant in self.plants if not plant.is_dead])
            
            if self.boundary_condition == 'periodic':
                positions_shifted, index_pairs, was_shifted = positions_shift_periodic(boundary=self.box, positions=positions, radii=radii, duplicates=True)
                radii_shifted = radii[index_pairs[:, 0]]
                collision_indices = get_all_collisions(positions_shifted, radii_shifted)
                
                indices_final_shifted = index_pairs[:, 1][~was_shifted]
                if len(collision_indices) > 0:
                    collision_losers_indices = np.unique([j if radii_shifted[i] > radii_shifted[j] else i for i, j in collision_indices])
                    indices_final_shifted = np.setdiff1d(index_pairs[:, 1][~was_shifted], collision_losers_indices)
                    
                indices_final = index_pairs[:, 0][indices_final_shifted]
            else:
                collision_indices = get_all_collisions(positions, radii)
                if len(collision_indices) > 0:
                    collision_losers_indices = np.unique([j if radii[i] > radii[j] else i for i, j in collision_indices])
                    indices_final = np.setdiff1d(np.arange(len(positions)), collision_losers_indices)
        else:
            indices_final = np.array([])
            
        # Collect non-dead plants and add them to the new state, and make sure all new plants get a unique id
        new_plants = []
        for i, plant in enumerate(self.plants):
            if i not in indices_final:
                plant.is_dead = True
        
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
            
        if self.verbose:
            runtime_str = time.strftime("%H:%M:%S")

            if self.t % 3 == 0:
                dots = '.  '
            elif self.t % 3 == 1:
                dots = '.. '
            else:
                dots = '...'

            data = self.collect_data()
            t, biomass, population, precipitation = data[[
                'Time', 'Biomass', 'Population', 'Precipitation']].values.reshape(-1)
            t = float(round(t, 2))

            print(f'{dots} Time: {runtime_str}' + ' '*5 + f'|  {t=:^8}  |  N = {population:<6}  |  B = {np.round(biomass, 4):<6}  |  P = {np.round(precipitation, 6):<8}', end='\r')
            
            if len(self.species_list) > 1:
                if _ % 100 == 0:
                    species_counts = {species.species_id: 0 for species in self.species_list}
                    for plant in self.plants:
                        species_counts[plant.species_id] += 1
                    print()
                    print(f'Species counts: {species_counts}')
                    print()

    def run(self, T, min_population=None, max_population=None, transient_period=0, delta_p=0, convergence_stop=False):

        start_time = time.time()
        n_iter = int(np.ceil(T / self.time_step))
        print(
            f'Simulation.run(): Running simulation for {n_iter} iterations from t = {self.t}...')
        try:
            for _ in range(0, n_iter):
                if _ > transient_period:
                    self.precipitation = max(0, self.precipitation + delta_p)
                self.step()

                # if the population exceeds the maximum allowed, stop the simulation
                l = len(self.plants)
                if (max_population is not None and l > max_population):
                    print(
                        f'\nSimulation.run(): Population exceeded {max_population}. Stopping simulation...')
                    break
                elif (min_population is not None and l < min_population):
                    print(
                        f'\nSimulation.run(): Population below {min_population}. Stopping simulation...')
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

    def set_folder(self, folder, alias=None, override=False):
        self.folder = folder
        if alias is None:
            alias = self.alias
        else:
            self.alias = alias

        kwargs_path = f'{folder}/kwargs-{alias}.json'
        data_buffer_path = f'{folder}/data_buffer-{alias}.csv'
        state_buffer_path = f'{folder}/state_buffer-{alias}.csv'
        density_field_buffer_path = f'{folder}/density_field_buffer-{alias}.csv'
        figure_folder = f'{folder}/figures'

        if override and os.path.exists(kwargs_path):
            do_override = input(
                f'Simulation.__init__(): OVERRIDE existing files in folder "{folder}" with alias "{alias}"? (Y/n):')
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

        self.data_buffer = DataBuffer(file_path=data_buffer_path)
        self.state_buffer = StateBuffer(file_path=state_buffer_path)
        self.density_field_buffer = FieldBuffer(
            file_path=density_field_buffer_path, resolution=self.density_field_resolution)
        for s in self.species_list:
            converted_dict = convert_dict(
                d=s.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
            save_dict(
                path=f'{folder}/species_{s.species_id}-{alias}.json', d=converted_dict)

        converted_dict = convert_dict(
            d=self.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
        save_dict(d=converted_dict, path=kwargs_path,
                  exclude=self.exclude_default)

    def set_state(self, state_df):
        self.plants = []
        if not state_df.empty:
            if 'species' not in state_df.columns:
                warnings.warn(
                    'Simulation.__init__(): "species" column not found in state_buffer. Assuming species_id = -1 for all plants.')
                state_df['species'] = -1

            for (id, x, y, r, species_id) in state_df[['id', 'x', 'y', 'r', 'species']].values:
                s = next(
                    (s for s in self.species_list if s.species_id == species_id), None)
                if s is None:
                    raise ValueError(
                        f'Simulation.__init__(): Species with id {species_id} not found in species_list.')
                self.plants.append(s.create_plant(id=id, x=x, y=y, r=r))

            self.t = state_df['t'].values[-1]

        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)
        self.data_buffer.add(data=self.collect_data())
        self.state_buffer.add(plants=self.plants, t=self.t)
        self.density_field_buffer.add(
            field=self.density_field.values, t=self.t)
        
    def finalize(self):
        self.data_buffer.finalize()
        self.state_buffer.finalize()
        self.density_field_buffer.finalize()
        converted_dict = convert_dict(
            d=self.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
        save_dict(d=converted_dict,
                  path=f'{self.folder}/kwargs-{self.alias}', exclude=self.exclude_default)
        print(f'Simulation.finalize(): Time: {time.strftime("%H:%M:%S")}')

    def convergence_check(self, trend_window=5000, trend_threshold=1):
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
    
    def remove_fraction(self, fraction):
        indices = np.random.choice(len(self.plants), int(fraction * len(self.plants)), replace=False)
        self.plants = [self.plants[i] for i in range(len(self.plants)) if i not in indices]
        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)

    def attempt_spawn(self, n, species_list=[]):
        if species_list == []:
            species_list = self.species_list

        # Take care of the decimal part of the spawn rate n
        decimal_part = n - int(n)
        n = int(n)
        if np.random.rand() < decimal_part:
            n += 1

        # Generate new positions and species
        new_positions = np.random.uniform(self.box[:, 0], self.box[:, 1], (n, 2))
        new_species = np.random.choice(species_list, n)
        
        self.attempt_germination(new_positions, new_species)

    def resolve_collisions(self, collisions):
        for i, j in collisions:
            self.plants[i].compete(self.plants[j])
    
    def local_density(self, pos):
        return self.density_field.query(pos)

    def attempt_germination(self, positions_to_germinate, parent_species):
        if len(positions_to_germinate) == 0:
            return
        parent_species = np.atleast_1d(parent_species)
        if len(parent_species) == 1:
            parent_species = np.repeat(parent_species, len(positions_to_germinate))
        elif len(parent_species) != len(positions_to_germinate):
            raise ValueError(
                'Simulation.attempt_germination(): The number of parent species must match the number of positions.')
        
        # ENFORCE BOUNDARIES
        boundary = self.box
        
        if self.boundary_condition == 'periodic':
            positions_new, index_pairs, was_shifted = positions_shift_periodic(boundary=boundary, positions=positions_to_germinate, duplicates=False)
            
            is_beyond_boundary = boundary_check(boundary=boundary, positions=positions_new)
            if is_beyond_boundary.any():
                print(f'Simulation.attempt_germination(): {np.sum(is_beyond_boundary)}/{len(positions_new)} positions are beyond the boundary.')
                for i, j in index_pairs:
                    if is_beyond_boundary[j].any():
                        print(f'Simulation.attempt_germination(): {positions_to_germinate[i]} -> {positions_new[j]}')
            
            positions_to_germinate = positions_new
            parent_species = parent_species[index_pairs[:, 0]]  
        else:
            is_beyond_boundary = np.any(boundary_check(boundary=boundary, positions=positions_to_germinate), axis=1)
            positions_to_germinate = positions_to_germinate[~is_beyond_boundary]
            parent_species = parent_species[~is_beyond_boundary]
            
        # # COLLISION CHECK
        # plant_positions = np.array([[plant.x, plant.y] for plant in self.plants])
        # radii = np.array([plant.r for plant in self.plants])
        # positions_before_collision_check = positions_to_germinate.copy()
        # parent_species_before_collision_check = parent_species.copy()
        # for i, pos in enumerate(positions_to_germinate):
        #     collision_indices = get_collisions_for_point(
        #         positions=plant_positions,
        #         radii=radii,
        #         point=pos,
        #         radius=parent_species_before_collision_check[i].r_min
        #     )
        #     if len(collision_indices) > 0:
        #         positions_to_germinate = np.delete(positions_before_collision_check, i, axis=0)
        #         parent_species = np.delete(parent_species_before_collision_check, i)
        
        species_germination_chances = np.array([s.germination_chance for s in parent_species])         
        germination_chances = np.maximum(self.land_quality, self.local_density(positions_to_germinate)) * self.precipitation * species_germination_chances
        random_values = np.random.uniform(0, 1, len(positions_to_germinate))
        germination_indices = np.where(germination_chances > random_values)[0]

        new_plants = []
        for i in germination_indices:
            new_plant = parent_species[i].create_plant(id=self.id_generator.get_next_id(), x=positions_to_germinate[i, 0], y=positions_to_germinate[i, 1], r=parent_species[i].r_min)
            new_plants.append(new_plant)
                    
        if len(new_plants) > 0:
            self.add(new_plants)
            # print(f'Simulation.attempt_germination(): {len(new_plants)} plants germinated.')

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

    def initiate_uniform_radii(self, n, species_list=[], box=None):
        if self.plants != []:
            raise ValueError(
                "Simulation.initiate_uniform_radii(): The simulation is not empty. Please initialize an empty simulation.")

        if species_list == []:
            species_list = self.species_list

        if box is None:
            box = self.box

        new_plants = []
        for i in range(n):
            current_species = np.random.choice(species_list)
            new_plants.append(
                current_species.create_plant(
                    id=self.id_generator.get_next_id(),
                    x=np.random.uniform(*box[0]),
                    y=np.random.uniform(*box[1]),
                    r=np.random.uniform(current_species.r_min,
                                        current_species.r_max))
            )

        self.add(new_plants)
        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)
        self.data_buffer.add(data=self.collect_data())
        self.state_buffer.add(plants=self.plants, t=self.t)
        
    def initiate_non_overlapping(self, n=None, species_list=[], max_attempts=None, box=None, target_density=None):
        if n is None and target_density is None:
            raise ValueError("Simulation.initiate_non_overlapping(): Either n or target_density must be specified.")
        
        if self.plants != []: 
            warnings.warn("Simulation.initiate_non_overlapping(): The simulation is not empty. Do you want to continue? (Y/n):")
            user_input = input()
            if user_input.lower() != 'y':
                raise ValueError("Simulation.initiate_non_overlapping(): Aborted by user.")
        if species_list == []:
            species_list = self.species_list
        L = self.L
        new_plants = []
        attempts = 0
        if max_attempts is None:
            max_attempts = 200 * n if n is not None else 50 * L
        species_counts = {s.species_id: 0 for s in species_list}
        weights = np.ones(len(species_list)) / len(species_list)
        
        if box is None:
            box = self.box
        
        
        try:
            stop_condition = False
            biomass = 0
            while stop_condition is False and attempts < max_attempts:
                if len(new_plants) > 0:
                    fractions = np.array([species_counts[s.species_id]
                            for s in species_list]) / len(new_plants)
                    weights = 1 - fractions
                    if weights.sum() == 0:
                        weights = np.ones(len(species_list)) / len(species_list)
                    else:
                        weights /= weights.sum()
                
                current_species = np.random.choice(species_list, p=weights)
                new_r = np.random.uniform(current_species.r_min, current_species.r_max)
                new_x = np.random.uniform(*box[0])
                new_y = np.random.uniform(*box[1])
                new_pos = np.array([new_x, new_y])

                attempts += 1
                if all(np.linalg.norm(new_pos - plant.pos()) >= new_r + plant.r for plant in new_plants):
                    new_plants.append(current_species.create_plant(
                    id=len(new_plants), x=new_x,  y=new_y, r=new_r))
                    species_counts[current_species.species_id] += 1
                    biomass = sum([plant.area for plant in new_plants])
                
                    print(
                    f'Simulation.spawn_non_overlapping(): {len(new_plants) = }/{n}   {biomass = :.3f}/{target_density}   {attempts = }/{max_attempts}', end='\r')
                        
                
                stop_condition = (len(new_plants) >= n) if (n is not None) else (sum([s.area for s in new_plants]) >= target_density)
        except KeyboardInterrupt:
            print('\nInterrupted by user...')

        print(f"Simulation.spawn_non_overlapping(): {len(new_plants)} plants with biomass {biomass:.3f} was placed after {attempts}/{max_attempts} attempts.")
        print()

        self.add(new_plants)
        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)
        self.data_buffer.add(data=self.collect_data())
        self.state_buffer.add(plants=self.plants, t=self.t)

    exclude_default = [
        'alias',
        'biomass_buffer',
        'buffer_preset_times',
        'buffer_size',
        'buffer_skip',
        'conversion_factors_default',
        'data_buffer',
        'density_check_radius',
        'density_field',
        'density_field_buffer',
        'folder',
        'half_height',
        'half_width',
        'box',
        'id_generator',
        'kt',
        'maximum_plant_size',
        'plants',
        'precipitation',
        'size_buffer',
        'species_list',
        'state_buffer',
        'state',
        'convert_dict',
        'verbose'
    ]

    def plot_buffers(self, title='', convergence=True, n_plots=20, fast=False):
        db_fig, db_ax, db_title = DataBuffer.plot(
            data=self.data_buffer.get_data(), title=title)
        sb_fig, sb_ax, sb_title = StateBuffer.plot(
            data=self.state_buffer.get_data(), title=title, n_plots=n_plots, box=self.box, boundary_condition=self.boundary_condition, fast=fast)
        # dfb_fig, dfb_ax, dfb_title = FieldBuffer.plot(
        #     data=self.density_field_buffer.get_data(), title=title, n_plots=n_plots, box=self.box, boundary_condition=self.boundary_condition)
        # figs = [db_fig, sb_fig, dfb_fig]
        # axs = [db_ax, sb_ax, dfb_ax]
        # titles = [db_title, sb_title, dfb_title]
        figs = [db_fig, sb_fig]
        axs = [db_ax, sb_ax]
        titles = [db_title, sb_title]
        return figs, axs, titles

    def plot(self, fast = False, field_buffer=True):
        state = pd.DataFrame([[p.id, p.x, p.y, p.r, p.species_id, self.t]
                             for p in self.plants], columns=['id', 'x', 'y', 'r', 'species', 't'])
        field = self.density_field.values

        sb_fig, sb_ax, sb_title = StateBuffer.plot_state(size=6, state=state, box=self.box, boundary_condition=self.boundary_condition, fast=fast)
        if field_buffer:
            fb_fig, fb_ax, fb_title = FieldBuffer.plot_field(size=6, field=field, box=self.box, boundary_condition=self.boundary_condition)

            figs = [sb_fig, fb_fig]
            axs = [sb_ax, fb_ax]
            titles = [sb_title, fb_title]
        else:
            figs = [sb_fig]
            axs = [sb_ax]
            titles = [sb_title]
        return figs, axs, titles
    
    def plot_state(self, fast = False):
        return self.plot(fast=fast, field_buffer=False)
