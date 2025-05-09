import copy
import json
import time
import warnings
import os
from typing import *
import shutil

import matplotlib.pyplot as plt
import matplotlib
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
        
        # If the folder does not exist, ask the user if they want to create it
        if not os.path.exists(folder):
            create_folder = input(f'Folder "{folder}" does not exist. Do you want to create it? (Y/n): ')
            if create_folder.lower() != 'y':
                raise ValueError('Simulation.__init__(): Aborted by user...')
            os.makedirs(folder, exist_ok=True)
        
        # Define paths for the data
        kwargs_path = f'{folder}/kwargs-{alias}.json'
        data_buffer_path = f'{folder}/data_buffer-{alias}.csv'
        state_buffer_path = f'{folder}/state_buffer-{alias}.csv'
        density_field_buffer_path = f'{folder}/density_field_buffer-{alias}.csv'
        figure_folder = f'{folder}/figures'

        # Check if data already exists and ask the user if they want to override it
        override = kwargs.get('override', False)
        force = kwargs.get('force', False)
        if (override or force) and os.path.exists(kwargs_path):
            if not force:
                override_input = input(
                    f'Simulation.__init__(): OVERRIDE existing files in folder "{folder}" with alias "{alias}"? (Y/n):')
            else:
                print()
                print("##################################################")
                print("##### WARNING: FORCED OVERRIDING IS ENABLED. #####")
                print("##################################################")
                print()
                override_input = 'y'
            
            if not (override_input.lower() == 'y' or force):
                raise ValueError('Simulation.__init__(): Aborted by user...')
            else:
                # Remove existing files if the user confirms the override
                for path in [kwargs_path, data_buffer_path, state_buffer_path, density_field_buffer_path]:
                    if os.path.exists(path):
                        os.remove(path)
                if os.path.exists(figure_folder):
                    for file in os.listdir(figure_folder):
                        if alias in file:
                            os.remove(os.path.join(figure_folder, file))

        # Create the folder for figures if it doesn't exist
        os.makedirs(folder + '/figures', exist_ok=True)
        
        # Load the kwargs from the file if it exists
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

        self.__dict__.update(default_kwargs)
        self.__dict__.update(kwargs)
        self.box = np.array(self.box, dtype=float)
        
        self.folder = folder
        self.alias = alias

        self.plants = PlantCollection()
        
        self.conversion_factors_default = {
            'r_min': self.L, # m
            'r_max': self.L, # m
            'dispersal_range': self.L, # m
            'density_range': self.L, # m
            'maturity_size': self.L, # m
            'spawn_rate': 1/ self.L**2 / self.time_step, # 1/ m2 / yr
            'growth_rate': self.L/self.time_step, # m / yr
        }

        # ensure that the species list is not empty
        if self.species_list == []:
            self.species_list = [PlantSpecies()]
            
        # Define species kwargs paths and save them         
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

        # Define the buffers and the density field
        self.data_buffer = DataBuffer(file_path=data_buffer_path)
        self.state_buffer = StateBuffer(file_path=state_buffer_path)
        if self.density_scheme == 'global':
            self.density_field_resolution = 1
        # self.density_field_buffer = FieldBuffer(
        #     file_path=density_field_buffer_path, resolution=self.density_field_resolution, skip=self.density_field_buffer_skip)
        self.density_field = DensityFieldCustom(box=self.box,
            resolution=self.density_field_resolution,
        )
        
        # Import existing data from the data buffer if it exists
        exisiting_data = self.data_buffer.get_data()
        if not exisiting_data.empty:
            self.precipitation = exisiting_data['Precipitation'].iloc[-1]
            self.t = exisiting_data['Time'].iloc[-1]

        # Import existing state from the state buffer if it exists
        if kwargs.get('state', None) is not None:
            print(f'Simulation.__init__(): Loading state from kwargs["state"]')
            self.set_state(kwargs['state'])
            self.__dict__.pop('state')
        elif not override:
            print(f'Simulation.__init__(): Loading state from state_buffer')
            state_df = self.state_buffer.get_last_state()
            if not state_df.empty:
                self.set_state(state_df)

        # Convert the kwargs to the correct units if needed
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

        self.seed = kwargs.get('seed', None)            
        print(f'Simulation.__init__(): Time: {time.strftime("%H:%M:%S")}')
        print(f'Simulation.__init__(): Folder: {folder}, Alias: {alias}')

    def add(self, plant):
        if isinstance(plant, Plant):
            self.plants.append(plant)
        elif isinstance(plant, PlantCollection):
            self.plants.add_plants(plant.plants)
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

    def update_kdtree(self, plants):
        if len(plants) == 0:
            self.kt = None
        else:
            self.kt = KDTree(plants.positions)

    def step(self):
        # Update all plants based on the current state of the simulation
        self.plants.grow()
        self.plants.mortality()
        
        # Disperse and spawn new plants
        disperse_positions, parent_species = self.plants.disperse(self)
        self.attempt_germination(disperse_positions, parent_species)
        self.attempt_spawn(n=self.spawn_rate)
        
        # Check for and resolve collisions
        self.resolve_collisions(positions=self.plants.positions, radii=self.plants.radii)
        
        # Remove dead plants
        self.plants.remove_dead_plants()

        prev_t = self.t
        self.t += self.time_step

        # Update necessary data structures
        self.density_field.update(self.plants, density_scheme=self.density_scheme)
        self.update_kdtree(self.plants)

        self.data_buffer.add(self.get_data())

        prev_mod_state = prev_t % self.state_buffer.skip
        mod_state = self.t % self.state_buffer.skip
        do_save_state = prev_mod_state >= mod_state
        # prev_mod_density_field = prev_t % self.density_field_buffer.skip
        # mod_density_field = self.t % self.density_field_buffer.skip
        # do_save_density_field = prev_mod_density_field >= mod_density_field

        if do_save_state:
            self.state_buffer.add(plants=self.plants, t=self.t)
        # if do_save_density_field:
        #     self.density_field_buffer.add(
        #         field=self.density_field.values, t=self.t)
            
        if self.verbose:
            self.print()
            
            if self.t % 100 == 0:
                print()
                if len(self.species_list) > 1:
                    species_counts = {species.species_id: 0 for species in self.species_list}
                    for plant in self.plants:
                        species_counts[plant.species_id] += 1
                    print(f'Species counts: {species_counts}')
                    print()

    def run(self, T, min_population=None, max_population=None, max_biomass = None, transient_period=0, dp=0, convergence_stop=False):
        run_start_time = self.t
        start_time = time.time()
        n_iter = int(np.ceil(T / self.time_step))
        print(
            f'Simulation.run(): Running simulation for {n_iter} iterations from t = {self.t}...')
        try:
            self.set_seed(self.seed)
            self.is_running = True
            self.print()
            print('\nSimulation.run(): Starting simulation...')
            for _ in range(0, n_iter):
                if _ > transient_period:
                    self.precipitation = min(1, max(0, self.precipitation + dp))
                self.step()

                # if the population exceeds the maximum allowed, stop the simulation
                l = len(self.plants)
                if isinstance(convergence_stop, int):
                    conv = self.convergence(t_min=run_start_time, trend_window=convergence_stop)
                else:
                    conv = self.convergence(t_min=run_start_time)
                stop_conditions = [
                    (max_population is not None and l > max_population, f'Population exceeded {max_population}'),
                    (min_population is not None and l < min_population, f'Population below {min_population}'),
                    (max_biomass is not None and self.get_biomass() > max_biomass, f'Biomass exceeded {max_biomass}'),
                    (convergence_stop and conv, f'Convergence reached at t = {self.t}')
                ]
                for condition, message in stop_conditions:
                    if condition:
                        print(f'\nSimulation.run(): {message}. Stopping simulation...')
                        break
                if condition:
                    break

        except KeyboardInterrupt:
            print('\nInterrupted by user...')
        self.is_running = False
        self.finalize()

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed_time_str = f"{str(int(hours))}".rjust(
            2, '0') + ":" + f"{str(int(minutes))}".rjust(2, '0') + ":" + f"{str(int(seconds))}".rjust(2, '0')
        print()
        print(f'Simulation.run(): Done. Elapsed time: {elapsed_time_str}')
        print()

    def find_collision_losers(self, collision_indices, radii, competition_scheme):
            if len(collision_indices) == 0:
                return np.empty(0, dtype=int)
            
            collision_indices = np.random.permutation(collision_indices)
            
            if competition_scheme.lower() == 'all':
                return np.unique([
                    j if radii[i] > radii[j] else i
                    for i, j in collision_indices
                ])
            elif competition_scheme.lower() == 'sparse':
                collision_losers_indices = set()
                for i, j in collision_indices:
                    if i not in collision_losers_indices and j not in collision_losers_indices:
                        if radii[i] > radii[j]:
                            collision_losers_indices.add(j)
                        else:
                            collision_losers_indices.add(i)
                return np.array(list(collision_losers_indices), dtype=int)
            else:
                raise ValueError(
                    f'Simulation.resolve_collisions(): Competition scheme "{competition_scheme}" not recognized.')

    def resolve_collisions(self, positions, radii, boundary_condition=None, competition_scheme=None):
        if len(positions) == 0:
            return np.empty(0, dtype=int)
        
        boundary_condition = boundary_condition or self.boundary_condition
        competition_scheme = competition_scheme or self.competition_scheme
        indices_dead = np.empty(0, dtype=int)

        if boundary_condition.lower() == 'periodic':
            positions_shifted, index_pairs, was_shifted = positions_shift_periodic(
                box=self.box, positions=positions, radii=radii, duplicates=True)
            radii_shifted = radii[index_pairs[:, 0]]
            collision_indices_shifted = get_all_collisions(positions_shifted, radii_shifted)
            
            collision_losers_indices = self.find_collision_losers(collision_indices_shifted, radii_shifted, competition_scheme)
            indices_dead_shifted = list(set(index_pairs[:, 1][~was_shifted]).intersection(collision_losers_indices))
            indices_dead = index_pairs[:, 0][indices_dead_shifted]

        elif boundary_condition.lower() == 'box':
            collision_indices = get_all_collisions(positions, radii)
            
            collision_losers_indices = self.find_collision_losers(collision_indices, radii, competition_scheme)
            indices_dead = collision_losers_indices

        else:
            raise ValueError(
                f'Simulation.step(): Boundary condition "{boundary_condition}" not recognized.')

        self.plants.is_dead[indices_dead] = True
        ### SHOULD BE OPTIMIZED ###
        for i in indices_dead:
            self.plants[i].is_dead = True
        return indices_dead
        
    def _flush_buffers(self):
        self.data_buffer._flush_buffer()
        self.state_buffer._flush_buffer()
        # self.density_field_buffer._flush_buffer()

    def set_seed(self, seed=None):
        if seed == 'random':
            seed = np.random.randint(0, 2**32, dtype=np.uint32)
        elif not isinstance(seed, (int, np.int32, np.int64, np.uint32, np.uint64)):
            raise ValueError(
                f'Simulation.set_seed(): Seed must be an integer or "random". Got {type(seed)} instead.')
        self.seed = seed
        np.random.seed(self.seed)
        print(f'Simulation.set_seed(): Seed set to {self.seed}')
        return self.seed

    def set_folder(self, folder, alias=None, override=False, force=False):
        self.folder = folder
        self.alias = alias or self.alias

        kwargs_path = f'{folder}/kwargs-{self.alias}.json'
        data_buffer_path = f'{folder}/data_buffer-{self.alias}.csv'
        state_buffer_path = f'{folder}/state_buffer-{self.alias}.csv'
        # density_field_buffer_path = f'{folder}/density_field_buffer-{self.alias}.csv'
        figure_folder = f'{folder}/figures'

        if force:
            print()
            print("##################################################")
            print("##### WARNING: FORCED OVERRIDING IS ENABLED. #####")
            print("##################################################")
            print()
            
        if (force or override) and os.path.exists(folder):
            if not force:
                do_override = input(
                    f'Simulation.set_folder(): OVERRIDE existing files in folder "{folder}" with alias "{self.alias}"? (Y/n):')
                do_override = do_override.lower()
                if do_override != 'y':
                    raise ValueError('Simulation.set_folder(): Aborted by user...')
            if force or do_override.lower() == 'y':
                for file in os.listdir(folder):
                    if self.alias in file:
                        file_path = os.path.join(folder, file)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            

        os.makedirs(folder + '/figures', exist_ok=True)

        self.data_buffer = DataBuffer(file_path=data_buffer_path)
        self.state_buffer = StateBuffer(file_path=state_buffer_path)
        # self.density_field_buffer = FieldBuffer(
        #     file_path=density_field_buffer_path, resolution=self.density_field_resolution)
        for s in self.species_list:
            converted_dict = convert_dict(
                d=s.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
            save_dict(
                path=f'{folder}/species_{s.species_id}-{self.alias}.json', d=converted_dict)

        converted_dict = convert_dict(
            d=self.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
        save_dict(d=converted_dict, path=kwargs_path,
                  exclude=self.exclude_default)
        
        return self

    def set_state(self, state_df):
        # self.plants = []
        self.plants = PlantCollection()
        if not state_df.empty:
            if 'species' not in state_df.columns:
                warnings.warn(
                    'Simulation.set_state(): "species" column not found in state_buffer. Assuming species_id = -1 for all plants.')
                state_df['species'] = -1

            for (id, x, y, r, species_id) in state_df[['id', 'x', 'y', 'r', 'species']].values:
                s = next(
                    (s for s in self.species_list if s.species_id == species_id), None)
                if s is None:
                    raise ValueError(
                        f'Simulation.set_state(): Species with id {species_id} not found in species_list.')
                self.plants.append(s.create_plant(id=id, x=x, y=y, r=r))

            self.t = state_df['t'].values[-1]

        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)
        self.data_buffer.add(data=self.get_data())
        self.state_buffer.add(plants=self.plants, t=self.t)
        # self.density_field_buffer.add(
        #     field=self.density_field.values, t=self.t)
        
    def finalize(self):
        print(f'Simulation.finalize(): Time: {time.strftime("%H:%M:%S")}')
        self.data_buffer.finalize()
        self.state_buffer.finalize()
        # self.density_field_buffer.finalize()
        converted_dict = convert_dict(
            d=self.__dict__, conversion_factors=self.conversion_factors_default, reverse=True)
        save_dict(d=converted_dict,
                  path=f'{self.folder}/kwargs-{self.alias}', exclude=self.exclude_default)

    def convergence(self, trend_window=1500, trend_threshold=1e-1, t_min=0):
        # If there are not enough data points to calculate convergence, return False
        if self.t - t_min < 2:
            return False
        
        # Calculate the number of time steps in the window
        window = int(trend_window//self.time_step)
        
        # If the number of available data points is less than the window, return False
        if self.t - t_min < window:
            return False
        
        # Get the last 'window' data points
        data = self.data_buffer.get_data().tail(window)
        x, biomass, population = data['Time'].values, data['Biomass'].values, data['Population'].values
        
        # If the whole window has not yet passed t_min return False
        if len(x) < 1:
            return False
        if x[0] < t_min:
            return False
        
        # Calculate whether the windowed data is converged
        is_converged_biomass = convergence_check(x, biomass, trend_threshold=trend_threshold)
        is_converged_population = convergence_check(x, population, trend_threshold=trend_threshold)
        is_converged = is_converged_biomass and is_converged_population
        return is_converged
        
    # def remove_fraction(self, fraction):
    #     # Determine the indices of plants to remove
    #     total_plants = len(self.plants)
    #     num_to_remove = int(fraction * total_plants)
    #     indices_to_remove = set(np.random.choice(total_plants, num_to_remove, replace=False))

    #     # Use set difference to determine the indices to keep
    #     indices_to_keep = set(range(total_plants)) - indices_to_remove

    #     # Create a new PlantCollection with only the plants to keep
    #     plants_to_keep = [self.plants[i] for i in indices_to_keep]
    #     self.plants = PlantCollection(plants=plants_to_keep)

    #     # Update the KDTree and density field
    #     self.update_kdtree(self.plants)
    #     self.density_field.update(self.plants)
    
    def remove_fraction(self, fraction):
        # Determine the indices of plants to remove
        total_biomass = self.get_biomass()
        target_biomass = total_biomass * (1 - fraction)
        
        plants = self.plants.copy()
        while len(plants) > 0 and sum(plants.radii**2 * np.pi) > target_biomass:
            random_index = np.random.randint(0, len(plants))
            plants.pop(random_index)
            
        self.plants = PlantCollection(plants=plants)

        # Update the KDTree and density field
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
    
    def local_density(self, pos):
        return self.density_field.query(pos)
        
    def get_density(self):
        return np.sum(self.density_field.values)

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
        if self.boundary_condition.lower() == 'periodic':
            positions_new, index_pairs, was_shifted = positions_shift_periodic(box=self.box, positions=positions_to_germinate, duplicates=False)
                        
            positions_to_germinate = positions_new
            parent_species = parent_species[index_pairs[:, 0]]
            
            
            box_check = outside_box_check(box=self.box, positions=positions_to_germinate)
            # print(f'Simulation.attempt_germination(): {box_check.shape} positions outside the box.')
            # print(f'Simulation.attempt_germination(): {positions_to_germinate.shape} positions to germinate.')
            # print(f'Simulation.attempt_germination(): {parent_species.shape} parent species to germinate.')
            # if np.any(box_check):
            #     print(f'Simulation.attempt_germination(): {box_check.sum()} positions outside the box.')
            #     print(f'Simulation.attempt_germination(): {len(positions_to_germinate) = } positions to germinate.')
            positions_to_germinate = positions_to_germinate[~np.any(box_check, axis=1)]
            parent_species = parent_species[~np.any(box_check, axis=1)]
            
        elif self.boundary_condition.lower() == 'box':
            box_check = np.any(outside_box_check(box=self.box, positions=positions_to_germinate), axis=1)
            positions_to_germinate = positions_to_germinate[~box_check]
            parent_species = parent_species[~box_check]
        else:
            raise ValueError(
                f'Simulation.attempt_germination(): Boundary condition "{self.boundary_condition}" not recognized.')
        
        species_germination_chances = np.array([s.germination_chance for s in parent_species])
        if self.density_scheme == 'local':
            germination_chances = np.maximum(self.land_quality, self.local_density(positions_to_germinate)) * self.precipitation * species_germination_chances
        elif self.density_scheme == 'global':
            germination_chances = np.maximum(self.land_quality, self.get_density()) * self.precipitation * species_germination_chances
        else:
            raise ValueError(
                f'Simulation.attempt_germination(): Density scheme "{self.density_scheme}" not recognized.')
        random_values = np.random.uniform(0, 1, len(positions_to_germinate))
        germination_indices = np.where(germination_chances > random_values)[0]

        new_plants = []
        for i in germination_indices:
            new_plant = parent_species[i].create_plant(
                id=self.id_generator.get_next_id(), 
                x=positions_to_germinate[i, 0], 
                y=positions_to_germinate[i, 1], 
                r=parent_species[i].r_min,
                is_dead=False
            )
            new_plants.append(new_plant)
                    
        if len(new_plants) > 0:
            self.add(new_plants)
            # print(f'Simulation.attempt_germination(): {len(new_plants)} plants germinated.')

    def get_data(self):
        biomass = self.plants.radii**2 * np.pi
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
    
    def get_biomass(self):
        return sum(self.plants.radii**2 * np.pi)
    
    def get_population(self):
        return len(self.plants)

    def initiate_uniform_radii(self, n, species_list=[], box=None):
        if len(self.plants) != 0:
            raise ValueError(
                "Simulation.initiate_uniform_radii(): The simulation is not empty. Please initialize an empty simulation.")

        if species_list == []:
            species_list = self.species_list

        if box is None:
            box = np.asarray(self.box, dtype=float)

        new_plants = []
        # COULD BE PARALLELIZED
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
        self.data_buffer.add(data=self.get_data())
        self.state_buffer.add(plants=self.plants, t=self.t)
        
    def spawn_non_overlapping(self, n=None, target_density=0.1, max_attempts=100_000, species_list=[], box=None, force=False, gaussian=False, verbose=True):
        if n is None and target_density is None:
            raise ValueError("Simulation.spawn_non_overlapping(): Either n or target_density must be specified.")
        if n is None:
            n = np.inf
        if target_density is None:
            target_density = np.inf
        if target_density > 1:
            raise ValueError("Simulation.spawn_non_overlapping(): target_density must be within range(-inf, 1].)")
            
        if len(self.plants) > 0 and not force:
            print(f"Simulation.spawn_non_overlapping(): {len(self.plants) = }")
            print(f"Simulation.spawn_non_overlapping(): {force = }")
            print(f"Simulation.spawn_non_overlapping(): {len(self.plants) > 0 and not force =}")
            warnings.warn("Simulation.spawn_non_overlapping(): The simulation is not empty. Do you want to continue? (Y/n):")
            user_input = input()
            if user_input.lower() != 'y':
                raise ValueError("Simulation.spawn_non_overlapping(): Aborted by user.")
        if species_list == []:
            species_list = self.species_list
        L = self.L
        new_plants = PlantCollection()
        attempts = 0
        if max_attempts is None:
            max_attempts = 200 * n if n is not None else 50 * L
        species_counts = {s.species_id: 0 for s in species_list}
        weights = np.ones(len(species_list)) / len(species_list)
        
        if box is None:
            box = self.box
        else:
            box = np.array(box)
            if box.shape != (2, 2):
                raise ValueError("Simulation.spawn_non_overlapping(): Box must be a 2x2 array.")
            if box[0, 0] >= box[0, 1] or box[1, 0] >= box[1, 1]:
                raise ValueError("Simulation.spawn_non_overlapping(): Box coordinates are invalid.")
        
        try:
            stop_condition = False
            biomass = 0
            while stop_condition is False:
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
                if gaussian:
                    center_of_box  = np.array([box[0, 0] + (box[0, 1] - box[0, 0]) / 2, box[1, 0] + (box[1, 1] - box[1, 0]) / 2])
                    scale = np.random.uniform((box[0, 1] - box[0, 0])* 0.01,
                                              (box[0, 1] - box[0, 0]))
                    new_x = np.random.normal(loc=center_of_box[0], scale=scale)
                    new_y = np.random.normal(loc=center_of_box[1], scale=scale)
                else:
                    new_x = np.random.uniform(*box[0])
                    new_y = np.random.uniform(*box[1])
                new_pos = np.array([new_x, new_y])


                attempts += 1
                if all(np.linalg.norm(new_pos - plant.pos()) >= new_r + plant.r for plant in new_plants):
                    new_plants.append(current_species.create_plant(
                    id=len(new_plants), x=new_x,  y=new_y, r=new_r))
                    species_counts[current_species.species_id] += 1
                    
                    
                    # biomass = sum([np.pi * plant.r**2 for plant in new_plants])
                    box = np.array(box)
                    biomass_in_box = sum([np.pi * plant.r**2 for plant in new_plants if np.all(np.logical_and(plant.pos() >= box[:, 0], plant.pos() <= box[:, 1]))])
                    box_area = (box[0, 1] - box[0, 0]) * (box[1, 1] - box[1, 0])
                    if box_area <= 0:
                        raise ValueError("Simulation.spawn_non_overlapping(): Box area is zero or negative.")
                    
                    density = biomass_in_box / box_area
                    # print()
                    # print(f"Simulation.spawn_non_overlapping(): {biomass_in_box =}")
                    # print(f"Simulation.spawn_non_overlapping(): {box_area =}")
                    # print(f"Simulation.spawn_non_overlapping(): {density =}")
                    # print(f"Simulation.spawn_non_overlapping(): {target_density =}")
                    # print(f"Simulation.spawn_non_overlapping(): {len(new_plants) = }")
                    # print(f"Simulation.spawn_non_overlapping(): {len(new_plants) >= n =}")
                    # print(f"Simulation.spawn_non_overlapping(): {density >= target_density =}")
                    # print(f"Simulation.spawn_non_overlapping(): {attempts >= max_attempts =}")
                    # print(f"Simulation.spawn_non_overlapping(): {stop_condition =}")

                    if verbose:
                        print(f'Simulation.spawn_non_overlapping(): {len(new_plants) = :>5}/{n}   {density = :.3f}/{target_density:.3f}   {attempts = :>6}/{max_attempts}', end='\r')
                    # print(
                    # f'Simulation.spawn_non_overlapping(): {len(new_plants) = :>5}/{n}   {density = :.3f}/{target_density:.3f}   {attempts = :>6}/{max_attempts}', end='\r')
                        
                
                stop_condition = (len(new_plants) >= n) or (density >= target_density) or (attempts >= max_attempts)
        except KeyboardInterrupt:
            print('\nInterrupted by user...')
        
        if verbose:
            print(f'Simulation.spawn_non_overlapping(): {len(new_plants) = :>5}/{n}   {density = :.3f}/{target_density:.3f}   {attempts = :>6}/{max_attempts}', end=' ' * 20 + '\r')

        self.add(new_plants)
        self.update_kdtree(self.plants)
        self.density_field.update(self.plants)
        self.data_buffer.add(data=self.get_data())
        self.state_buffer.add(plants=self.plants, t=self.t)
        
        return new_plants, density, attempts

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
        'size_buffer',
        'species_list',
        'state_buffer',
        'state',
        'convert_dict',
        'verbose',
        '_m'
    ]

    def print(self, exclude=None):
        print(self.__str__(), end='\r')
    
    def __str__(self):
        if self.is_running:
            runtime_str = time.strftime("%H:%M:%S")
            dots = ['.  ', '.. ', '...'][int(self.t % 3)]
            data = self.get_data()
            t, biomass, population, precipitation = data[[
                'Time', 'Biomass', 'Population', 'Precipitation']].values.reshape(-1)
            t = float(round(t, 2))

            s = f'{dots} Time: {runtime_str}' + ' '*5 + f'|  {t=:^8}  |  N = {population:<6}  |  B = {np.round(biomass, 4):<6}  |  P = {np.round(precipitation, 6):<8}'
        else: 
            s = f'Simulation {self.alias} is not running.'
            s += '\n'
        return s
        

    def get_kwargs(self, exclude=None):
        exclude = exclude or self.exclude_default
        exclude = exclude if isinstance(exclude, list) else [exclude]
        
        kwargs = copy.deepcopy(self.__dict__)
        for k in exclude:
            if k in kwargs:
                kwargs.pop(k)
        # Convert the kwargs to the correct units
        kwargs = convert_dict(
            d=kwargs, conversion_factors=self.conversion_factors_default, reverse=True)
        
        for species in self.species_list:
            species_title = species.name
            kwargs[species_title] = copy.deepcopy(convert_dict(species.__dict__, conversion_factors=self.conversion_factors_default, reverse=True))
            kwargs[species_title].pop('name')        
        return kwargs

    def plot_buffers(self, title='', fast=False, data=True, plants=True, density_field=False, n_plots=20, dpi=300, save=False, folder=None):
        n = sum([density_field, plants])
        if n == 0:
            raise ValueError("Simulation.plot(): No plots requested. Set density_field or plants to True.")
        figs = []
        axs = []
        titles = []
        if data and self.data_buffer is not None:
            fig, axs_db, title_db = DataBuffer.plot(
                data=self.data_buffer.get_data(),
                title=title,
                dict_to_print=self.get_kwargs()
                )
            figs.append(fig)
            axs.append([ax for ax in axs_db])
            titles.append(title_db)
        if plants and self.state_buffer is not None:
            fig, ax, title_sb = StateBuffer.plot(
            data=self.state_buffer.get_data(),
            title=title,
            n_plots=n_plots,
            box=self.box,
            boundary_condition=self.boundary_condition,
            fast=fast
            )
            figs.append(fig)
            axs.append(ax)
            titles.append(title_sb)
        if density_field and self.density_field_buffer is not None:
            fig, ax, title_fb = FieldBuffer.plot(
                data=self.density_field_buffer.get_data(),
                title=title,
                n_plots=n_plots,
                box=self.box,
                boundary_condition=self.boundary_condition,
                density_scheme=self.density_scheme,
                )
            figs.append(fig)
            axs.append(ax)
            titles.append(title_fb)   
            
        if save:
            folder = folder or self.folder
            os.makedirs(folder + '/figures', exist_ok=True)
            for i, (fig, title) in enumerate(zip(figs, titles)):
                title = title.replace(' ', '-')
                fig.savefig(f'{folder}/figures/{title}.png', dpi=dpi)
        return figs, axs, titles

    def plot(self, title='', fast=False, data=False, plants=True, density_field=True, plot_dead=False):
        state = pd.DataFrame([[p.id, p.x, p.y, p.r, p.species_id, self.t, p.is_dead]
                             for p in self.plants], columns=['id', 'x', 'y', 'r', 'species', 't', 'is_dead'])
        
        n = sum([plants, density_field])
        m = sum([data])
        if n == 0:
            raise ValueError("Simulation.plot(): No plots requested. Set density_field or plants to True.")
        
        fig, axs = plt.subplots(1, n, figsize=(16, 6))
        if plants:
            StateBuffer.plot_state(size=6, state=state, title=title, box=self.box, 
               boundary_condition=self.boundary_condition, fast=fast, 
               plot_dead=plot_dead, ax=axs[0])
        if density_field:
            DensityFieldCustom.plot(field=self.density_field.get_values(), ax=axs[1], title=title, fast=fast, box=self.box,
               boundary_condition=self.boundary_condition, plot_dead=plot_dead, competition_scheme=self.competition_scheme, density_scheme=self.density_scheme, t=self.t)
        if data:
            data_fig, data_axs, data_titles = DataBuffer.plot(data=self.data_buffer.get_data(), title=title, dict_to_print=self.get_kwargs())
            # for data_ax in data_axs:
            #     fig.add_subplot(data_ax)

        for ax in axs[1:]:
            ax.sharex(axs[0])
            ax.sharey(axs[0])
            

        return fig, axs, title
    
    def plot_plants(self, title='', fast=False, plot_dead=False):
        fig, ax = plt.subplots(figsize=(12, 6))
        state = pd.DataFrame([[p.id, p.x, p.y, p.r, p.species_id, self.t, p.is_dead]
                             for p in self.plants], columns=['id', 'x', 'y', 'r', 'species', 't', 'is_dead'])
        StateBuffer.plot_state(size=6, state=state, title=title, box=self.box,
                       boundary_condition=self.boundary_condition, fast=fast, ax=ax, plot_dead=plot_dead)
        return fig, ax, title
