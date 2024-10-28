import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from plant import Plant
from simulation import Simulation
import json
import os

# Save the data in a folder with the current time as the name
save_folder = f'Data/{time.strftime("%Y%m%d_%H%M%S")}/'

num_plants = 1_00
n_iter = 10_0
m2pp = m2_per_plant = 10_000  # m2/plant


half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height
_m = np.sqrt(A_bound/(m2pp*num_plants))

sgcs = [0.05, 0.06, 0.07, 0.08][::-1]
lqs = [0.0, 0.0001, 0.001][::-1]

k = 0
for sgc in sgcs:
    for lq in lqs:
        seed = np.random.randint(0, 1_000)
        np.random.seed(seed)

        # Initialize simulation
        plant_kwargs = {
            'r_min': 0.1 * _m,
            'r_max': 30 * _m,
            'growth_rate': 0.1 * _m,
            'dispersal_range': 100 * _m,
            'species_germination_chance': sgc,
        }
        sim_kwargs = {
            'seed': seed,
            'n_iter': n_iter,
            'half_width': half_width,
            'half_height': half_height,

            'm2_per_plant': m2_per_plant,
            '_m': _m,
            'num_plants': num_plants,
            'land_quality': lq,

            'kt_leafsize': 10,

            'density_check_radius': 100 * _m,
            'density_field_resolution': 50,

            'density_field_buffer_size': 15,
            'density_field_buffer_skip': np.ceil(n_iter/15).astype(int),
            'density_field_buffer_preset_times': [],

            'state_buffer_size': 15,
            'state_buffer_skip': np.ceil(n_iter/15).astype(int),
            'state_buffer_preset_times': [],
        }

        print(f'{seed = }')
        print(f'1 m = {_m} u')
        print(f'1 u = {1/_m} m')
        print(
            f'species_germination_chance = {plant_kwargs["species_germination_chance"]}')
        print(f'land_quality = {sim_kwargs["land_quality"]}')

        sim = Simulation(**sim_kwargs)

        sim.initiate_uniform_lifetimes(
            n=num_plants, t_min=1, t_max=300, **plant_kwargs)

        sim.run(n_iter)

        sim.data_buffer.finalize()

        print(f'{seed = }')
        print(f'1 m = {_m} u')
        print(f'1 u = {1/_m} m')
        print(
            f'species_germination_chance = {plant_kwargs["species_germination_chance"]}')
        print(f'land_quality = {sim_kwargs["land_quality"]}')
        print('\nSimulation over.' + ' '*20)

        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj

        def save_kwargs(kwargs, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + '.json', 'w') as f:
                serializable_kwargs = convert_to_serializable(kwargs)
                json.dump(serializable_kwargs, f, indent=4)

        # def save_kwargs(kwargs, path):
        #     os.makedirs(os.path.dirname(path), exist_ok=True)
        #     with open(path + '.json', 'w') as f:
        #         json.dump(kwargs, f, indent=4)

        combined_kwargs = {
            'plant_kwargs': plant_kwargs,
            'sim_kwargs': sim_kwargs
        }

        surfix = f'{k}'

        save_kwargs(combined_kwargs, f'{save_folder}/kwargs_{surfix}')
        sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
        sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
        sim.density_field_buffer.save(
            f'{save_folder}/density_field_buffer_{surfix}')

        print('Data saved.')
        k += 1
