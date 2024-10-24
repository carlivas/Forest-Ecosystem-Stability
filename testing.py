import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from plant import Plant
from simulation import Simulation
import json
import os

num_plants = 10_000
n_iter = 10_000
m2pp = m2_per_plant = 250  # m2/plant


half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height
_m = np.sqrt(A_bound/(m2pp*num_plants))

sgcs = [0.2]
lqs = [-0.1]

k = 10
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
            'num_plants': num_plants,
            'land_quality': lq,

            'kt_leafsize': 10,

            'density_check_radius': 100 * _m,
            'density_field_resolution': 50,

            'density_field_buffer_size': 20,
            'density_field_buffer_skip': 200,
            'density_field_buffer_preset_times': [0, 200, 400, 600, 800, 1000],

            'state_buffer_size': 20,
            'state_buffer_skip': 200,
            'state_buffer_preset_times': [0, 200, 400, 600, 800, 1000],
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

        def save_kwargs(kwargs, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + '.json', 'w') as f:
                json.dump(kwargs, f, indent=4)

        combined_kwargs = {
            'plant_kwargs': plant_kwargs,
            'sim_kwargs': sim_kwargs
        }

        save_folder = 'Data/temp'
        surfix = f'{k}'

        save_kwargs(combined_kwargs, f'{save_folder}/kwargs_{surfix}')
        sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
        sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
        sim.density_field_buffer.save(
            f'{save_folder}/density_field_buffer_{surfix}')

        print('Data saved.')
        k += 1
