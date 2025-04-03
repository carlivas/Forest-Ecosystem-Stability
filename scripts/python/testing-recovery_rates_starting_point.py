import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd

from mods.plant import *
from mods.simulation import *
from mods.buffers import *
from mods.utilities import *

perturbation_fractions = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
n_experiments = 100

for i in range(n_experiments):
    print('\n\n')
    print(f'Experiment {i+1}/{n_experiments}')
    print('---------------------------------')
    perturbation_fraction = np.random.choice(perturbation_fractions)
        
    starting_point_folder = 'D:/baseline/L2000'
    starting_point_alias = 'baseline_L2000_P4e_01_250328_122116'

    starting_point_sim = Simulation(
        folder=starting_point_folder, alias=starting_point_alias)

    kwargs_to_copy = [
        'L',
        'boundary_condition',
        'box',
        'conversion_factors_default',
        'density_field_resolution',
        'time_step',
        'precipitation',
        'land_quality',
        'spawn_rate',
        'species_list'
    ]

    sim_kwargs = {k: v for k, v in starting_point_sim.__dict__.items()
                if k in kwargs_to_copy}

    data = starting_point_sim.data_buffer.get_data()
    random_start_time = np.random.choice(data['Time'].iloc[-3000:])
    starting_state = starting_point_sim.state_buffer.get_specific_data(
        t=random_start_time)
    mean_population = data['Population'].iloc[3000:].mean()
    del starting_point_sim

    temp_folder = 'Data/temp'
    os.makedirs(temp_folder, exist_ok=True)

    save_alias = starting_point_alias.replace('baseline', 'recovery_rates')
    
    sim = Simulation(folder=temp_folder, alias=save_alias, state=starting_state,
                    override_force=True, **sim_kwargs, convert_kwargs=False)

    seed = np.random.randint(0, 1_000_000_000)
    np.random.seed(seed)
    sim.seed = seed

    start_time = sim.t
    population_pre_perturbation = len(sim.plants)
    print()
    print(f'{start_time = :.3f}')
    print(f'{population_pre_perturbation = :.3f}')
    print(f'{perturbation_fraction = :.3f}')
    print(f'{mean_population = :.3f}')
    print()
    
    sim.remove_fraction(perturbation_fraction)


    sim.run(T=5000, min_population=1, max_population=mean_population)
    did_recover = len(sim.plants) >= mean_population
    recovery_time = sim.t - start_time
    
    if not did_recover:
        recovery_time = np.nan

    file_path = f'Data/dynamics/recovery_rates/data/{save_alias}.csv'

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
                        'start_time', 'population_pre_perturbation', 'perturbation_fraction', 'recovery_time'])
        df.to_csv(file_path, index=False)

    if did_recover:
        df = pd.read_csv(file_path)
        new_row = pd.DataFrame([{
            'start_time': start_time,
            'population_pre_perturbation': population_pre_perturbation,
            'perturbation_fraction': perturbation_fraction,
            'recovery_time': recovery_time
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_path, index=False)

