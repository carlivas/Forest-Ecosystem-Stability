import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pandas as pd
from scipy.optimize import curve_fit

from mods.plant import *
from mods.simulation import *
from mods.buffers import *
from mods.utilities import *

perturbation_fractions = [0.1, 0.15, 0.2, 0.25, 0.3]
n_experiments = 10

starting_point_folder = 'Data/recovery_rates_test'
# starting_point_alias = 'baseline1_L2e3_PR1e_1_DE3e_1_250427_131238_local'
starting_point_alias = 'baseline_global_L2e3_PR1e_1_DE3e_1_250421_122730'

temp_folder = 'Data/recovery_rates_test/temp'
os.makedirs(temp_folder, exist_ok=True)

for i in range(n_experiments):
    print('\n\n')
    print('---------------------------------')
    print(f'Experiment {i+1}/{n_experiments}')

    starting_point_sim = Simulation(
        folder=starting_point_folder, alias=starting_point_alias)
    data = starting_point_sim.data_buffer.get_data()
    seed = np.random.randint(0, 2**32, dtype=np.uint32)
    np.random.seed(seed)
    
    random_start_time = np.random.choice(data['Time'].iloc[-3000:])
    var_biomass = data['Biomass'].iloc[3000:].var()
    mean_biomass = data['Biomass'].iloc[3000:].mean()
    target_biomass = mean_biomass
        
    starting_state = starting_point_sim.state_buffer.get_specific_data(
        t=random_start_time)

    save_alias = starting_point_alias.replace('baseline', 'recovery_rates') + f'_{seed}'
    sim = starting_point_sim.set_folder(
        folder=temp_folder, alias=save_alias, force=True)
    sim.set_state(starting_state)
    sim.set_seed(seed)
    
    start_time = sim.t
    biomass_pre_perturbation = sim.get_biomass()
    print()
    print(f'{start_time=:.6f}')
    print(f'{biomass_pre_perturbation=:.6f}')
    
    perturbation_fraction = np.random.choice(perturbation_fractions)
    sim.remove_fraction(perturbation_fraction)
    biomass_post_perturbation = sim.get_biomass()
    print(f'{biomass_post_perturbation=:.6f}')
    print(f'{perturbation_fraction=:.6f}')
    print(f'{target_biomass=:.6f}')
    print()

    sim.run(T=1000, min_population=1, convergence_stop=50)
    
    did_recover = sim.get_biomass() >= target_biomass
    params = [np.nan, np.nan, np.nan]
    if did_recover:
        # Define the exponential function
        def exponential(x, a, b, c):
            return a * np.exp(b * x) + c

        # Fit the data
        data = sim.data_buffer.get_data()
        t_data = data['Time'].iloc[1:].values
        y_data = data['Biomass'].iloc[1:].values # Normalize the data
        t_data = t_data  # Normalize time
        try:
            params, covariance = curve_fit(
                exponential, 
                t_data - t_data[0], 
                y_data, 
                p0=(y_data[0], -0.01, y_data[-1]), 
                bounds=([-np.inf, -np.inf, -np.inf], [np.inf, 0, np.inf]),  # Add bounds to constrain the fit
                maxfev=10000
            )
        except RuntimeError:
            print("Error: Unable to fit the data even with bounds.")


    ### RECOVERY RATE IS DEFINED AS λ in the exponential growth model:
    ### Biomass(t) = a * exp(- λ * t) + c
    recovery_rate = -params[1]
    
    print(f'{did_recover=}, {recovery_rate=}')
    
    if did_recover:
        fig, ax = plt.subplots(figsize=(10, 5))
        data = sim.data_buffer.get_data()
        ax.plot(data['Time'], data['Biomass'], label='Biomass')
        exp_fit = exponential(data['Time'] - data['Time'].iloc[0], *params)
        ax.plot(data['Time'], exp_fit, label='Exponential Fit', linestyle='--')

    file_path = f'Data/recovery_rates_test/data/recovery_rates.csv'

    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=[
            'recovery_rate',
            'precipitation',
            'perturbation_fraction',
            'biomass_pre_perturbation',
            'start_time',
            'L',
            'density_scheme',
            'boundary_condition',
            'alias',
        ])
        df.to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    new_row = pd.DataFrame([{
        'recovery_rate': f'{recovery_rate:.6f}',
        'precipitation': f'{sim.precipitation:.6f}',
        'biomass_pre_perturbation': f'{biomass_pre_perturbation:.6f}',
        'perturbation_fraction': f'{perturbation_fraction:.6f}',
        'start_time': f'{start_time:.6f}',
        'L': f'{sim.L:.6f}',
        'density_scheme': sim.density_scheme,
        'boundary_condition': sim.boundary_condition,
        'seed': seed,
        'alias': starting_point_alias,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(file_path, index=False)


plt.show()
