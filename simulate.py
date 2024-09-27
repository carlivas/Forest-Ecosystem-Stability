
# load libraries and define parameters
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import subprocess
import sys
import noise

from KDTree_plant import Plant
from KDTree_simulation import Simulation

# Increase the recursion limit
sys.setrecursionlimit(100000)


def save_states_in_chunks(filename, states, chunk_size):
    start_time = time.time()
    l = len(states)
    with open(filename, 'wb') as f:
        for i in range(0, len(states), chunk_size):
            pickle.dump(states[i:i + chunk_size], f)
            last_chunk_saved = min(i + chunk_size, l)
            elapsed_time = time.time() - start_time
            print(
                f'Saved states {last_chunk_saved}/{l} ({last_chunk_saved * 100 // l} %) to {filename}. Elapsed time {elapsed_time:.0f} s.', end='\r')


def save_metadata(metadata, filename):
    with open(filename, 'wb') as f:
        pickle.dump(metadata, f)
    print(f'\nSaved metadata to {filename}')


def simulate(num_plants, n_iter, seed, plant_kwargs, simulation_kwargs, save_file=None, file_prefix=None, render=False):

    if save_file is None:
        print('Please specify whether to save the results or not.')
        return

    if save_file is True and file_prefix is None:
        print('Please specify the file path to save the results.')
        return

    if save_file is True:
        file_path_states = f'Data/{file_prefix}_states.pkl'
        file_path_metadata = f'Data/{file_prefix}_metadata.pkl'
    else:
        file_path_states = None
        file_path_metadata = None

    metadata = {
        'num_plants': num_plants,
        'n_iter': n_iter,
        'seed': seed,
        'plant_kwargs': plant_kwargs,
        'simulation_kwargs': simulation_kwargs,
        'chunk_size': np.clip(n_iter//10, 1, 50),
        'file_path_states': file_path_states,
    }

    simulation = Simulation(**simulation_kwargs)
    half_width = simulation_kwargs.get('half_width', 0.5)
    half_height = simulation_kwargs.get('half_height', half_width)

    simulation.initialize_random(num_plants, **plant_kwargs)

    N_ITERATIONS_TRACK = 5.  # Number of iterations to track
    MAX_GROWTH_RATE_FRACTION = 5.  # Maximum allowed growth rate fraction

    # Initialize a list to track the number of plants over the last N_ITERATIONS_TRACK iterations
    plant_counts = [len(simulation.qt.all_points())]
    states = []
    # Run simulation
    print('\nRunning simulation...')
    try:
        start_time = time.time()
        for t in range(n_iter):
            simulation.step()
            states.append(simulation.get_state())

            l = len(simulation.qt.all_points())

            # Track the number of plants
            plant_counts.append(l)
            if len(plant_counts) > N_ITERATIONS_TRACK:
                plant_counts.pop(0)

            # Calculate the growth rate
            if len(plant_counts) == N_ITERATIONS_TRACK:
                growth_rate = (plant_counts[-1] -
                               plant_counts[0]) / plant_counts[0]
                if growth_rate > MAX_GROWTH_RATE_FRACTION:
                    print(
                        f"Emergency break: Growth rate ({growth_rate:.2f}) exceeded the allowed fraction ({MAX_GROWTH_RATE_FRACTION}).")
                    break

            elapsed_time = time.time() - start_time  # in seconds
            print(
                f'Iteration {t + 1 :^5}/{n_iter} ({(t + 1)*100//n_iter:^3}%). Plants left {l:^5}/{num_plants}. Elapsed time {elapsed_time:.0f} s.', end='\r')
            if l == 0:
                break

    except KeyboardInterrupt:
        print('\nSimulation interrupted.')

    print('\nSimulation finished.')
    if save_file:
        print('Saving results...')
        n_iter = len(states)
        chunk_size = np.clip(n_iter//10, 1, 50)
        save_states_in_chunks(file_path_states, states,
                              chunk_size=chunk_size)
        metadata['file_path_states'] = file_path_states
        metadata['n_iter'] = n_iter
        metadata['chunk_size'] = chunk_size
        save_metadata(metadata, file_path_metadata)
    if render:
        print('Rendering results...')
        # Run the rendering.py file
        subprocess.run(['python', 'render.py'])

    print('\nDone!')

    return states, metadata

# # Save the biomass to a pickle file
# biomass_file_path = f'Data/biomass_{seed}.pkl'
# with open(biomass_file_path, 'wb') as f:
#     pickle.dump(biomass, f)
# print(f'Saved biomass to {biomass_file_path}')
