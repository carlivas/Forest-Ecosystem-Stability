
# load libraries and define parameters
import numpy as np
import pickle
import time
import subprocess
import sys
import noise


from plant import Plant
from simulation import Simulation
import quadT

# Increase the recursion limit
sys.setrecursionlimit(100000)


def save_states_in_chunks(filename, states, chunk_size):
    l = len(states)
    with open(filename, 'wb') as f:
        for i in range(0, len(states), chunk_size):
            pickle.dump(states[i:i + chunk_size], f)
            last_chunk_saved = min(i + chunk_size, l)
            print(
                f'Saved states {last_chunk_saved}/{l} ({last_chunk_saved * 100 // l} %) to {filename}', end='\r')


def save_metadata(metadata, filename):
    with open(filename, 'wb') as f:
        pickle.dump(metadata, f)
    print(f'\nSaved metadata to {filename}')


print('\nRunning simulation...')
surfix = 'temp'
file_path_states = f'Data\states_{surfix}.pkl'
file_path_metadata = f'Data\metadata_{surfix}.pkl'

save_file = True
render = True

seed = 0
np.random.seed(seed)

num_plants = 10000
n_iter = 100000

half_width = half_height = 0.5
A_bound = 2 * half_width * 2 * half_height  # ms2

m2pp = m2_per_plant = 1000  # m2/plant
_m = meter_to_dimless = np.sqrt(A_bound/(m2pp*num_plants))  # ms/m
print(f'1 m = {_m} units')
r_start = 0.01 * _m  # m
r_max = 30 * _m  # m
growth_rate = 0.1 * _m  # m horizontal growth per year
r_chance = 1  # probability of reproduction (besides site quality)
# Initialize simulation

plant_kwargs = {'r_start': r_start,
                'r_max': r_max,
                'growth_rate': growth_rate,
                'reproduction_chance': r_chance}
simulation_kwargs = {'r_max_global': plant_kwargs['r_max']}

metadata = {
    'num_plants': num_plants,
    'n_iter': n_iter,
    'seed': seed,
    'plant_kwargs': plant_kwargs,
    'simulation_kwargs': simulation_kwargs,
    'chunk_size': np.clip(n_iter//10, 1, 50)
}

simulation = Simulation(
    quadT.QuadTree((0, 0), half_width, half_height, capacity=4), **simulation_kwargs)

i = 0
while i < num_plants:
    rand_pos = np.random.uniform(-half_width, half_width, 2)
    this_plant_kwargs = plant_kwargs.copy()
    this_plant_kwargs['r_start'] = np.random.uniform(
        plant_kwargs['r_start']*0.8, plant_kwargs['r_start']*1.2)
    plant = Plant(rand_pos, **this_plant_kwargs)
    simulation.add_plant(plant)
    print(f'Planted {i+1}/{num_plants} plants', end='\r')
    i += 1


# Run simulation
try:
    start_time = time.time()
    for t in range(n_iter):
        simulation.step()

        l = len(simulation.qt.all_points())
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
    n_iter = len(simulation.states)
    chunk_size = np.clip(n_iter//10, 1, 50)
    save_states_in_chunks(file_path_states, simulation.states,
                          chunk_size=chunk_size)
    metadata['file_path_states'] = file_path_states
    metadata['n_iter'] = n_iter
    metadata['chunk_size'] = chunk_size
    save_metadata(metadata, file_path_metadata)
if render:
    print('Rendering results...')
    # Run the rendering.py file
    subprocess.run(['python', 'render_chunks.py'])

print('\nDone!')
