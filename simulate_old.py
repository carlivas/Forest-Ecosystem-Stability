
# load libraries and define parameters
import numpy as np
import pickle
import time
import subprocess

from plant import Plant
from simulation import Simulation
from container_old import Container

save_file = True
render = True

np.random.seed(0)

num_plants = 1
n_iter = 1

simulation_kwargs = {'r_max_global': 0.1}
plant_kwargs = {'r_start': 0.001,
                'r_max': 0.05,
                'growth_rate': 0.0001,
                'reproduction_chance': 0.1}

# Initialize simulation
center = (0, 0)
width = height = 1
simulation = Simulation(
    Container(center, width, height), **simulation_kwargs)


def save_simulation(simulation, filename):
    with open(filename, 'wb') as f:
        pickle.dump(simulation.export_states(), f)
    print('Saved simulation states' + ' '*100)


for i in range(num_plants):
    rand_pos = np.random.rand(2)-0.5
    plant = Plant(rand_pos, **plant_kwargs)
    simulation.add_plant(plant)

    print(f'Planted {i+1}/{num_plants} plants', end='\r')



# Run simulation
try:
    # Record the start time
    start_time = time.time()
    for t in range(n_iter):
        # Run the simulation for one step
        simulation.step()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Estimate remaining time
        plants_killed = num_plants - len(simulation.plants)
        avg_time_per_iter = elapsed_time / (t + 1)
        remaining_time = avg_time_per_iter * (n_iter - t - 1)

        print(
            f'Iteration {t + 1 :^5}/{n_iter} ({(t + 1)*100//n_iter:^3}%). Plants left {len(simulation.plants):^5}/{num_plants}. Elapsed time: {elapsed_time:.0f} s. Estimated remaining time {remaining_time:.0f} s.' + ' '*100, end='\r')

        l = len(simulation.plants)
        if l < 1:
            print('\nNo more plants left' + ' '*100)
            break

except KeyboardInterrupt:
    print('\nSimulation interrupted.')


if save_file:
    print('Saving results...')
    save_simulation(simulation, filename='Data\simulation_states_temp.pkl')
if render:
    print('Rendering results...')
    # Run the rendering.py file
    subprocess.run(['python', 'simulation_renderer.py'])
