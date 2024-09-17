
# load libraries and define parameters
import numpy as np
import pickle
import time
import subprocess

from plant import Plant
from simulation import Simulation
import quadT

for r_chance in [4e-1, 6e-1, 8e-1, 10e-1, 12e-1, 14e-1, 16e-1, 18e-1, 20e-1]:
    print('\nRunning simulation...')
    # filename = f'Data\states_temp.pkl'
    filename = f'Data\states_10000_rep_p{r_chance}.pkl'
    save_file = True
    render = False

    np.random.seed(0)

    num_plants = 10000
    n_iter = 25000

    plant_kwargs = {'r_start': 1/num_plants,
                    'r_max': 5/num_plants,
                    'growth_rate': 0.01/num_plants,
                    'reproduction_chance': r_chance}
    simulation_kwargs = {'r_max_global': plant_kwargs['r_max']}

    # Initialize simulation
    center = (0, 0)
    half_width = half_height = 0.5
    simulation = Simulation(
        quadT.QuadTree(center, half_width, half_height, capacity=4), **simulation_kwargs)

    for i in range(num_plants):
        rand_pos = np.random.uniform(-0.5, 0.5, 2)
        this_plant_kwargs = plant_kwargs.copy()
        this_plant_kwargs['r_start'] = np.random.uniform(
            plant_kwargs['r_start'], plant_kwargs['r_start']*4)
        plant = Plant(rand_pos, **this_plant_kwargs)
        simulation.add_plant(plant)

        print(f'Planted {i+1}/{num_plants} plants', end='\r')

    def save_simulation(simulation, filename):
        with open(filename, 'wb') as f:
            pickle.dump(simulation.states, f)
        print('Saved simulation states' + ' '*100)

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
        save_simulation(simulation, filename=filename)
    if render:
        print('Rendering results...')
        # Run the rendering.py file
        subprocess.run(['python', 'render.py'])
