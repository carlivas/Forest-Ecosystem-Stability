import json
import time
import pandas as pd
import matplotlib.pyplot as plt

from mods.simulation import save_simulation_results, plot_simulation_results, load_sim_data, sim_from_data

save_results = True
plot_results = True
folder = 'Data/temp/load_save_test'
surfix = '247088000'

for i in range(2):
    sim_data = load_sim_data(folder, surfix, state_buffer=True)
    sim = sim_from_data(sim_data, times_to_load='last')
    print(f'\nSimulation Loaded. Time: {time.strftime("%H:%M:%S")}')
    
    # sim.precipitation = sim.precipitation - 0.0001
    sim.run(T=10, transient_period=1000)
    
    if len(sim.state) == 0:
        print('No plants left. Exiting...')
        break

    surfix = f'{surfix.split("_")[0]}_{i+1}'
    if save_results:
        save_simulation_results(sim, folder, surfix)
        print('Data saved in folder:', folder)

    if plot_results:
        plot_simulation_results(sim, convergence=True)
        print('Plotting...')
        
