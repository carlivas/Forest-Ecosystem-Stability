import json
import time
import pandas as pd
import matplotlib.pyplot as plt

from mods.simulation import sim_from_data

save_results = True
plot_results = True
save_folder = f'Data/temp/kwarg_cleanup'

path_kwargs = 'Data/temp/kwarg_cleanup/kwargs_test.json'
path_state_buffer = 'Data/temp/kwarg_cleanup/state_buffer_test.csv'

with open(path_kwargs, 'r') as file:
    kwargs = json.load(file)

kwargs['precipitation'] = 5000e-5

state_buffer_data = pd.read_csv(path_state_buffer, header=0)
sim = sim_from_data(state_buffer_data=state_buffer_data, kwargs=kwargs)

print(f'\nSimulation Loaded. Time: {time.strftime("%H:%M:%S")}')

sim.print_dict()
sim.run(T=1000)

if save_results:
    surfix = 'test2'
    sim.save_dict(path=f'{save_folder}/kwargs_{surfix}')
    sim.state_buffer.save(f'{save_folder}/state_buffer_{surfix}')
    sim.data_buffer.save(f'{save_folder}/data_buffer_{surfix}')
    sim.density_field_buffer.save(f'{save_folder}/density_field_buffer_{surfix}')

    print('Data saved in folder:', save_folder)

if plot_results:
    print('Plotting...')
    L = kwargs['L']
    N0 = len(sim.state_buffer.states[0])
    title = f'{L =}, $N_0$ = {N0}'
    sim.state_buffer.plot(title=f'{title}')
    sim.density_field_buffer.plot(title=f'{title}')
    sim.data_buffer.plot(title=f'{title}')
    plt.show()
