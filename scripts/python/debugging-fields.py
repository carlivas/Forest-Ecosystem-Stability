import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime
from matplotlib.gridspec import GridSpec

# seed = np.random.randint(0, 1_000_000_000)
seed = 2
kwargs = {
    'L': 4000,
    'precipitation': 0.08,
    'seed': seed,
    'boundary_condition': 'periodic',
    'box': [[-0.5, 0.5], [-0.5, 0.5]],
    'density_field_resolution': 30,
    'density_scheme': 'global',
    'field_buffer_skip': 1
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/debugging'
alias = f'temp'

sim = Simulation(folder=folder, alias=alias, **kwargs, override=True, force=True)

stop_condition = False
attempts = 0
while not stop_condition:
    (x, y) = np.random.uniform(np.array([-0.5, -0.5]), np.array([0.5, 0.5]), 2)
    L = sim.L
    s = np.random.uniform(50, 600)/2/sim.L # scale
    box = np.array([[x-s, x+s], [y-s, y+s]])
    d = np.random.uniform(0.01, 0.3)
    N, D, A = sim.spawn_non_overlapping(target_density=d, box=box, force=True, gaussian=True, verbose=False)
    print(f'{sim.get_population() = :>5}, {sim.get_biomass() = :.5f}, {attempts = :>6f}' + ' ' * 20, end='\r')
    attempts += A
    stop_condition = (sim.get_biomass() > 0.2 or sim.get_population() > 5000 or attempts < 100_000)
print(f'\n{sim.get_population() = :>5}, {sim.get_biomass() = :.5f}, {attempts = :>6f}' + ' ' * 20)
run = None
for n in range(1):
    run = sim.run(T=2_000, dp=-0.1/2_000)
    sim.plot(data=True, plants=True, density_field=True, title=alias)
    plt.show()

sim.plot_buffers(data=False, plants=True, density_field=True, title=alias, n_plots=5)
plt.show()
    
# plant_ids = np.unique([plant.id for plant in sim.plants])
# print(f'Number of unique plant ids: {len(plant_ids)}')
# sim.step()
# for i in range(10):
#     print(f'\nIteration {i}')
#     plant_ids_old = plant_ids
#     sim.step()
#     plant_ids = np.unique([plant.id for plant in sim.plants])

#     dead_plant_ids = np.setdiff1d(plant_ids_old, plant_ids)
#     new_plant_ids = np.setdiff1d(plant_ids, plant_ids_old)
#     print(f'Change due to new plants: {len(new_plant_ids)}')
#     print(f'Change due to dead plants: {-len(dead_plant_ids)}')
#     print(f'Total change in plant ids: {len(new_plant_ids) - len(dead_plant_ids)}')
#     print(f'Number of unique plant ids: {len(plant_ids)}')

# figs, ax_list, titles = sim.plot_buffers(title=alias)
# plt.show()