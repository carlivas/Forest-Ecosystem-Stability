import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

seed = np.random.randint(0, 1_000_000_000)
kwargs = {
    'L': 500,
    'precipitation': 0.2,
    'seed': seed,
    'competition_scheme': 'all',
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/debugging'
alias = f'new_competition_L500_ALL3'


sim = Simulation(folder=folder, alias=alias, **kwargs, override=True)
sim.initiate_non_overlapping(target_density=0.3, box=np.array([[-0.5, 0.5], [-0.5, 0.5]]))

sim.run(T=3000)
sim.plot_buffers(title=alias)
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