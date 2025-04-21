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
    'precipitation': 0.0,
    'seed': seed,
    'land_quality': 0.0,
    'boundary_condition': 'periodic',
    
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/debugging/collisions'
alias = f'250403_210300'

species = PlantSpecies()
species.growth_rate = 0.0001 * kwargs['L']
os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, **kwargs, override=True, species_list=[species])

sim.spawn_non_overlapping(n = 20, box=np.array([[-0.5, 0.5], [0.2, 0.5]]))
sim.spawn_non_overlapping(n = 20, box=np.array([[-0.5, 0.5], [-0.2, -0.5]]))


plants = []
s = species.growth_rate
for i in range(1, 30):
    r = 10*i*s
    x = sim.box[0,0] + r + 2*sum(plant.r for plant in plants) + 1.5*s*i**2
    y = 0
    plant = species.create_plant(id=i, x=x, y=y, r=r)
    print(f'{i=:.6f}, {r=:.6f}, {x=:.6f}, {y=:.6f}')
    plants.append(plant)
sim.add(plants)
sim.update_kdtree(sim.plants)
sim.data_buffer.add(data=sim.collect_data())
sim.state_buffer.add(plants=sim.plants, t=sim.t)
# sim.plot_state()

species.r_max = 10000000
for p in sim.plants:
    p.r_max = 10000000
for i in range(100):
    sim.density_field.values = sim.density_field.values*0
    sim.step()
    print(f'\nDone with iteration {sim.t}')
anim, title = StateBuffer.animate(sim.state_buffer.buffer, box=sim.box, boundary_condition=sim.boundary_condition, interval=500)
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