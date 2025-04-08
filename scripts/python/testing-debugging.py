import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

# seed = np.random.randint(0, 1_000_000_000)
seed = 2
kwargs = {
    'L': 2000,
    'precipitation': 1.0,
    'seed': seed,
    'boundary_condition': 'periodic',
    'competition_scheme': 'sparse',
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/debugging'
alias = f'temp'

sim = Simulation(folder=folder, alias=alias, **kwargs, override=True)
sim.species_list[0].growth_rate = sim.species_list[0].growth_rate*40

sim.initiate_non_overlapping(n=3000)
print('\nAFTER INITIATION:')
print(f'Number of alive plants: {np.sum(~sim.plants.is_dead)}')
print(f'Number of dead plants: {np.sum(sim.plants.is_dead)}')
print(f'Number of plants: {len(sim.plants)}')
print(f'Mean plant size: {np.mean(sim.plants.radii*sim.L)}')
sim.plot_state(title='Step 0: Initiated' + f' ({kwargs["competition_scheme"]})', fast=True)

sim.plants.grow()
print('\nAFTER GROWTH:')
print(f'Number of alive plants: {np.sum(~sim.plants.is_dead)}')
print(f'Number of dead plants: {np.sum(sim.plants.is_dead)}')
print(f'Number of plants: {len(sim.plants)}')
print(f'Mean plant size: {np.mean(sim.plants.radii*sim.L)}')
sim.plot_state(title='Step 1: Grown' + f' ({kwargs["competition_scheme"]})', fast=True)

sim.plants.mortality()
print('\nAFTER MORTALITY:')
print(f'Number of alive plants: {np.sum(~sim.plants.is_dead)}')
print(f'Number of dead plants: {np.sum(sim.plants.is_dead)}')
print(f'Number of plants: {len(sim.plants)}')
fig, ax, title = sim.plot_state(title='Step 2: Mortality' + f' ({kwargs["competition_scheme"]})', fast=True, plot_dead=True)
    

disperse_positions, parent_species = sim.plants.disperse(sim)
sim.attempt_germination(disperse_positions, parent_species)
sim.attempt_spawn(n=sim.spawn_rate)
print('\nAFTER SPAWNING:')
print(f'Number of alive plants: {np.sum(~sim.plants.is_dead)}')
print(f'Number of dead plants: {np.sum(sim.plants.is_dead)}')
print(f'Number of plants: {len(sim.plants)}')
sim.plot_state(title='Step 3: After germination and spawn' + f' ({kwargs["competition_scheme"]})', fast=True, plot_dead=True)

sim.resolve_collisions(positions=sim.plants.positions, radii=sim.plants.radii)
indices_dead = np.where(sim.plants.is_dead)[0]
print('\nAfter COLLISION:')
print(f'Number of alive plants: {np.sum(~sim.plants.is_dead)}')
print(f'Number of dead plants: {np.sum(sim.plants.is_dead)}')
print(f'Number of plants: {len(sim.plants)}')
print(f'Indices of dead plants: {indices_dead}')
fig, ax, title = sim.plot_state(title='Step 4: Collisions' + f' ({kwargs["competition_scheme"]})', fast=True, plot_dead=True)

sim.plants.remove_dead_plants()
print('\nAFTER DEAD PLANT REMOVAL:')
print(f'Number of alive plants: {np.sum(~sim.plants.is_dead)}')
print(f'Number of dead plants: {np.sum(sim.plants.is_dead)}')
print(f'Number of plants: {len(sim.plants)}')
sim.plot_state(title='Step 5: After removing dead plants' + f' ({kwargs["competition_scheme"]})', fast=True, plot_dead=True)
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