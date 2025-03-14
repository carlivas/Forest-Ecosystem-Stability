import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 2000,
    'precipitation': 0.5,
    'seed': seed,
}
num_plants = int(kwargs['L'])

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'../../Data/dynamics/recovery_rates'

alias = f'rec_rate_{current_time}'
print(f'{seed = }')

os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, **kwargs)

print(f'num_plants: {num_plants}')
sim.initiate_non_overlapping(n=num_plants, species_list=sim.species_list, max_attempts=50*num_plants)

n_steps = 10
T = 2000
precipitation_step = - kwargs['precipitation'] / (n_steps - 1)
print(f'{precipitation_step = }')
for i in range(n_steps):
    print(f'Running step {i+1}/{n_steps}')
    sim.run(T=T)
    
    # Remove a fraction of the plants
    sim.remove_fraction(0.15)
    
    sim.run(T=T)
    
    sim.precipitation = sim.precipitation + precipitation_step
    


figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    title = title.replace(' ', '-').lower()
    fig.savefig(f'{folder}/figures/{title}.png', dpi=1000)

# anim, _ = StateBuffer.animate(
#     sim.state_buffer.get_data(), skip=10, title=alias, fast=True)
# anim.save(f'{folder}/figures/state_anim-{alias}.mp4', writer='ffmpeg', dpi=600)

# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
