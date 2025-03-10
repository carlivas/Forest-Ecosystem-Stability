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
    'L': 500,
    'precipitation': 1,
    'seed': seed,
    'spawn_rate': 1,
}
num_plants = int(kwargs['L'])

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
current_time = 'test2'
folder = f'Data/species_experiments/experiment4/tests/{current_time}'
alias = f'exp4_{current_time}'
print(f'{seed = }')

os.makedirs(folder, exist_ok=True)

species1 = PlantSpecies(r_min=15,
                        r_max=30,
                        growth_rate=0.1,
                        dispersal_range=90,
                        density_range=90,
                        germination_chance=1,
                        species_id=1,
                        name='Large, fast lived')

species2 = PlantSpecies(r_min=1,
                        r_max=15,
                        growth_rate=0.1,
                        dispersal_range=10,
                        density_range=10,
                        germination_chance=1,
                        species_id=2,
                        name='Small, fast lived')

species_list = [species1, species2]

sim = Simulation(folder=folder, alias=alias,
                 species_list=species_list, **kwargs, override=True)

print(f'num_plants: {num_plants}')
sim.initiate_non_overlapping(n=num_plants, species_list=sim.species_list, max_attempts=50*num_plants)

T = 10_000
dp = - kwargs['precipitation'] / (T - 1000)
sim.run(T=T, min_population=1, delta_p=dp)


figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    tilte = title.replace(' ', '-').lower()
    fig.savefig(f'{folder}/figures/{title}.png', dpi=1000)

anim, _ = StateBuffer.animate(
    sim.state_buffer.get_data(), skip=10, title=alias, fast=True)
anim.save(f'{folder}/figures/state_anim-{alias}.mp4', writer='ffmpeg', dpi=600)

# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
