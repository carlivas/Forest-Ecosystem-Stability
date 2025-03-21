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
    'seed': seed,
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/linear_precipitation/L{kwargs["L"]}'
alias = f'lin_prec_L{kwargs["L"]}_temp'

os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, **kwargs, override=True)
sim.precipitation = 0.3

num_plants = int(kwargs['L'])
sim.initiate_non_overlapping(n=num_plants, species_list=sim.species_list, max_attempts=50*num_plants)

T = 31000
precipitation_step = - sim.precipitation / int(T - 1000)
print(f'{precipitation_step = }')

sim.run(T=T, delta_p=precipitation_step)
    


figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    title = title.replace(' ', '-').lower()
    fig.savefig(f'{folder}/figures/{title}.png', dpi=600)

# anim, _ = StateBuffer.animate(
#     sim.state_buffer.get_data(), skip=10, title=alias, fast=True)
# anim.save(f'{folder}/figures/state_anim-{alias}.mp4', writer='ffmpeg', dpi=600)

# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
