import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

save_figs = False

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)

species = PlantSpecies()
species.maturity_size = 15

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/maturity_sizes'
alias = f'maturity_size{species.maturity_size}_{current_time}'

alias = alias.replace(' ', '_').replace('-', '_').replace('.', '_')
os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, species_list=[species], L=1000, seed=seed, boundary_condition='box')
sim.precipitation = 1

sim.initiate_non_overlapping(target_density=0.25)
sim.run(T=5000)

figs, axs, titles = sim.plot_buffers(title=alias)
if save_figs:
    os.makedirs(folder + '/figures', exist_ok=True)
    for i, (fig, title) in enumerate(zip(figs, titles)):
        title = title.replace(' ', '-')
        fig.savefig(f'{folder}/figures/{title}.png', dpi=600)
        plt.close(fig)
else:
    plt.show()

# anim, _ = StateBuffer.animate(
#     sim.state_buffer.get_data(), skip=10, title=alias, fast=True)
# anim.save(f'{folder}/figures/state_anim-{alias}.mp4', writer='ffmpeg', dpi=600)

# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
