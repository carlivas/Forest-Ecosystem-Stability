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
    'precipitation': 0.1,
    'seed': seed,
    'competition_scheme': 'all'
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'../../Data/linear_precipitation/L{kwargs['L']}/new'
alias = f'lin_prec_L{kwargs['L']}_{current_time}'
sim = Simulation(folder=folder, alias=alias, **kwargs)

sim.initiate_non_overlapping(target_density=0.5)

T = 25_000
dp = - kwargs['precipitation'] / T
sim.run(T=T, delta_p=dp)


figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    tilte = title.replace(' ', '-').lower()
    fig.savefig(f'{folder}/figures/{title}.png', dpi=600)
