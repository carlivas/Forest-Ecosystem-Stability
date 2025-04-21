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
    'competition_scheme': 'all',
    'density_scheme': 'global',
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/linear_precipitation/L{kwargs['L']}/'
alias = generate_alias(id='linprec_global', keys=['L', 'precipitation'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)

sim.spawn_non_overlapping(target_density=0.2)

T = 50_000
dp = - kwargs['precipitation'] / T
sim.run(T=T, dp=dp)


figs, axs, titles = sim.plot_buffers(title=alias, save=True, dpi=300)
plt.show()