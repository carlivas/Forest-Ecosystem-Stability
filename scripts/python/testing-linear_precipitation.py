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
    'density_scheme': 'global',
}

folder = f'../../Data/linear_precipitation/L{kwargs['L']}/new/'
alias = generate_alias(id=f'linprec_{kwargs["density_scheme"]}', keys=['L'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)

sim.spawn_non_overlapping(target_density=0.2)

T = 30_000
dp = - kwargs['precipitation'] / 25_000
sim.run(T=T, dp=dp)


figs, axs, titles = sim.plot_buffers(title=alias, save=True, dpi=300)
plt.show()