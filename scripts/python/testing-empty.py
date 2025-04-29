import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

seed = np.random.randint(0, 2**32, dtype=np.uint32)
np.random.seed(seed)
kwargs = {
    'L': 2000,
    'precipitation': 0.0,
    'seed': seed,
    'density_scheme': 'global'
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'../../Data/empty'
alias = generate_alias(id='empty_global', keys=['L', ], abrevs=['L'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)

T = 100_000
sim.run(T=T, max_population=100)
sim.run(T=1000, convergence_stop=False)
sim.run(T=10000, convergence_stop=True)

figs, axs, titles = sim.plot_buffers(title=alias, save=True)
    
plt.show()