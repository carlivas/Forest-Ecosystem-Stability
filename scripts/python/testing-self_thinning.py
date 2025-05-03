import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

save_figs = True

seed = np.random.randint(2**32, dtype=np.uint32)
kwargs = {
    'L': 500,
    'precipitation': 0.0,
    'density_initial': 0.5,
    'seed': seed,
    'boundary_condition': 'periodic',
    'competition_scheme': 'all',
    'density_scheme': 'global',
}
folder = f'Data/self_thinning'
alias = generate_alias(id='baseline2', keys=['L', 'precipitation', 'density_initial'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)
s = sim.species_list[0]
sim.spawn_non_overlapping(target_density=kwargs['density_initial'],
                          r_min=s.r_min,
                          r_max=s.r_min
                          )
sim.run(T=400)


figs, axs, titles = sim.plot_buffers(title=alias, save=True, dpi=300)
plt.show()