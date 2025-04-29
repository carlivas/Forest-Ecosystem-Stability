import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

save_figs = True

Ps = np.arange(0.1, 0.2001, 0.01)
Ds = [0.0, 0.3]
for p in Ps:
    for d in Ds:
        seed = np.random.randint(2**32, dtype=np.uint32)
        kwargs = {
            'L': 2000,
            'precipitation': p,
            'density_initial': d,
            'seed': seed,
            'boundary_condition': 'periodic',
            'competition_scheme': 'all',
            'density_scheme': 'local',
        }
        folder = f'../../Data/baseline/L{kwargs["L"]}'
        alias = generate_alias(id='baseline2', keys=['L', 'precipitation', 'density_initial'], time=True, **kwargs)
        sim = Simulation(folder=folder, alias=alias, **kwargs)
        sim.spawn_non_overlapping(target_density=kwargs['density_initial'])
        sim.run(T=15000)
        
        
        figs, axs, titles = sim.plot_buffers(title=alias, save=True, dpi=300)
        # plt.show()


# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])