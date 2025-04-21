import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime

save_figs = True

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 2000,
    'precipitation': 0.0675,
    'density_initial': 0.05,
    'seed': seed,
    'boundary_condition': 'periodic',
    'competition_scheme': 'all',
    'density_scheme': 'global',
}
folder = f'Data/baseline/L{kwargs["L"]}'
alias = generate_alias(id='baseline_global', keys=['L', 'precipitation', 'density_initial'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)
sim.spawn_non_overlapping(target_density=kwargs['density_initial'])
sim.run(T=2500)


figs, axs, titles = sim.plot_buffers(title=alias, save=True, dpi=300)
plt.show()


# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])