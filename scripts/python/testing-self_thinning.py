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
    'L': 1000,
    'precipitation': 0.0,
    'density_initial': 0.4/15,
    'seed': seed,
    'boundary_condition': 'periodic',
    'competition_scheme': 'all',
    'density_scheme': 'global',
    'time_step': 2/365,
}
folder = f'Data/self_thinning'
alias = generate_alias(id='baseline2', keys=['L', 'precipitation', 'density_initial'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)
s = sim.species_list[0]
s.growth_rate = 0.012
s.r_max = 10/sim.L
sim.spawn_non_overlapping(target_density=kwargs['density_initial'],
                          r_min=s.r_min,
                          r_max=s.r_min
                          )
sim.run(T=400)


figs, axs, titles = sim.plot_buffers()

db = sim.data_buffer.get_data()
db = db[db['Population'] > 0]
db = db[db['Biomass'] > 0]
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

norm = plt.Normalize(vmin=0, vmax=db['Time'].max())
cmap = plt.get_cmap('viridis')
scatter = ax.scatter(db['Population'], db['Biomass'], c=db['Time'], cmap=cmap, norm=norm, alpha=0.5, label='Data')
cbar = fig.colorbar(scatter, ax=ax)
ax.set_xscale('log')
ax.set_yscale('log')
# plt.ylim(1e3, 2e3)
ax.set_aspect('equal')
ax.set_ylabel('Biomass')
ax.set_xlabel('Population')
ax.set_title('Self-thinning')
plt.show()