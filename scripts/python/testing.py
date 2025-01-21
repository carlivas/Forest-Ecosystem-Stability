import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_m2pp, _m_from_domain_sides, save_simulation_results, plot_simulation_results
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable
from mods.buffers import StateBuffer, DataBuffer, FieldBuffer, HistogramBuffer
import shutil
import os

save_results = True
plot_results = True

T = 100_000
num_plants = 3000

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 4500,
    'T': T,
    'precipitation': 6500e-5,
    'seed': seed,
}

surfix = str(seed)
folder = f'../../Data/starting_contenders'

sim = Simulation(folder=folder, alias=surfix, **kwargs)
sim.initiate_uniform_radii(n=num_plants, r_min=sim.r_min/sim._m, r_max=sim.r_max/sim._m)
sim.run(T=T)


figs, axs = sim.plot_buffers()
for i, fig in enumerate(figs):
    fig.savefig(f'{folder}/figures/fig_{seed}_{i}.png', dpi=600)
# plt.show()



#### THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])











# if plot_results:
#     sim.plot()

# sim.cleanup()
# del sim
