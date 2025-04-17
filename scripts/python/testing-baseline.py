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
    'L': 10000,
    'precipitation': 0.0675,
    'density_initial': 0.3,
    'seed': seed,
    'boundary_condition': 'periodic',
    'competition_scheme': 'all'
}
folder = f'../../Data/baseline/L{kwargs["L"]}'
alias = generate_alias(id='baseline', keys=['L', 'precipitation', 'density_initial'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)
sim.initiate_non_overlapping(target_density=kwargs['density_initial'])
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


# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])