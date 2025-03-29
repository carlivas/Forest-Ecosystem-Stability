import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *
from datetime import datetime
current_time = datetime.now().strftime("%y%m%d_%H%M%S")

save_figs = False

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 4000,
    'seed': seed,
    'precipitation': 0.3,
    'land_quality': 0.01,
}

folder = f'Data/empty'
alias = f'empty_test'


alias = alias.replace(' ', '_').replace('-', '_').replace('.', '_')
os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, **kwargs, override=False)
sim.__dict__.update(kwargs)

T = 300

sim.run(T=T)
    


# figs, axs, titles = sim.plot_buffers(title=alias)
figs, axs, titles = zip(DataBuffer.plot(sim.data_buffer.get_data()), sim.plot())
if save_figs:
    os.makedirs(folder + '/figures', exist_ok=True)
    for i, (fig, title) in enumerate(zip(figs, titles)):
        title = title.replace(' ', '-')
        fig.savefig(f'{folder}/figures/{title}.png', dpi=600)
else:
    plt.show()
# anim, _ = StateBuffer.animate(
#     sim.state_buffer.get_data(), skip=10, title=alias, fast=True)
# anim.save(f'{folder}/figures/state_anim-{alias}.mp4', writer='ffmpeg', dpi=600)

# THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
