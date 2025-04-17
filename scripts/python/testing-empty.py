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
    'precipitation': 0.0,
    'seed': seed,
    'competition_scheme': 'all',
    'land_quality': 0.01,
    'spawn_rate': 1e-6,
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/empty_temp'
alias = generate_alias(id='empty_test', keys=['land_quality', 'spawn_rate'], abrevs=['LQ', 'SR'], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)

T = 100_000
dp = 1 / T
sim.run(T=T, delta_p=dp, max_population=10)
sim.run(T=1000, delta_p=0, convergence_stop=False)
sim.run(T=10000, delta_p=0, convergence_stop=True)

figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    title = title.replace(' ', '-')
    fig.savefig(f'{folder}/figures/{title}.png', dpi=600)
    
plt.show()