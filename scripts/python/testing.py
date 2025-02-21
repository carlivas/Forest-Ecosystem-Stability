import numpy as np
import matplotlib.pyplot as plt
import os

from mods.simulation import Simulation
from mods.buffers import StateBuffer

save_results = True
plot_results = True

T = 5_000

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 1000,
    'T': T,
    'precipitation': 2,
    'seed': seed,
    'maturity_size': 1,
    'r_min': 1,
    'r_max': 5,
    'growth_rate': 0.5,
    'dispersal_range': 10,
    'spawn_rate': 1,
    'time_step': 1,
}
num_plants = int(2/3 * kwargs['L'])



folder = f'Data/starting_point_parameter_shift/L1000_shifted'
alias = 'shifted-test-2'
dp = 0 #-kwargs['precipitation']*1/T

sim = Simulation(folder=folder, alias=alias, **kwargs)
sim.initiate_uniform_radii(n=num_plants, r_min=sim.r_min/sim._m, r_max=sim.r_max/sim._m)
sim.run(T=T, delta_p=dp)

# anim, _ = StateBuffer.animate(sim.state_buffer.get_data())
# anim.save(f'{folder}/figures/{alias}.mp4', writer='ffmpeg')

figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    fig.savefig(f'{folder}/figures/_{title}.png', dpi=600)

# #### THINK ABOUT THIS
# # field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
