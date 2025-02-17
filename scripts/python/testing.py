import numpy as np
import os

from mods.simulation import Simulation

save_results = True
plot_results = True

T = 100_000

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 4500,
    'T': T,
    'precipitation': 0.5,
    'seed': seed,
}
num_plants = int(2/3 * kwargs['L'])

surfix = 'linear'+str(seed)
folder = f'../../Data/linear_precipitation/L{kwargs['L']}'

dp = -kwargs['precipitation']*1/T

sim = Simulation(folder=folder, alias=surfix, **kwargs)
sim.initiate_uniform_radii(n=num_plants, r_min=sim.r_min/sim._m, r_max=sim.r_max/sim._m)
sim.run(T=T, delta_p = dp)

figs, axs, titles = sim.plot_buffers()
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    fig.savefig(f'{folder}/figures/_{title}.png', dpi=600)
# plt.show()

#### THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
