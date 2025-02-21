import numpy as np
import matplotlib.pyplot as plt
import os

from mods.simulation import Simulation

save_results = True
plot_results = True

load_folder = f'Data/linear_precipitation/L1000'
save_folder = f'Data/starting_point_parameter_shift/L1000_shifted'
dp = 0 #-kwargs['precipitation']*1/T
alias = 'rmin2-linear454524143'
sim = Simulation(folder=load_folder, alias=alias, verbose=True)

new_alias = f'shifted-test'
sim.set_folder(save_folder, alias=new_alias)
# sim.run(T=T, delta_p = dp)

# figs, axs, titles = sim.plot_buffers(title=surfix)
# os.makedirs(folder + '/figures', exist_ok=True)
# for i, (fig, title) in enumerate(zip(figs, titles)):
#     fig.savefig(f'{folder}/figures/_{title}.png', dpi=600)

#### THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
