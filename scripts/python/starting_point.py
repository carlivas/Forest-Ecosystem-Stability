import numpy as np
import matplotlib.pyplot as plt
import os

from mods.simulation import Simulation
from mods.utilities import *
from mods.fields import DensityFieldCustom, DensityFieldSPH

save_results = True
plot_results = True

### STARTING POINT ###
load_folder = 'D:/774322652_finished' # REMEBER TO SET THE FOLDER
load_alias = '774322652_1'
sim = Simulation(folder=load_folder, alias=load_alias)


### DESTINATION ###
sim.set_folder(folder='Data/half_precipitation', alias='half_precipitation_774322652_1', override=True)


sim.precipitation = sim.precipitation * 0.5
sim.density_range = 100 / sim.L
sim.density_check_radius = 100 / sim.L
sim.maturity_size = sim.r_min
sim.density_field = DensityFieldCustom(half_width=0.5, half_height=0.5, resolution=sim.density_field_resolution)
 
converted_dict = convert_dict(sim.__dict__, conversion_factors = sim.conversion_factors_default)
save_dict(
    path=f'{load_folder}/kwargs-{alias}'.json,
          dictionary=converted_dict, exclude=sim.exclude_default
          )
print_dict(converted_dict)
sim.run(T=10000, min_population=1)






# figs, axs, titles = sim.plot_buffers(title=surfix)
# os.makedirs(folder + '/figures', exist_ok=True)
# for i, (fig, title) in enumerate(zip(figs, titles)):
#     fig.savefig(f'{folder}/figures/_{title}.png', dpi=600)

# plt.show()
#### THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
