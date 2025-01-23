import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import time

from mods.plant import Plant
from mods.simulation import Simulation, save_simulation_results, plot_simulation_results
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable
from mods.buffers import StateBuffer, DataBuffer, FieldBuffer, HistogramBuffer

save_results = True
plot_results = True

T = 100
num_plants = 1000

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 2500,
    'T': T,
    'precipitation': 6500e-5,
    'seed': seed,
}

surfix = 'test2'
folder = f'Data/temp/new_buffers_test'

print(seed)
sim = Simulation(folder=folder, alias=surfix, **kwargs)
print(sim.seed)
# sim.initiate_uniform_radii(n=num_plants, r_min=0.1, r_max=30)
sim.run(T=T)
sim.plot_buffers()
plt.show()



#### THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])











# if plot_results:
#     sim.plot()

# sim.cleanup()
# del sim