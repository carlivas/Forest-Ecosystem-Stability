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
    'L': 200,
    'precipitation': 0.5,
    'seed': seed,
    'density_scheme': 'global',
}

folder = f'../../Data/hysteresis/'
current_time = datetime.now().strftime("%y%m%d_%H%M%S")
alias = generate_alias(id=f'hysteresis_{kwargs["density_scheme"]}', keys=[], time=True, **kwargs)
sim = Simulation(folder=folder, alias=alias, **kwargs)

# spawn non-overlapping plants
print('\ntesting_hysteresis: spawn non-overlapping plants')
sim.spawn_non_overlapping(target_density=0.3)

# equilibrate the system
print('\ntesting_hysteresis: equilibrate the system')
sim.run(T=1000, dp=0)
data_buffer = sim.data_buffer.get_data()
equilibriated_biomass = data_buffer['Biomass'][data_buffer['Time'] > 500].mean()

# run linear precipitation decrease until extinction
print('\ntesting_hysteresis: run linear precipitation decrease until extinction')
dp = - 1/25000
sim.is_running = True
while len(sim.plants) > 0:
    sim.step()
    sim.precipitation += dp
    if sim.precipitation <= 0:
        break
sim.is_running = False

# equilibrate the system
print('\ntesting_hysteresis: equilibrate the system')
sim.run(T=1000, dp=0)

# run linear precipitation increase until repopulation
print('\ntesting_hysteresis: run linear precipitation increase until repopulation')
sim.is_running = True
while abs(sim.get_biomass() - equilibriated_biomass) > 0.01:
    sim.step()
    sim.precipitation -= dp
    if sim.precipitation >= 1:
        break
sim.is_running = False



fig, ax = plt.subplots(1, 1, figsize=(10, 5))
data_buffer = sim.data_buffer.get_data()
transient_time = 1000
ax.plot(data_buffer['Precipitation'][data_buffer['Time'] > transient_time],
         data_buffer['Biomass'][data_buffer['Time'] > transient_time], 'g')
ax.set_xlabel('Precipitation')
ax.set_ylabel('Biomass')
ax.set_title('Hysteresis')
ax.grid()

save_figs = True
if save_figs:
    folder = os.path.join(sim.folder, 'figures')
    os.makedirs(folder, exist_ok=True)
    fig.savefig(os.path.join(folder, f'hysteresis_{alias}.png'), dpi=300)
    print(f'\ntesting_hysteresis: saved figure to {os.path.join(sim.folder, f"hysteresis_{alias}.png")}')

figs, axs, titles = sim.plot_buffers(title=alias, save=save_figs, dpi=300)
plt.show()