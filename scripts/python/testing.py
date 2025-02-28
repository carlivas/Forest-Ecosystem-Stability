import numpy as np
import matplotlib.pyplot as plt
import os

from mods.plant import PlantSpecies
from mods.simulation import Simulation
from mods.buffers import StateBuffer
from mods.utilities import *

save_results = True
plot_results = True

seed = np.random.randint(0, 1_000_000_000)
np.random.seed(seed)
kwargs = {
    'L': 200,
    'precipitation': 1,
    'seed': seed,
}
num_plants = int(2/3 * kwargs['L'])


folder = f'Data/species_tests'
alias = f'species_test'

os.makedirs(folder, exist_ok=True)

species = PlantSpecies()


species_list = []
for i in range(5):
    species = PlantSpecies(species_id=i, r_min=1, r_max=15, growth_rate = 0.1, dispersal_range=90, density_range=90, maturity_size=1, germination_chance = 1)
    for k, v in species.__dict__.items():
        if k not in ['species_id', 'name']:
            species.__dict__[k] = v*(1 + np.random.uniform(-0.1, 0.1)) if isinstance(v, (int, float)) else v
    species_list.append(species)


sim = Simulation(folder=folder, alias=alias, species_list=species_list, **kwargs, override=True)

sim.initiate_uniform_radii(
    n=num_plants, species_list=sim.species_list)
sim.step()
sim.plot()
plt.show()
l = len(sim.plants)
print(f'Number of plants: {l}')

while l > 0:
    sim.run(T=2_000, convergence_stop=True, min_population=1)
    sim.precipitation = sim.precipitation * 0.5
    
    l = len(sim.plants)
    print(f'Precipitation: {sim.precipitation}')
    print(f'Number of plants: {l}')



figs, axs, titles = sim.plot_buffers(title=alias)
os.makedirs(folder + '/figures', exist_ok=True)
for i, (fig, title) in enumerate(zip(figs, titles)):
    tilte = title.replace(' ', '-').lower()
    fig.savefig(f'{folder}/figures/{title}.png', dpi=600)

# anim, _ = StateBuffer.animate(sim.state_buffer.get_data())
# anim.save(f'{folder}/figures/state_anim-{alias}.mp4', writer='ffmpeg')

#### THINK ABOUT THIS
# field = pd.DataFrame([[x, y, d] for (x, y), d in zip(self.density_field.positions, self.density_field.values)], columns=['x', 'y', 'd'])
