import matplotlib.pyplot as plt
from mods.simulation import *
from mods.utilities import *

starting_point_folder = 'Data/temp'
aliases = get_unique_aliases(starting_point_folder)
# starting_point_alias = 'baseline_global_L2e3_PR675e_4_DE3e_1_250420_141706'

for starting_point_alias in aliases:
    print(f'Running simulation for {starting_point_alias}')
    sim = Simulation(
        folder=starting_point_folder, alias=starting_point_alias)
    while sim.get_population() < 100 and sim.t < 15_000:
        sim.run(T=1000, max_population=100, convergence_stop=False)

    T = 15_000 - sim.t
    sim.run(T=T, convergence_stop=True)

    figs, axs, titles = sim.plot_buffers(title=starting_point_alias, save=True)
plt.show()
