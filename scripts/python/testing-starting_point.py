import matplotlib.pyplot as plt
from mods.simulation import *       
starting_point_folder = 'Data/baseline/L2000'
starting_point_alias = 'baseline_global_L2e3_PR675e_4_DE3e_1_250420_141706'

sim = Simulation(
    folder=starting_point_folder, alias=starting_point_alias)
sim.run(T=10_000, convergence_stop=False)

figs, axs, titles = sim.plot_buffers(title=starting_point_alias, save=True)
plt.show()
