import matplotlib.pyplot as plt
from mods.simulation import *       
starting_point_folder = 'Data/empty_temp'
starting_point_alias = 'empty_test_LQ1e_2_SR1e_6_250417_182002'

sim = Simulation(
    folder=starting_point_folder, alias=starting_point_alias)
sim.run(T=1_000, convergence_stop=False)
sim.run(T=10_000, convergence_stop = True)

figs, axs, titles = sim.plot_buffers(title='starting_point:' + starting_point_alias)
plt.show()
