import numpy as np
import matplotlib.pyplot as plt

from mods.simulation import Simulation

folder = 'Data/temp/load_save_test'
surfix = '0'

T = 600
sim = Simulation(folder=folder, alias=surfix)
sim.run(T=T)

for i in range(10):
    new_surfix = str(int(surfix) + i + 1)
    sim.set_path(folder, new_surfix)
    sim.precipitation = sim.precipitation - 1e-4
    print(f'precipitation = {sim.precipitation}')
    sim.run(T=T)

plt.show()