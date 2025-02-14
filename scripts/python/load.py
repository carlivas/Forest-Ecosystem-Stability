import numpy as np
import matplotlib.pyplot as plt

from mods.simulation import Simulation

folder = '../../Data/starting_contenders/partial48775395' # REMEBER TO SET THE FOLDER
surfix = '12' #REMEMBER TO CHECK ONLY KWARGS NUMBERS AS THERE MIGHT BE PREMATURELY STOPPED VALUES IN BUFFERS
seed = folder.split('/')[-1]
alias = seed + '-' + surfix

sim = Simulation(folder=folder, alias=alias)

T = 100_000
for i in range(int(surfix), 1000):
    new_surfix = str(i + 1)
    new_alias = seed + '-' + new_surfix
    print(f'{new_surfix = }, {new_alias = }')
    sim.set_path(folder, new_alias)
    sim.precipitation = sim.precipitation - 1e-4
    print(f'precipitation = {sim.precipitation}')
    sim.run(T=T)
    if len(sim.plants) < 3:
        break

# plt.show()