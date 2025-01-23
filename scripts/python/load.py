import matplotlib.pyplot as plt

from mods.simulation import Simulation

save_results = True
plot_results = True
folder = 'Data/temp/load_save_test'
surfix = '247088000_0'
sim = Simulation(folder=folder, alias=surfix)

for i in range(10):
    split = surfix.split('_')
    new_surfix = split[0] + '_' + str(int(split[1]) + 1)
    sim.set_path(folder, new_surfix, overwrite=True)
    print(sim.precipitation)
    sim.precipitation = sim.precipitation - 1e-4
    print(sim.precipitation)
    sim.run(T=1000)
    sim.data_buffer.plot()

plt.show()