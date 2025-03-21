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
    'L': 1000,
    'precipitation': 0.5,
    'seed': seed,
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/debugging'
alias = f'temp1'

os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, **kwargs, override=False)

num_plants = int(kwargs['L'])
sim.initiate_non_overlapping(n=num_plants, species_list=sim.species_list, max_attempts=50*num_plants)

sim.step()
sim.plot()

plt.show()