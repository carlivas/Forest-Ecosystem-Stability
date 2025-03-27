import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from mods.plant import *
from mods.simulation import *
from mods.buffers import *
from mods.utilities import *

# seed = np.random.randint(0, 1_000_000_000)
seed = 0
kwargs = {
    'L': 2000,
    'precipitation': 0.05,
    'seed': seed,
    'land_quality': 0.0,
}

current_time = datetime.now().strftime("%y%m%d_%H%M%S")
folder = f'Data/debugging'
alias = f'250326_120746'

os.makedirs(folder, exist_ok=True)
sim = Simulation(folder=folder, alias=alias, **kwargs, override=False)
print(f'{sim.boundary_condition}')

FieldBuffer.plot_field(np.zeros((sim.density_field_buffer.resolution, sim.density_field_buffer.resolution)), title=alias, box=sim.box, boundary_condition=sim.boundary_condition)
plt.show()