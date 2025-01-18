import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

from mods.plant import Plant
from mods.simulation import Simulation, _m_from_m2pp, _m_from_domain_sides, save_simulation_results, plot_simulation_results
from mods.utilities import save_kwargs, print_nested_dict, convert_to_serializable
from mods.buffers import StateBuffer, DataBuffer, FieldBuffer, HistogramBuffer

save_results = True
plot_results = True


T = 30000
for _ in range(10):
    seed = np.random.randint(0, 1_000_000_000)
    np.random.seed(seed)
    num_plants = np.random.randint(50, 300)
    kwargs = {
        'seed': seed,
        'L': 1000,
        'T': T,
        'dispersal_range': 90,
        'precipitation': 7000e-5,
        'spawn_rate': 1,
        'growth_rate': 0.1,

        'time_step': 1,
        
        'density_field_resolution': 100,

        'buffer_size': T + 1,
        'buffer_skip': 1,
        
        'verbose': True,
    }

    save_folder = f'Data/temp/convergence_test'
    sim = Simulation(**kwargs)
    sim.initiate_uniform_radii(n=num_plants, r_min=0.1, r_max=30)

    sim.run(T=T)

    sim.print_dict()

    if save_results:
        save_simulation_results(sim, save_folder, surfix=f'{seed}')


    if plot_results:
        plot_simulation_results(sim, convergence=True)

    sim.cleanup()
    del sim