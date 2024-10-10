import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plant import Plant
from simulation import Simulation

biomass = np.genfromtxt(r'Data/biomass.csv', delimiter=',', skip_header=1)
density_field = np.genfromtxt(
    r'Data/density_field.csv', delimiter=',', skip_header=1)
num_plants = np.genfromtxt(r'Data/num_plants.csv',
                           delimiter=',', skip_header=1)

fig, ax = plt.subplots(2, 1, figsize=(5, 3), sharex=True)
ax[0].plot(biomass, color='green', label='Biomass')
# ax[0].set_title('Biomass over time')
# ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Biomass')
ax[0].set_ylim(0, max(biomass)*1.1)


ax[1].plot(num_plants, color='blue', label='Density')
# ax[1].set_title('Density over time')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Density')
ax[1].set_ylim(0, max(num_plants)*1.1)

fig.tight_layout()


fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.set_title('Density field')
ax.set_xlabel('Width (u)')
ax.set_ylabel('Height (u)')
ax.set_aspect('equal', 'box')



def update(frame):
    global cbar
    ax.clear()
    ax.set_title(f'Density field {frame}/{len(density_field)}')
    ax.set_xlabel('Width (u)')
    ax.set_ylabel('Height (u)')
    ax.set_aspect('equal', 'box')
    im = ax.imshow(density_field[frame].reshape(
        int(np.sqrt(density_field.shape[1])), -1), cmap='viridis', vmin=0, vmax=np.max(density_field))
    ax.set_xticks([])
    ax.set_yticks([])



ani = animation.FuncAnimation(
    fig, update, frames=len(density_field), interval=50)
plt.show()
# print('\nSaving animation...')
