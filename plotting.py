import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from plant import Plant
from simulation import Simulation, StateBuffer, DataBuffer


# def plot_data_buffer(buffer, size=6):
#     fig, ax = plt.subplots(2, 1, figsize=(
#         size, size))
#     fig.tight_layout(pad=3.0)
#     ax[0].plot(buffer[:, 0], buffer[:, 1],
#                label='Biomass', color='green')
#     # ax[0].set_xticks([])
#     ax[1].plot(buffer[:, 0], buffer[:, 2],
#                label='Population Size', color='teal')
#     ax[1].set_xlabel('Time')

#     for ax_i in ax:
#         ax_i.grid()
#         ax_i.legend()
#     return fig, ax


# def plot(size=6):
#     plot_state(self.get_state(), size=size)
#     return fig, ax


# def plot_state(state, t=None, size=6, fig=None, ax=None, half_width=0.5, half_height=0.5):
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(size, size))
#     # ax.set_title('State')
#     # ax.set_xlabel('Width (u)')
#     # ax.set_ylabel('Height (u)')
#     ax.set_xlim(-half_width, half_width)
#     ax.set_ylim(-half_height, half_height)
#     ax.set_aspect('equal', 'box')
#     for plant in state:
#         if t is not None:
#             ax.set_title(f't = {t}', fontsize=7)
#         ax.add_artist(plt.Circle(plant.pos, plant.r,
#                                  color='green', fill=True, transform=ax.transData))

#     ax.set_xticks([])
#     ax.set_yticks([])
#     return fig, ax


# def plot_states(states, times=None, size=6):
#     l = len(states)
#     n_rows = int(np.floor(l / np.sqrt(l)))
#     n_cols = (l + 1) // n_rows + (l % n_rows > 0)
#     print(f'simulation: {n_rows = }, {n_cols = }')

#     fig, ax = plt.subplots(
#         n_rows, n_cols, figsize=(size*n_cols, size*n_rows))
#     fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95,
#                         top=0.95, wspace=0.05, hspace=0.05)
#     fig.tight_layout()
#     if len(states) == 1:
#         plot_state(states[0], t=times[0], size=size, ax=ax)
#     else:
#         i = 0
#         # for i, state in enumerate(states):
#         while i < len(states):
#             state = states[i]
#             if n_rows == 1:
#                 k = i
#                 plot_state(
#                     state=state, t=times[i], size=size, fig=fig, ax=ax[k])
#             else:
#                 l = i//n_cols
#                 k = i % n_cols
#                 plot_state(
#                     state=state, t=times[i], size=size, fig=fig, ax=ax[l, k])

#             i += 1
#     if len(states) < n_rows*n_cols:
#         for j in range(len(states), n_rows*n_cols):
#             if n_rows == 1:
#                 ax[j].axis('off')
#             else:
#                 l = j//n_cols
#                 k = j % n_cols
#                 ax[l, k].axis('off')

#     return fig, ax


kwargs = pd.read_json('Data/sim10/kwargs_0.json')
plant_kwargs = kwargs['plant_kwargs']
sim_kwargs = kwargs['sim_kwargs']
print('plotting.py: Loaded kwargs...')

# state_buffer = pd.read_pickle('Data/sim10/state_buffer_0.pkl')
state_buffer_arr = pd.read_csv(
    'Data/sim10/state_buffer_0.csv', header=None).to_numpy()
state_buffer = StateBuffer(data=state_buffer_arr, plant_kwargs=plant_kwargs)
print('plotting.py: Loaded state_buffer...')
state_buffer.plot(size=2)

data_buffer_arr = pd.read_csv(
    'Data/sim10/data_buffer_0.csv', header=None).to_numpy()
data_buffer = DataBuffer(data=data_buffer_arr)
print('plotting.py: Loaded data_buffer...')
data_buffer.plot()

plt.show()
