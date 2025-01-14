import numpy as np
import json
import pandas as pd
import time
import matplotlib.pyplot as plt

from mods.simulation import sim_from_data
from mods.fields import getDensity
from scipy.signal import correlate2d

np.random.seed(0)  # For reproducibility

fig, ax = plt.subplots(2, 4, figsize=(12, 8))

res = 100
xx = np.arange(0, res)
yy = np.arange(0, res)
XY = np.meshgrid(xx, yy)
grid_points = np.vstack([XY[0].ravel(), XY[1].ravel()]).T

ref_field = np.random.rand(res, res)

num_positions1 = 500
positions1 = np.random.randint(0, res, size=(num_positions1, 2))
weights1 = np.ones(num_positions1)
bw1 = 2
density_field1 = getDensity(grid_points, positions1, weights1, bw1**2).reshape(res, res)

num_positions2 = 20
positions2 = np.random.randint(res//5, 3*res//4, size=(num_positions2, 2))
positions2[:, 1] = res // 2
weights2 = np.ones(num_positions2)
bw2 = 5
density_field2 = getDensity(grid_points, positions2, weights2, bw2**2).reshape(res, res)

num_positions3 = 1
# positions3 = np.random.randint(0, res, size=(num_positions3, 2))
positions3 = np.array([[50, 50]])
weights3 = np.ones(num_positions3)
bw3 = 10
density_field3 = getDensity(grid_points, positions3, weights3, bw3**2).reshape(res, res)


ax[0, 0].imshow(ref_field, cmap='hot', interpolation='nearest')
ax[0, 1].imshow(density_field1, cmap='hot', interpolation='nearest')
ax[0, 2].imshow(density_field2, cmap='hot', interpolation='nearest')
ax[0, 3].imshow(density_field3, cmap='hot', interpolation='nearest')
# ax[0, 1].scatter(positions1[:, 0], positions1[:, 1], c='k', s=bw1**2)
# ax[0, 2].scatter(positions2[:, 0], positions2[:, 1], c='k', s=bw2**2)
# ax[0, 3].scatter(positions3[:, 0], positions3[:, 1], c='k', s=bw3**2)

cross_correlation_ref = correlate2d(ref_field, ref_field, mode='full') / (res**4)
cross_correlation1 = correlate2d(density_field1, density_field1, mode='full') /(res**4)
cross_correlation2 = correlate2d(density_field2, density_field2, mode='full') /(res**4)
cross_correlation3 = correlate2d(density_field3, density_field3, mode='full') /(res**4)

cross_correlation_ref_value = cross_correlation_ref.sum()
cross_correlation_value1 = cross_correlation1.sum()
cross_correlation_value2 = cross_correlation2.sum()
cross_correlation_value3 = cross_correlation3.sum()

ax[1, 0].imshow(cross_correlation_ref, cmap='hot', interpolation='nearest')
ax[1, 1].imshow(cross_correlation1, cmap='hot', interpolation='nearest')
ax[1, 2].imshow(cross_correlation2, cmap='hot', interpolation='nearest')
ax[1, 3].imshow(cross_correlation3, cmap='hot', interpolation='nearest')

ax[1, 0].set_title(f"{cross_correlation_ref_value:.3e}")
ax[1, 1].set_title(f"{cross_correlation_value1:.3e}")
ax[1, 2].set_title(f"{cross_correlation_value2:.3e}")
ax[1, 3].set_title(f"{cross_correlation_value3:.3e}")

for a in ax.flatten():
    a.axis('off')
    fig.colorbar(a.images[0], ax=a)
plt.show()


