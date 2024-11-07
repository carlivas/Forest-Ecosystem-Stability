import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree

import time

m2pp = 2
ns = [10, 100, 1_000, 10_000, 20_000]

for i, n in enumerate(ns[::-1]):
    _m = float(np.sqrt(1/(m2pp * n)))
    print(f'{n=}, {_m=}')
    # Define the grid on which to evaluate the KDE

    positions = np.random.uniform(-0.5, 0.5, (n, 2))
    areas = np.pi * (np.random.uniform(0.01, 30, len(positions)) * _m)**2

    kt = KDTree(positions, leafsize=10)

    start_time = time.time()
    # USING POSITIONS
    res = 100
    print(f'{res=}')
    xx = np.linspace(-0.5, 0.5, res)
    X, Y = np.meshgrid(xx, xx)
    XY_vstack = np.vstack([X.ravel(), Y.ravel()])

    bandwidth = _m
    kde = gaussian_kde(positions.T, bw_method=bandwidth, weights=areas)
    Z = kde(XY_vstack).reshape(X.shape).T
    # Z_shift = Z - Z.mean()
    # Z = Z_shift

    print(f'Density field build time: {time.time() - start_time:.5f} s')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    im2 = ax.imshow(Z.T, origin='lower',
                    extent=(-0.5, 0.5, -0.5, 0.5), cmap='Greys', alpha=1)
    ax.scatter(positions[:, 0], positions[:, 1],
               s=areas*10, c='r', alpha=1)
    # levels = np.array([0.0, 0.1, 0.5, 1.0, np.inf])
    levels = [1.0]
    # colors = plt.cm.Reds(np.linspace(0, 1, len(levels)))
    colors = ['r', 'g', 'b', 'y']
    ax.contour(X, Y, Z.T, levels=levels, colors=colors,
               alpha=0.5, linewidths=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'n = {n}')
    fig.colorbar(im2, ax=ax)

    plt.tight_layout()
plt.show()
