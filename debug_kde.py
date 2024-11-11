import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from scipy.stats import gaussian_kde as kde


class custom_kde(gaussian_kde):
    def __init__(self, dataset, bw_method=None, weights=None):
        super().__init__(dataset, bw_method=bw_method, weights=weights)

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        self._data_covariance = np.eye(self.dataset.shape[0])
        self._data_cho_cov = np.eye(self.dataset.shape[0])
        self.covariance = self._data_covariance * self.factor**2
        self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
        self.log_det = 2*np.log(np.diag(self.cho_cov
                                        * np.sqrt(2*np.pi))).sum()


m2pp = 10_000
ns = [10, 100, 1_000, 10_000, 20_000]


for i, n in enumerate(ns):
    _m = float(np.sqrt(1/(m2pp * n)))
    print(f'{n=}, {_m=}')
    # Define the grid on which to evaluate the KDE

    positions1 = np.random.normal(loc=[-0.3, -0.3], scale=0.1, size=(n//3, 2))
    positions2 = np.random.normal(loc=[0.3, 0.3], scale=0.1, size=(n//3, 2))
    positions3 = np.random.normal(loc=[0.0, 0.0], scale=0.1, size=(n//3, 2))
    positions = np.concatenate((positions1, positions2, positions3), axis=0)
    positions = positions[(positions[:, 0] >= -0.5) & (positions[:, 0] <= 0.5)
                          & (positions[:, 1] >= -0.5) & (positions[:, 1] <= 0.5)]
    kt = KDTree(positions, leafsize=10)
    radii = np.random.uniform(0.01, 30, len(positions)) * _m
    areas = np.pi * radii**2

    bw = 100*_m
    res = 100

    print(f'{res=}')

    x = np.linspace(-0.5, 0.5, res)
    X, Y = np.meshgrid(x, x)
    XY_vstack = np.vstack([X.ravel(), Y.ravel()])

    # Calculate the kernel density estimate
    kde = custom_kde(positions.T, bw_method=bw, weights=areas)
    Z = kde(XY_vstack).reshape(X.shape)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(Z, origin='lower', extent=(
        x[0], x[-1], x[0], x[-1]), cmap='Greys', label='kde')
    ax.contour(X, Y, Z, levels=[0.1, 0.5, 1.0], colors=[
        'r', 'g', 'b'][::-1], alpha=0.5, linewidths=1)
    ax.scatter(positions[:, 0], positions[:, 1],
               s=1e5*areas, c='g', alpha=1, label='plants')
    ax.plot([-0.45, -0.45 + bw],
            [0.45, 0.45], 'b--', lw=1, label='bandwidth')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'n = {n}')
    ax.legend(loc='lower right')
    fig.colorbar(im, ax=ax)

fig.dpi = 300
fig.tight_layout()
plt.show()
