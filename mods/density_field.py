import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


SQRT2 = np.sqrt(2)
PI = np.pi
SQRTPI = np.sqrt(PI)


class kde(gaussian_kde):
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


class DensityField():
    def __init__(self, half_width, half_height, check_radius, resolution, values=None):
        print('!WARNING! DensityField is using gaussian_kde instead of RegularGridInterpolator, and more testing is needed.')
        self.resolution = resolution
        self.xx = np.linspace(-half_width, half_width, self.resolution)
        self.yy = np.linspace(-half_height, half_height, self.resolution)

        self.bandwidth = check_radius
        self.kde = None

    def query(self, pos):
        return self.kde(pos)

    def update(self, simulation):
        if simulation.kt is None:
            return
        X, Y = np.meshgrid(self.xx, self.yy)
        XY_vstack = np.vstack([X.ravel(), Y.ravel()])

        positions = np.array([plant.pos
                             for plant in simulation.plants])
        areas = np.array([plant.area for plant in simulation.plants])
        self.kde = kde(
            positions.T, bw_method=self.bandwidth, weights=areas)

    def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        im = ax.imshow(self.get_values(), origin='lower', cmap='Greys',
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=7)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Density')
        # ax.set_xticks([])
        # ax.set_yticks([])
        return fig, ax

    def get_values(self):
        X, Y = np.meshgrid(self.xx, self.yy)
        XY_vstack = np.vstack([X.ravel(), Y.ravel()])
        values = self.kde(XY_vstack).reshape(self.resolution, self.resolution)
        return copy.deepcopy(values)
