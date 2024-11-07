import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


SQRT2 = np.sqrt(2)
PI = np.pi
SQRTPI = np.sqrt(PI)


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
        self.kde = gaussian_kde(
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
