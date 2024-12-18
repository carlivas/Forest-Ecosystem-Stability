import numpy as np
import matplotlib.pyplot as plt
import copy
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable

SQRT2 = np.sqrt(2)
PI = np.pi
SQRTPI = np.sqrt(PI)


@njit
def W(x, y, hSq):
    rSq = x**2 + y**2
    w = 1/(2 * PI * hSq) * np.exp(-rSq / (2 * hSq))
    return w


@njit
def getPairwiseSeparations(ri, rj):
    M = ri.shape[0]
    N = rj.shape[0]

    dx = np.empty((M, N))
    dy = np.empty((M, N))

    for i in prange(M):
        for j in prange(N):
            dx[i, j] = ri[i, 0] - rj[j, 0]
            dy[i, j] = ri[i, 1] - rj[j, 1]

    return dx, dy


@njit(parallel=True)
def getDensity(r, pos, m, hSq):
    M = r.shape[0]
    N = pos.shape[0]

    dx, dy = getPairwiseSeparations(r, pos)

    rho = np.zeros(M)
    for i in prange(M):
        for j in prange(N):
            rho[i] += m[j] * W(dx[i, j], dy[i, j], hSq)

    return rho.reshape((M, 1))


class DensityFieldSPH:
    def __init__(self, half_width, half_height, check_radius, resolution, simulation=None):
        print('DensityFieldSPH: DensityField is using smoothed particle hydrodynamics density estimation.')
        dx = 1 / resolution
        dy = 1 / resolution
        xx = np.linspace(-half_width + dx/2,
                         half_width - dx/2, resolution)
        yy = np.linspace(-half_height + dy/2,
                         half_height - dy/2, resolution)
        X, Y = np.meshgrid(xx, yy)

        self.resolution = resolution
        self.grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        self.KDTree = KDTree(self.grid_points)

        self.bandwidthSq = check_radius**2
        self.values = np.zeros((resolution, resolution))

        self.simulation = simulation

    def query(self, pos):
        # Find the nearest neighbors using KDTree
        dist, idx = self.KDTree.query(pos)
        # Convert the 1D index from KDTree query to 2D index for the values array
        idx_2d = np.unravel_index(idx, (self.resolution, self.resolution))
        return self.values[idx_2d]

    def update(self):
        if self.simulation.kt is None:
            self.values = np.zeros((self.resolution, self.resolution))
        else:
            positions = np.array([plant.pos
                                  for plant in self.simulation.state])
            areas = np.array([plant.area for plant in self.simulation.state])
            self.values = getDensity(
                self.grid_points, positions, areas, self.bandwidthSq).reshape(self.resolution, self.resolution)

    def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5], colorbar=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        im = ax.imshow(self.get_values(), origin='lower', cmap='Greys',
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=7)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Density')
        # ax.set_xticks([])
        # ax.set_yticks([])
        return fig, ax

    def get_values(self):
        return copy.deepcopy(self.values)


# class kde(gaussian_kde):
#     def __init__(self, dataset, bw_method=None, weights=None):
#         self.dataset = np.atleast_2d(np.asarray(dataset))
#         if not self.dataset.size > 1:
#             raise ValueError("`dataset` input should have multiple elements.")

#         self.d, self.n = self.dataset.shape

#         if weights is not None:
#             self._weights = np.atleast_1d(weights).astype(float)
#             self._weights /= sum(self._weights)
#             if self.weights.ndim != 1:
#                 raise ValueError("`weights` input should be one-dimensional.")
#             if len(self._weights) != self.n:
#                 raise ValueError("`weights` input should be of length n")
#             self._neff = 1/sum(self._weights**2)

#         self.set_bandwidth(bw_method=bw_method)

#     def _compute_covariance(self):
#         """Computes the covariance matrix for each Gaussian kernel using
#         covariance_factor().
#         """
#         self.factor = self.covariance_factor()
#         self._data_covariance = np.eye(self.dataset.shape[0])
#         self._data_cho_cov = np.eye(self.dataset.shape[0])
#         self.covariance = self._data_covariance * self.factor**2
#         self.cho_cov = (self._data_cho_cov * self.factor).astype(np.float64)
#         self.log_det = 2*np.log(np.diag(self.cho_cov
#                                         * np.sqrt(2*PI))).sum()


# class DensityFieldKDE:
#     def __init__(self, half_width, half_height, check_radius, resolution, simulation=None):
#         print('DensityFieldKDE: DensityField is using gaussian kernel density estimation.')
#         dx = 1 / resolution
#         dy = 1 / resolution
#         xx = np.linspace(-half_width + dx/2,
#                          half_width - dx/2, resolution)
#         yy = np.linspace(-half_height + dy/2,
#                          half_height - dy/2, resolution)
#         X, Y = np.meshgrid(xx, yy)

#         self.resolution = resolution
#         self.grid_points = np.vstack([X.ravel(), Y.ravel()])

#         self.bandwidth = check_radius
#         self.kde = None

#         self.simulation = simulation

#     def query(self, pos):
#         return self.kde(pos) * self.simulation.biomass

#     def update(self):
#         if self.simulation.kt is None:
#             return

#         positions = np.array([plant.pos
#                               for plant in self.simulation.state])
#         areas = np.array([plant.area for plant in self.simulation.state])
#         self.kde = kde(
#             positions.T, bw_method=self.bandwidth, weights=areas)

#     def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(size, size))
#         im = ax.imshow(self.get_values(), origin='lower', cmap='Greys',
#                        vmin=vmin, vmax=vmax, extent=extent)
#         ax.set_title(title, fontsize=7)
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.05)
#         cbar = plt.colorbar(im, cax=cax)
#         cbar.set_label('Density')
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         return fig, ax

#     def get_values(self):
#         values = self.kde(self.grid_points).reshape(
#             self.resolution, self.resolution) * self.simulation.biomass
#         return copy.deepcopy(values)


# # class DensityFieldKDE():
# #     def __init__(self, half_width, half_height, check_radius, resolution, values=None):
# #         self.resolution = resolution
# #         self.xx = np.linspace(-half_width, half_width, self.resolution)
# #         self.yy = np.linspace(-half_height, half_height, self.resolution)

# #         if values is not None:
# #             self.values = values
# #         else:
# #             self.values = np.zeros((resolution, resolution))

# #         self.check_radius = check_radius
# #         # self.check_radius = SQRT2 * half_width / resolution
# #         self.check_area = PI * self.check_radius**2

# #         self.interp_func = RegularGridInterpolator(
# #             (self.xx, self.yy), self.values, method='linear')

# #     # def smoothing_kernel(self, x, r):
# #     #     σ = r
# #     #     return 1 / (σ * SQRT2 * SQRTPI) * np.exp(- 1/2 * np.dot(x, x) / σ**2)

# #     # def density_local(self, simulation, pos):
# #     #     indices = simulation.kt.query_ball_point(
# #     #         x=pos, r=self.check_radius, workers=-1)
# #     #     plants_nearby = [simulation.state[i] for i in indices]
# #     #     if len(plants_nearby) == 0:
# #     #         return 0
# #     #     else:
# #     #         density = 0
# #     #         for plant in plants_nearby:
# #     #             density += self.smoothing_kernel(pos - plant.pos, plant.r)
# #     #         return density

# #     def density_local(self, simulation, pos):
# #         indices = simulation.kt.query_ball_point(
# #             x=pos, r=self.check_radius, workers=-1)
# #         plants_nearby = [simulation.state[i] for i in indices]
# #         if len(plants_nearby) == 0:
# #             return 0
# #         else:
# #             plant_covered_area = sum(plant.area for plant in plants_nearby)
# #             density_nearby = plant_covered_area/self.check_area
# #             return density_nearby

# #     def query(self, pos):
# #         return self.interp_func(pos)

# #     def update(self, simulation):
# #         self.values = np.zeros_like(self.values)
# #         if simulation.kt is not None:
# #             (I, J) = (self.resolution, self.resolution)
# #             for i in range(I):
# #                 for j in range(J):
# #                     pos = np.array([self.xx[i],
# #                                     self.yy[j]])
# #                     self.values[i, j] = self.density_local(simulation, pos)

# #         self.interp_func = RegularGridInterpolator(
# #             (self.xx, self.yy), self.values, method='linear')

# #     def get_interpolated_values(self, fine_resolution):
# #         xx_fine = np.linspace(self.xx[0], self.xx[-1], fine_resolution)
# #         yy_fine = np.linspace(self.yy[0], self.yy[-1], fine_resolution)

# #         # Interpolate the values
# #         X_fine, Y_fine = np.meshgrid(xx_fine, yy_fine)
# #         points_fine = np.array([X_fine.flatten(), Y_fine.flatten()]).T
# #         values_fine = self.interp_func(points_fine).reshape(
# #             fine_resolution, fine_resolution).T

# #         return values_fine

# #     def get_interpolated_field(self, fine_resolution):
# #         values_fine = self.get_interpolated_values(fine_resolution)
# #         return DensityField(self.xx[-1], self.yy[-1], self.check_radius, resolution=fine_resolution, values=values_fine)

# #     def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
# #         if ax is None:
# #             fig, ax = plt.subplots(1, 1, figsize=(size, size))
# #         im = ax.imshow(self.values.T, origin='lower', cmap='Greys',
# #                        vmin=vmin, vmax=vmax, extent=extent)
# #         ax.set_title(title, fontsize=7)
# #         divider = make_axes_locatable(ax)
# #         cax = divider.append_axes("right", size="5%", pad=0.05)
# #         cbar = plt.colorbar(im, cax=cax)
# #         cbar.set_label('Density')
# #         # ax.set_xticks([])
# #         # ax.set_yticks([])
# #         return fig, ax

# #     def get_values(self):
# #         return copy.deepcopy(self.values)
