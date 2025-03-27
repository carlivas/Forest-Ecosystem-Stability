import numpy as np
import matplotlib.pyplot as plt
import copy
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import gaussian_kde
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mods.spatial import *

SQRT2 = np.sqrt(2)
PI = np.pi
SQRTPI = np.sqrt(PI)


# @njit
# def W(x, y, hSq):
#     rSq = x**2 + y**2
#     w = 1/(2 * PI * hSq) * np.exp(-rSq / (2 * hSq))
#     return w


# @njit(parallel=True)
# def getPairwiseSeparations(ri, rj):
#     M = ri.shape[0]
#     N = rj.shape[0]

#     dx = np.empty((M, N))
#     dy = np.empty((M, N))

#     for i in prange(M):
#         for j in prange(N):
#             dx[i, j] = ri[i, 0] - rj[j, 0]
#             dy[i, j] = ri[i, 1] - rj[j, 1]

#     return dx, dy


# @njit(parallel=True)
# def getDensity(r, pos, m, hSq):
#     M = r.shape[0]
#     N = pos.shape[0]

#     dx, dy = getPairwiseSeparations(r, pos)

#     rho = np.zeros(M)
#     for i in prange(M):
#         for j in prange(N):
#             rho[i] += m[j] * W(dx[i, j], dy[i, j], hSq[j])

#     return rho.reshape((M, 1))

# class DensityFieldSPH:
#     def __init__(self, half_width, half_height, resolution):
#         print('DensityFieldSPH: DensityField is using smoothed particle hydrodynamics density estimation.')

#         dx = 1 / resolution
#         dy = 1 / resolution
#         xx = np.linspace(-half_width + dx/2,
#                          half_width - dx/2, resolution)
#         yy = np.linspace(-half_height + dy/2,
#                          half_height - dy/2, resolution)
#         X, Y = np.meshgrid(xx, yy)

#         self.resolution = resolution
#         self.grid_points = np.vstack([X.ravel(), Y.ravel()]).T
#         self.KDTree = KDTree(self.grid_points)
#         self.values = np.zeros((resolution, resolution))

#     def query(self, pos):
#         # Find the nearest neighbors using KDTree
#         dist, idx = self.KDTree.query(pos)
#         # Convert the 1D index from KDTree query to 2D index for the values array
#         idx_2d = np.unravel_index(idx, (self.resolution, self.resolution))
#         return self.values[idx_2d]

#     def update(self, state):
#         if len(state) == 0:
#             self.values = np.zeros((self.resolution, self.resolution))
#         else:
#             positions = np.array([(plant.x, plant.y)
#                                   for plant in state])
#             radiiSq = np.array([plant.r for plant in state])**2

#             sigmasSq = radiiSq * self.bandwidth_factor**2
            
#             self.values = getDensity(r=self.grid_points,
#                                      pos=positions,
#                                      m=np.pi*radiiSq,
#                                      hSq=sigmasSq).reshape(
#                 self.resolution, self.resolution)

#     def integrate(self):
#         return np.sum(self.values) / self.resolution**2

#     def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5], colorbar=True):
#         if ax is None:
#             fig, ax = plt.subplots(1, 1, figsize=(size, size))
#         im = ax.imshow(self.get_values(), origin='lower', cmap='Greys',
#                        vmin=vmin, vmax=vmax, extent=extent)
#         ax.set_title(title, fontsize=7)
#         if colorbar:
#             divider = make_axes_locatable(ax)
#             cax = divider.append_axes("right", size="5%", pad=0.05)
#             cbar = plt.colorbar(im, cax=cax)
#             cbar.set_label('Density')
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         return fig, ax

#     def get_values(self):
#         return copy.deepcopy(self.values)

def compute_density_field(grid_points, positions, radii, query_radius_factor, bandwidth_factor):
    positions = np.atleast_2d(positions)
    radii = np.atleast_1d(radii)
    density_field = np.zeros(grid_points.shape[0])
    kdtree_grid = KDTree(grid_points)
    
    sigmas = bandwidth_factor * radii
    query_radii = query_radius_factor * sigmas
    
    radiiSq = radii**2
    sigmasSq = sigmas**2
    weights = np.pi * radiiSq

    for i, p in enumerate(positions):
        for idx in kdtree_grid.query_ball_point(p, r=query_radii[i]):
            distSq = np.sum((p - grid_points[idx])**2)
            density_field[idx] += weights[i] * np.exp(-distSq / (2 * sigmasSq[i])) / (2 * np.pi * sigmasSq[i])

    return density_field

class DensityFieldCustom:
    def __init__(self, box, resolution):
        # print('DensityFieldCustom: DensityField is using custom density estimation.')
        self.box = box
        dx = 1 / resolution
        dy = 1 / resolution
        xx = np.linspace(box[0, 0] + dx/2,
                         box[0, 1] - dx/2, resolution)
        yy = np.linspace(box[1, 0] + dy/2,
                         box[1, 1] - dy/2, resolution)
        X, Y = np.meshgrid(xx, yy)

        self.grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        self.values = np.zeros(self.grid_points.shape[0])
        self.KDTree_grid = KDTree(self.grid_points)
        self.resolution = resolution

    def query(self, pos):
        inside_boundary = ~np.any(boundary_check(self.box, pos), axis=1)
        # Find the nearest neighbors using KDTree
        dist, idx = self.KDTree_grid.query(pos)
        return self.values[idx] * inside_boundary

    def update(self, plants):
        if len(plants) == 0:
            return
        
        bandwidth_factor = 2
        query_radius_factor = 3

        positions = np.array([(plant.x, plant.y) for plant in plants])
        radii = np.array([plant.r for plant in plants])
        query_radii = query_radius_factor * bandwidth_factor * radii
    
        positions_new, index_pairs, was_shifted = positions_shift_periodic(boundary=self.box, positions=positions, radii=query_radii, duplicates=True)
    
        radii_new = radii[index_pairs[:, 0]]
        
        self.values = np.zeros(self.grid_points.shape[0])
        self.values = compute_density_field(self.grid_points, positions_new, radii_new, query_radius_factor, bandwidth_factor)
    
    def integrate(self):
        return np.sum(self.values) / self.resolution**2

    def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5], colorbar=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        im = ax.imshow(self.get_values().reshape(self.resolution, self.resolution), origin='lower', cmap='Greys',
                       vmin=vmin, vmax=vmax, extent=extent)
        ax.set_title(title, fontsize=7)
        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Density')
        ax.set_xticks([])
        ax.set_yticks([])
        return fig, ax

    def get_values(self):
        return copy.deepcopy(self.values)
