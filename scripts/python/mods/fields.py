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


@njit(parallel=True)
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
            rho[i] += m[j] * W(dx[i, j], dy[i, j], hSq[j])

    return rho.reshape((M, 1))


def boundary_check(pos, r, xlim=(-0.5, 0.5), ylim=(-0.5, 0.5)):
    is_close_left_boundary = pos[0] - r < xlim[0]
    is_close_right_boundary = pos[0] + r > xlim[1]
    is_close_bottom_boundary = pos[1] - r < ylim[0]
    is_close_top_boundary = pos[1] + r > ylim[1]
    is_close_boundary = np.array(
        [is_close_left_boundary, is_close_right_boundary, is_close_bottom_boundary, is_close_top_boundary])
    return is_close_boundary


class DensityFieldSPH:
    def __init__(self, half_width, half_height, resolution):
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
        self.values = np.zeros((resolution, resolution))

    def query(self, pos):
        # Find the nearest neighbors using KDTree
        dist, idx = self.KDTree.query(pos)
        # Convert the 1D index from KDTree query to 2D index for the values array
        idx_2d = np.unravel_index(idx, (self.resolution, self.resolution))
        return self.values[idx_2d]

    def update(self, state):
        if len(state) == 0:
            self.values = np.zeros((self.resolution, self.resolution))
        else:
            positions = np.array([(plant.x, plant.y)
                                  for plant in state])
            radiiSq = np.array([plant.r for plant in state])**2

            sigmasSq = radiiSq * self.bandwidth_factor**2
            
            self.values = getDensity(r=self.grid_points,
                                     pos=positions,
                                     m=np.pi*radiiSq,
                                     hSq=sigmasSq).reshape(
                self.resolution, self.resolution)

    def integrate(self):
        return np.sum(self.values) / self.resolution**2

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

@njit(parallel=True)
def compute_density(grid_points, positions, radiiSq, sigmas, resolution):
    values = np.zeros(grid_points.shape[0])
    query_radius_factor = 3
    
    sigmaSq = sigmas**2
    mass = np.pi * radiiSq
    for i in prange(len(positions)):
        query_radiusSq = (query_radius_factor * sigmas[i])**2
        
        for j in prange(grid_points.shape[0]):
            distSq = np.sum((positions[i] - grid_points[j])**2)
            
            if distSq <= query_radiusSq:
                values[j] += mass[i]   *   1/(2 * np.pi * sigmaSq[i])   *   np.exp(-distSq / (2 * sigmaSq[i]))   
    return values

class DensityFieldCustom:
    def __init__(self, half_width, half_height, resolution):
        # print('DensityFieldCustom: DensityField is using custom density estimation.')

        dx = 1 / resolution
        dy = 1 / resolution
        xx = np.linspace(-half_width + dx/2,
                         half_width - dx/2, resolution)
        yy = np.linspace(-half_height + dy/2,
                         half_height - dy/2, resolution)
        X, Y = np.meshgrid(xx, yy)

        self.grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        self.values = np.zeros(self.grid_points.shape[0])
        self.KDTree_grid = KDTree(self.grid_points)
        self.resolution = resolution

    def query(self, pos):
        # Find the nearest neighbors using KDTree
        dist, idx = self.KDTree_grid.query(pos)
        return self.values[idx]

    def update(self, plants):
        if len(plants) == 0:
            return

        positions = np.array([(plant.x, plant.y) for plant in plants])
        radii = np.array([plant.r for plant in plants])
        sigmas = np.array([plant.r * plant.density_range/plant.r_max for plant in plants])
        
        sigmasSq = sigmas**2
        
        radiiSq = radii**2
        areas = np.pi * radiiSq

        self.values = np.zeros(self.grid_points.shape[0])
        self.values = compute_density(self.grid_points, positions, radiiSq, sigmas, self.resolution)
    
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
        # ax.set_xticks([])
        # ax.set_yticks([])
        return fig, ax

    def get_values(self):
        return copy.deepcopy(self.values)
