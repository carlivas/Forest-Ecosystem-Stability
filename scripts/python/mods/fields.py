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
        box = np.asarray(box, dtype=float)
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
        inside_boundary = ~np.any(outside_box_check(self.box, pos), axis=1)
        # Find the nearest neighbors using KDTree
        dist, idx = self.KDTree_grid.query(pos)
        return self.values[idx] * inside_boundary

    def update(self, plants, density_scheme='local'):
        if len(plants) == 0:
            return
        
        if density_scheme == 'local':
            bandwidth_factor = 2
            query_radius_factor = 3

            positions = plants.positions
            radii = plants.radii
            query_radii = query_radius_factor * bandwidth_factor * radii
        
            positions_new, index_pairs, was_shifted = positions_shift_periodic(box=self.box, positions=positions, radii=query_radii, duplicates=True)
        
            radii_new = radii[index_pairs[:, 0]]
            
            self.values = np.zeros(self.grid_points.shape[0])
            self.values = compute_density_field(self.grid_points, positions_new, radii_new, query_radius_factor, bandwidth_factor)
        elif density_scheme == 'global':
            box_area = np.prod(np.diff(self.box, axis=1))
            self.values = np.sum(plants.radii**2) * np.pi / box_area
    
    def integrate(self):
        return np.sum(self.values) / self.resolution**2
    
    @staticmethod
    def plot(field, size=2, title='', fig=None, ax=None, vmin=0, vmax=1, box=None, colorbar=True, boundary_condition='box', t=None, **kwargs):
        resolution = np.sqrt(field.size).astype(int)
        field = field.reshape(resolution, resolution)
        
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))

        if box is None:
            box = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        else:
            box = np.array(box, dtype=float)

        print('DensityFieldCustom: DensityField is using global density scheme.')
        im = ax.imshow(field, origin='lower', cmap='Greens',
                    vmin=vmin, vmax=vmax, extent=[box[0, 0], box[0, 1], box[1, 0], box[1, 1]])
        ax.set_title(title, fontsize=7)
        
        if boundary_condition.lower() == 'periodic':
            rect = plt.Rectangle(
                (box[0, 0], box[1, 0]),  # Bottom-left corner
                box[0, 1] - box[0, 0],  # Width
                box[1, 1] - box[1, 0],  # Height
                edgecolor='black',
                facecolor='none',
                linestyle='-',
                linewidth=1,
                alpha=0.2
            )
            ax.add_patch(rect)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    extent_shifted = [
                        box[0, 0] + dx * (box[0, 1] - box[0, 0]),
                        box[0, 1] + dx * (box[0, 1] - box[0, 0]),
                        box[1, 0] + dy * (box[1, 1] - box[1, 0]),
                        box[1, 1] + dy * (box[1, 1] - box[1, 0]),
                    ]
                    ax.imshow(field, origin='lower', cmap='Greys',
                              vmin=vmin, vmax=vmax, extent=extent_shifted, alpha=1)

        
        density = np.sum(field) / (resolution**2)
        title = f'{title}  total density = {density:.2f}'
        ax.set_title(title, fontsize=10)
            
        scale = 1.2
        ax.set_xlim(box[0, 0] * scale, box[0, 1] * scale)
        ax.set_ylim(box[1, 0] * scale, box[1, 1] * scale)
        if t is not None:
            t = float(round(t, 2))
            ax.text(0.0, -0.6*scale, f'{t=}', ha='center', fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        
        
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
