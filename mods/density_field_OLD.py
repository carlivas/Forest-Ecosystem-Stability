import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.interpolate import RegularGridInterpolator
from mpl_toolkits.axes_grid1 import make_axes_locatable


SQRT2 = np.sqrt(2)
PI = np.pi
SQRTPI = np.sqrt(PI)


class DensityField():
    def __init__(self, half_width, half_height, check_radius, resolution, values=None):
        self.resolution = resolution
        self.xx = np.linspace(-half_width, half_width, self.resolution)
        self.yy = np.linspace(-half_height, half_height, self.resolution)

        if values is not None:
            self.values = values
        else:
            self.values = np.zeros((resolution, resolution))

        self.check_radius = check_radius
        # self.check_radius = SQRT2 * half_width / resolution
        self.check_area = PI * self.check_radius**2

        self.interp_func = RegularGridInterpolator(
            (self.xx, self.yy), self.values, method='linear')

    # def smoothing_kernel(self, x, r):
    #     σ = r
    #     return 1 / (σ * SQRT2 * SQRTPI) * np.exp(- 1/2 * np.dot(x, x) / σ**2)

    # def density_local(self, simulation, pos):
    #     indices = simulation.kt.query_ball_point(
    #         x=pos, r=self.check_radius, workers=-1)
    #     plants_nearby = [simulation.plants[i] for i in indices]
    #     if len(plants_nearby) == 0:
    #         return 0
    #     else:
    #         density = 0
    #         for plant in plants_nearby:
    #             density += self.smoothing_kernel(pos - plant.pos, plant.r)
    #         return density

    def density_local(self, simulation, pos):
        indices = simulation.kt.query_ball_point(
            x=pos, r=self.check_radius, workers=-1)
        plants_nearby = [simulation.plants[i] for i in indices]
        if len(plants_nearby) == 0:
            return 0
        else:
            plant_covered_area = sum(plant.area for plant in plants_nearby)
            density_nearby = plant_covered_area/self.check_area
            return density_nearby

    def query(self, pos):
        return self.interp_func(pos)

    def update(self, simulation):
        self.values = np.zeros_like(self.values)
        if simulation.kt is not None:
            (I, J) = (self.resolution, self.resolution)
            for i in range(I):
                for j in range(J):
                    pos = np.array([self.xx[i],
                                    self.yy[j]])
                    self.values[i, j] = self.density_local(simulation, pos)

        self.interp_func = RegularGridInterpolator(
            (self.xx, self.yy), self.values, method='linear')

    def get_interpolated_values(self, fine_resolution):
        xx_fine = np.linspace(self.xx[0], self.xx[-1], fine_resolution)
        yy_fine = np.linspace(self.yy[0], self.yy[-1], fine_resolution)

        # Interpolate the values
        X_fine, Y_fine = np.meshgrid(xx_fine, yy_fine)
        points_fine = np.array([X_fine.flatten(), Y_fine.flatten()]).T
        values_fine = self.interp_func(points_fine).reshape(
            fine_resolution, fine_resolution).T

        return values_fine

    def get_interpolated_field(self, fine_resolution):
        values_fine = self.get_interpolated_values(fine_resolution)
        return DensityField(self.xx[-1], self.yy[-1], self.check_radius, resolution=fine_resolution, values=values_fine)

    def plot(self, size=2, title='Density field', fig=None, ax=None, vmin=0, vmax=None, extent=[-0.5, 0.5, -0.5, 0.5]):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(size, size))
        im = ax.imshow(self.values.T, origin='lower', cmap='Greys',
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
        return copy.deepcopy(self.values)
