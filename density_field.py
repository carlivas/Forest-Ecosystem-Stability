import numpy as np
from scipy.interpolate import RegularGridInterpolator


class DensityField():
    def __init__(self, half_width, half_height, check_radius, check_resolution=None):
        self.half_width = half_width
        self.half_height = half_height
        if check_resolution is None:
            self.shape = (int(2 * half_width / check_radius),
                          int(2 * half_height / check_radius))
        else:
            self.shape = (check_resolution, check_resolution)
        self.values = np.zeros(self.shape)
        self.xx = np.linspace(-half_width, half_width, self.shape[0])
        self.yy = np.linspace(-half_height, half_height, self.shape[1])

        self.check_radius = check_radius
        self.check_area = np.pi * self.check_radius**2

        self.interp_func = RegularGridInterpolator(
            (self.xx, self.yy), self.values, method='linear')

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
            (I, J) = self.shape
            for i in range(I):
                for j in range(J):
                    pos = np.array([self.xx[i],
                                    self.yy[j]])
                    self.values[i, j] = self.density_local(simulation, pos)

        self.interp_func = RegularGridInterpolator(
            (self.xx, self.yy), self.values, method='linear')

    def get_values(self):
        return self.values.copy()
