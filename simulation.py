import numpy as np
from scipy.spatial import KDTree
import copy

from plant import Plant


def check_pos_collision(pos, plant):
    return np.sum((pos - plant.pos) ** 2) < plant.r ** 2


def check_collision(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2) < (p1.r + p2.r) ** 2


class Simulation:
    def __init__(self, **kwargs):
        self.plants = []

        self.land_quality = kwargs.get('land_quality')
        self.density_check_radius = kwargs.get('density_check_radius')
        self.density_check_area = np.pi * self.density_check_radius**2

        self.kt_leafsize = kwargs.get('kt_leafsize')
        self.kt = None

    def add(self, plant):
        if isinstance(plant, Plant):
            self.plants.append(plant)
        elif isinstance(plant, (list, np.ndarray)):
            for p in plant:
                if isinstance(p, Plant):
                    self.plants.append(p)
                else:
                    raise ValueError(
                        "All elements in the array must be Plant objects")
        else:
            raise ValueError(
                "Input must be a Plant object or an array_like of Plant objects")

    def update_kdtree(self):
        if len(self.plants) == 0:
            self.kt = None
        else:
            self.kt = KDTree(
                [plant.pos for plant in self.plants], leafsize=self.kt_leafsize)

    def step(self):
        # First Phase: Update all plants
        for plant in self.plants:
            plant.update(self)

        # Second Phase: Collect non-dead plants and add them to the new state
        new_plants = []
        for plant in self.plants:
            if not plant.is_dead:
                new_plants.append(plant)
        self.plants = new_plants

        # Update KDTree
        self.update_kdtree()

    def get_collisions(self, plant):
        plant.is_colliding = False
        collisions = []
        indices = self.kt.query_ball_point(
            x=plant.pos, r=plant.d, workers=-1)
        for i in indices:
            other_plant = self.plants[i]
            if other_plant != plant:
                if check_collision(plant, other_plant):
                    plant.is_colliding = True
                    other_plant.is_colliding = True
                    collisions.append(other_plant)
        return collisions

    def site_quality(self, pos):
        indices = self.kt.query_ball_point(
            x=pos, r=self.density_check_radius, workers=-1)
        plants_nearby = [self.plants[i] for i in indices]
        collisions_nearby = np.array(
            [check_pos_collision(pos, plant) for plant in plants_nearby])
        if len(plants_nearby) == 0:
            return 0
        elif collisions_nearby.any():
            return 0
        else:
            plant_covered_area = sum(plant.area for plant in plants_nearby)
            density_nearby = plant_covered_area/self.density_check_area
            return density_nearby + self.land_quality

    def state(self):
        return copy.deepcopy(self.plants)
