import numpy as np
import quadT


def dist_sq(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2)


def check_collision(p1, p2):
    return dist_sq(p1, p2) < (p1.r + p2.r) ** 2


class Simulation:
    def __init__(self, qt: quadT.QuadTree, **kwargs):
        self.qt = qt
        self.qt_new = qt
        self.states = []

        self.r_max_global = kwargs.get('r_max_global', None)

    def add_plant(self, plant):
        self.qt_new.insert(quadT.Point(plant.pos, data=plant))

    def site_quality(self, pos):
        bb_half_width = self.r_max_global
        max_plant_area = np.pi*self.r_max_global**2
        bb = quadT.BoundingBox(pos, bb_half_width)
        plants_nearby = [point.data for point in self.qt.query(bb)]
        if len(plants_nearby) == 0:
            return 0
        density_nearby = sum(
            plant.r**2*np.pi for plant in plants_nearby)/(len(plants_nearby)*max_plant_area)
        return density_nearby

    def get_collisions(self, plant, r_max_global):
        collisions = []
        bb = quadT.BoundingCircle(plant.pos, 2*(plant.r + r_max_global))
        for point in self.qt.query(bb):
            other_plant = point.data
            if other_plant != plant:
                if check_collision(plant, other_plant):
                    collisions.append(other_plant)
        return collisions

    def step(self):
        self.qt_new = quadT.QuadTree(center=self.qt.boundary.center, half_width=self.qt.boundary.half_width,
                                     half_height=self.qt.boundary.half_height, capacity=self.qt.capacity)

        for point in self.qt.all_points():
            plant = point.data.copy()
            collisions = self.get_collisions(plant, self.r_max_global)
            plant.update(self, collisions)

            if not plant.is_dead:
                self.add_plant(plant)

        self.qt = self.qt_new
        self.states.append(self.qt)
