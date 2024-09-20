import numpy as np
import quadT

# def dist_sq(p1, p2):
#     return np.sum((p1.pos - p2.pos) ** 2)


# def check_collision(p1, p2):
#     return dist_sq(p1, p2) < (p1.r + p2.r) ** 2


class Simulation:
    def __init__(self, qt: quadT.QuadTree, **kwargs):
        self.qt = qt
        self.qt_new = qt
        self.states = []

        self.land_quality = kwargs.get('land_quality', 0.1)

        self.r_max_global = kwargs.get('r_max_global', None)
        self.max_plant_area = np.pi*self.r_max_global**2

    def add_plant(self, plant):
        self.qt_new.insert(quadT.Point(plant.pos, data=plant))

    def site_quality(self, pos):
        bb_half_width = self.r_max_global

        bb = quadT.BoundingCircle(pos, bb_half_width)
        plants_nearby = [point.data for point in self.qt.query(bb)]
        if len(plants_nearby) == 0:
            return 0
        density_nearby = sum(
            plant.A for plant in plants_nearby)/(len(plants_nearby)*self.max_plant_area)
        return density_nearby + self.land_quality

    def step(self):
        self.qt_new = quadT.QuadTree(center=self.qt.boundary.center, half_width=self.qt.boundary.half_width,
                                     half_height=self.qt.boundary.half_height, capacity=self.qt.capacity)

        plants = [point.data for point in self.qt.all_points()]

        # First Phase: Update all plants
        for plant in plants:
            plant.update(self)

        # Second Phase: Collect non-dead plants and add them to the new state
        for plant in plants:
            if not plant.is_dead:
                self.add_plant(plant.copy())
                del plant

        self.qt = self.qt_new
        self.states.append(self.qt)
