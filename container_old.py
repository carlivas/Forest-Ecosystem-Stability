import numpy as np
import quads


def dist_sq(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2)


def check_collision(p1, p2):
    return dist_sq(p1, p2) < (p1.r + p2.r) ** 2


class Container(quads.QuadTree):
    def __init__(self, center: np.ndarray, half_width: float, half_height: float):
        super().__init__(center, 2*half_width, 2*half_height)
        self.area = 2*half_width * 2*half_height

    def update(self, plants: list[quads.Point]):
        center = self.center
        width = self.width
        height = self.height
        self = quads.QuadTree(center=center, width=width, height=height)

        for plant in plants:
            self.insert(quads.Point(tuple(plant.pos), data=plant))
        return self

    def is_within_bounds(self, pos):
        x, y = pos
        return x >= self.center.x - self.width/2 and x <= self.center.x + self.width/2 and y >= self.center.y - self.height/2 and y <= self.center.y + self.height/2

    def create_bb(self, pos, width, height=None):
        if height is None:
            height = width
        xmin = pos[0] - width/2
        xmax = pos[0] + width/2
        ymin = pos[1] - height/2
        ymax = pos[1] + height/2

        bb = quads.BoundingBox(xmin, ymin, xmax, ymax)
        return bb

    def get_collisions(self, plant, r_max_global):
        collisions = []
        bb = self.create_bb(plant.pos, 2*(plant.r + r_max_global))
        for point in self.nearest_neighbors(quads.Point(plant.pos[0], plant.pos[1])):
            other_plant = point.data
            if other_plant is not plant:
                if check_collision(plant, other_plant):
                    collisions.append(other_plant)
        return collisions
