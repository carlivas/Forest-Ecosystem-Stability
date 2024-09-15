from typing import Optional
import numpy as np
import quadT


def dist_sq(p1, p2):
    return np.sum((p1.pos - p2.pos) ** 2)


def check_collision(p1, p2):
    return dist_sq(p1, p2) < (p1.r + p2.r) ** 2


class Container(quadT.QuadTree):
    def __init__(self, center: np.ndarray, half_width: float, half_height: float, capacity: int):
        super().__init__(center, half_width, half_height, capacity)
        self.center = center
        self.half_width = half_width
        self.half_height = half_height
        self.area = half_width * 2 * half_height * 2

        self.bb = quadT.BoundingBox(center, half_width, half_height)

        self.capacity = capacity

    def update(self, plants):
        self = quadT.QuadTree(
            center=self.center, half_width=self.half_width, half_height=self.half_height, capacity=self.capacity)

        for plant in plants:
            self.insert(quadT.Point(plant.pos, data=plant))
        return self

    def is_within_bounds(self, pos):
        return self.bb.contains(quadT.Point(pos))

    # def create_bb(self, center: np.ndarray, half_width: float, half_height: Optional[float] = None):
    #     if half_height is None:
    #         half_height = half_width
    #     return quadT.BoundingBox(center, half_width, half_height)

    def get_collisions(self, plant, r_max_global):
        collisions = []
        bb = quadT.BoundingBox(plant.pos, 2*(plant.r + r_max_global))
        for point in self.query(bb):
            other_plant = point.data
            if other_plant is not plant:
                if check_collision(plant, other_plant):
                    collisions.append(other_plant)
        return collisions

    def copy(self):
        return Container(self.center, self.half_width, self.half_height, self.capacity)
