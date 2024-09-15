from typing import Optional
import numpy as np
import numba
import pygame

import rendering


class Point:
    def __init__(self, pos: np.ndarray, data: Optional[dict] = None):
        self.x = pos[0]
        self.y = pos[1]
        self.data = data

    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y and self.data == other.data
        return False

    def __hash__(self):
        return hash((self.x, self.y))

    def to_array(self):
        return np.array([self.x, self.y])


@numba.jit(nopython=True)
def contains_jit(boundary_center, boundary_half_width, boundary_half_height, point):
    return (boundary_center[0] - boundary_half_width <= point[0] <= boundary_center[0] + boundary_half_width and boundary_center[1] - boundary_half_height <= point[1] <= boundary_center[1] + boundary_half_height)


@numba.jit(nopython=True)
def intersects_jit(boundary_center, boundary_half_width, boundary_half_height, other_center, other_half_width, other_half_height):
    return not (other_center[0] - other_half_width > boundary_center[0] + boundary_half_width or
                other_center[0] + other_half_width < boundary_center[0] - boundary_half_width or
                other_center[1] - other_half_height > boundary_center[1] + boundary_half_height or
                other_center[1] + other_half_height < boundary_center[1] - boundary_half_height)


# Axis-aligned bounding box with half dimension and center
class BoundingBox():
    def __init__(self, center: np.ndarray, half_width: float, half_height: Optional[float] = None):
        self.center = center
        self.half_width = half_width
        self.half_height = half_height if half_height is not None else half_width
        self.area = 4 * half_width * half_height

    def contains(self, point: Point) -> bool:
        return contains_jit(self.center, self.half_width, self.half_height, point.to_array())

    def intersects(self, other: 'BoundingBox') -> bool:
        return intersects_jit(self.center, self.half_width, self.half_height, other.center, other.half_width, other.half_height)

    def show(self, screen: pygame.Surface, **kwargs):
        color = kwargs.get('color', (0, 0, 0))
        boundary_scale = kwargs.get('boundary_scale', 1.0)
        line_width = kwargs.get('line_width', 1)

        # Get the screen coordinates of the center
        screen_center = rendering.pos_to_screen(screen, self.center)

        bb_top_left = np.array([self.center[0] - self.half_width,
                                self.center[1] + self.half_height])
        screen_top_left = rendering.pos_to_screen(
            screen, bb_top_left, boundary_scale)

        # Get the screen dimensions of the bounding box
        width = 2 * self.half_width * screen.get_width() * boundary_scale
        height = 2 * self.half_height * screen.get_height() * boundary_scale

        # Draw the bounding box
        bb_rect = pygame.Rect(
            screen_top_left[0], screen_top_left[1], width, height)
        pygame.draw.rect(
            screen, color, bb_rect, line_width)


class QuadTree():
    def __init__(self, center: np.ndarray, half_width: float, half_height: float, capacity: int):
        self.boundary = BoundingBox(center, half_width, half_height)
        self.capacity = capacity
        self.points = []
        self.children = []
        self.divided = False

    def subdivide(self):
        center = self.boundary.center
        half_width = self.boundary.half_width
        half_height = self.boundary.half_height
        quarter_width = half_width / 2
        quarter_height = half_height / 2

        q1 = QuadTree(np.array([center[0] + quarter_width,
                                center[1] + quarter_height]),
                      quarter_width,
                      quarter_height,
                      self.capacity)

        q2 = QuadTree(np.array([center[0] - quarter_width,
                                center[1] + quarter_height]),
                      quarter_width,
                      quarter_height,
                      self.capacity)

        q3 = QuadTree(np.array([center[0] - quarter_width,
                                center[1] - quarter_height]),
                      quarter_width,
                      quarter_height,
                      self.capacity)

        q4 = QuadTree(np.array([center[0] + quarter_width,
                                center[1] - quarter_height]),
                      quarter_width,
                      quarter_height,
                      self.capacity)

        self.children = [q1, q2, q3, q4]
        self.divided = True

    def insert(self, point: Point) -> bool:
        if not self.boundary.contains(point):
            return False

        if len(self.points) < self.capacity:
            self.points.append(point)
            return True

        else:
            if not self.divided:
                self.subdivide()
            for child in self.children:
                if child.insert(point):
                    return True

        # If the point cannot be inserted, return False
        return False

    def remove(self, point: Point) -> bool:
        if not self.boundary.contains(point):
            return False

        if point in self.points:
            self.points.remove(point)
            return True
        else:
            if self.divided:
                for child in self.children:
                    if child.remove(point):
                        return True
        return False

    def query(self, boundary: BoundingBox, found: Optional[list] = None) -> list:
        if found is None:
            found = []

        if self.boundary.intersects(boundary):
            for point in self.points:
                if boundary.contains(point):
                    found.append(point)

            if self.divided:
                for child in self.children:
                    child.query(boundary, found)

        return found

    def all_points(self) -> list[Point]:
        all_points = self.points.copy()
        if self.divided:
            for child in self.children:
                all_points.extend(child.all_points())
        return all_points

    def show(self, screen: pygame.Surface, **kwargs):
        color = kwargs.get('color', (0, 0, 0))
        boundary_scale = kwargs.get('boundary_scale', 1.0)
        point_size = kwargs.get('point_size', 1)
        line_width = kwargs.get('line_width', 1)
        show_points = kwargs.get('show_points', True)
        show_boundary = kwargs.get('show_boundary', True)
        show_center = kwargs.get('show_center', True)

        if show_boundary:
            # Draw the boundary box
            self.boundary.show(screen, **kwargs)

        if show_points:
            for point in self.points:
                # Draw the points
                p_screen = rendering.pos_to_screen(
                    screen, point.to_array(), boundary_scale)
                pygame.draw.circle(screen, color, p_screen, point_size)

        if show_center:
            # Draw the center
            screen_center = rendering.pos_to_screen(
                screen, self.boundary.center, boundary_scale)

            pygame.draw.line(
                screen, color, (screen_center[0] - 2 * point_size, screen_center[1]), (screen_center[0] + 2 * point_size, screen_center[1]), line_width)
            pygame.draw.line(
                screen, color, (screen_center[0], screen_center[1] - 2 * point_size), (screen_center[0], screen_center[1] + 2 * point_size), line_width)

        # Recursively draw the children
        if self.divided:
            for child in self.children:
                child.show(screen, **kwargs)
