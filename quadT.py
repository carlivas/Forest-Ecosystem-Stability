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


# Custom clipping function for Numba compatibility
@numba.jit(nopython=True)
def custom_clip(value, min_value, max_value):
    if value < min_value:
        return min_value
    elif value > max_value:
        return max_value
    else:
        return value


# Axis-aligned bounding box with half dimension and center
class BoundingBox:
    def __init__(self, center: np.ndarray, half_width: float, half_height: Optional[float] = None):
        if half_height is None:
            half_height = half_width
        self.center = center
        self.half_width = half_width
        self.half_height = half_height
        self.area = 4 * self.half_width * self.half_height

    @staticmethod
    @numba.jit(nopython=True)
    def contains_box_jit(center, half_width, half_height, point):
        return (center[0] - half_width <= point[0] <= center[0] + half_width and
                center[1] - half_height <= point[1] <= center[1] + half_height)

    @staticmethod
    @numba.jit(nopython=True)
    def intersects_box_jit(center1, half_width1, half_height1, center2, half_width2, half_height2):
        return not (center2[0] - half_width2 > center1[0] + half_width1 or
                    center2[0] + half_width2 < center1[0] - half_width1 or
                    center2[1] - half_height2 > center1[1] + half_height1 or
                    center2[1] + half_height2 < center1[1] - half_height1)

    def contains_box(self, point: 'Point') -> bool:
        return self.contains_box_jit(self.center, self.half_width, self.half_height, point.to_array())

    def intersects_box(self, other: 'BoundingBox') -> bool:
        return self.intersects_box_jit(self.center, self.half_width, self.half_height, other.center, other.half_width, other.half_height)

    @staticmethod
    @numba.jit(nopython=True)
    def intersects_circle_jit(box_center, box_half_width, box_half_height, circle_center, circle_radius):
        closest_x = custom_clip(
            circle_center[0], box_center[0] - box_half_width, box_center[0] + box_half_width)
        closest_y = custom_clip(
            circle_center[1], box_center[1] - box_half_height, box_center[1] + box_half_height)
        distance_x = circle_center[0] - closest_x
        distance_y = circle_center[1] - closest_y
        return (distance_x**2 + distance_y**2) <= (circle_radius**2)

    def intersects_circle(self, circle: 'BoundingCircle') -> bool:
        return self.intersects_circle_jit(self.center, self.half_width, self.half_height, circle.center, circle.radius)

    def show(self, screen: pygame.Surface, **kwargs):
        boundary_scale = kwargs.get('boundary_scale', 1.0)
        line_width = kwargs.get('line_width', 1)
        point_size = kwargs.get('point_size', 1)
        center_size = kwargs.get('center_size', 1)

        show_center = kwargs.get('show_center', False)
        color = kwargs.get('color', (0, 0, 0))
        color_boundary = kwargs.get('color_boundary', color)
        color_center = kwargs.get('color_center', color)

        # Get the screen coordinates of the center
        screen_center = rendering.pos_to_screen(
            screen, self.center, boundary_scale)

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
            screen, color_boundary, bb_rect, line_width)

        if show_center:
            pygame.draw.line(
                screen, color_center, (screen_center[0] - center_size, screen_center[1]), (screen_center[0] + center_size, screen_center[1]), line_width)
            pygame.draw.line(
                screen, color_center, (screen_center[0], screen_center[1] - center_size), (screen_center[0], screen_center[1] + center_size), line_width)


class BoundingCircle:
    def __init__(self, center: np.ndarray, radius: float):
        self.center = center
        self.radius = radius
        self.area = np.pi * self.radius**2

    @staticmethod
    @numba.jit(nopython=True)
    def contains_circle_jit(center, radius, point):
        return np.sum((point - center)**2) <= radius**2

    @staticmethod
    @numba.jit(nopython=True)
    def intersects_circle_jit(center1, radius1, center2, radius2):
        return np.sum((center1 - center2)**2) <= (radius1 + radius2)**2

    def contains_circle(self, point: 'Point') -> bool:
        return self.contains_circle_jit(self.center, self.radius, point.to_array())

    def intersects_circle(self, other: 'BoundingCircle') -> bool:
        return self.intersects_circle_jit(self.center, self.radius, other.center, other.radius)

    def intersects_box(self, box: 'BoundingBox') -> bool:
        return box.intersects_circle(self)

    def show(self, screen: pygame.Surface, **kwargs):
        boundary_scale = kwargs.get('boundary_scale', 1.0)
        line_width = kwargs.get('line_width', 1)
        point_size = kwargs.get('point_size', 1)
        center_size = kwargs.get('center_size', 1)

        show_center = kwargs.get('show_center', False)
        color = kwargs.get('color', (0, 0, 0))
        color_boundary = kwargs.get('color_boundary', color)
        color_center = kwargs.get('color_center', color)

        # Get the screen coordinates of the center
        screen_center = rendering.pos_to_screen(
            screen, self.center, boundary_scale)

        # Get the screen radius of the circle
        screen_radius = rendering.r_to_screen(
            screen, self.radius, boundary_scale)

        # Draw the bounding circle
        pygame.draw.circle(screen, color_boundary, screen_center,
                           screen_radius, line_width)

        if show_center:
            pygame.draw.line(
                screen, color_center, (screen_center[0] - 2 * point_size, screen_center[1]), (screen_center[0] + 2 * point_size, screen_center[1]), line_width)
            pygame.draw.line(
                screen, color_center, (screen_center[0], screen_center[1] - 2 * point_size), (screen_center[0], screen_center[1] + 2 * point_size), line_width)


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
        if not self.boundary.contains_box(point):
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
        if not self.boundary.contains_box(point):
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

    def query(self, boundary, found: Optional[list] = None) -> list:
        if found is None:
            found = []

        if isinstance(boundary, BoundingBox):
            if self.boundary.intersects_box(boundary):
                for point in self.points:
                    if boundary.contains_box(point):
                        found.append(point)

                if self.divided:
                    for child in self.children:
                        child.query(boundary, found)
        elif isinstance(boundary, BoundingCircle):
            if self.boundary.intersects_circle(boundary):
                for point in self.points:
                    if boundary.contains_circle(point):
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
    
    def copy(self):
        qt = QuadTree(self.boundary.center, self.boundary.half_width, self.boundary.half_height, self.capacity)
        qt.points = self.points.copy()
        qt.divided = self.divided
        if self.divided:
            qt.children = [child.copy() for child in self.children]
        return qt

    def show(self, screen: pygame.Surface, **kwargs):
        boundary_scale = kwargs.get('boundary_scale', 1.0)
        point_size = kwargs.get('point_size', 1)

        show_boundary = kwargs.get('show_boundary', True)
        show_children = kwargs.get('show_children', True)
        show_points = kwargs.get('show_points', True)

        color = kwargs.get('color', (0, 0, 0))
        color_points = kwargs.get('color_point', color)

        if show_boundary:
            # Draw the boundary box
            self.boundary.show(screen, **kwargs)

        if show_points:
            for point in self.points:
                # Draw the points
                p_screen = rendering.pos_to_screen(
                    screen, point.to_array(), boundary_scale)
                pygame.draw.circle(screen, color_points, p_screen, point_size)

        if show_children:
            # Recursively draw the children
            if self.divided:
                for child in self.children:
                    child.show(screen, **kwargs)
