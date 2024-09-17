from typing import Optional
import pygame
import numpy as np
import numba


@numba.jit(nopython=True)
def pos_to_screen_jit(screen_width: int, screen_height: int, pos: np.ndarray, boundary_scale: float = 1.0, x_offset: int = 0, y_offset: int = 0) -> np.ndarray:
    screen_center = np.array(
        [screen_width / 2 + x_offset, screen_height / 2 + y_offset])
    screen_x = int(screen_center[0] + pos[0] * screen_width * boundary_scale)
    screen_y = int(screen_center[1] - pos[1] * screen_height * boundary_scale)
    return np.array([screen_x, screen_y])


@numba.jit(nopython=True)
def screen_to_pos_jit(screen_width: int, screen_height: int, screen_pos: np.ndarray, boundary_scale: float = 1.0) -> np.ndarray:
    screen_center = np.array([screen_width / 2, screen_height / 2])
    pos_x = (screen_pos[0] - screen_center[0]) / \
        (screen_width * boundary_scale)
    pos_y = (screen_center[1] - screen_pos[1]) / \
        (screen_height * boundary_scale)
    return np.array([pos_x, pos_y])


# Wrapper functions to extract screen dimensions and call the Numba-optimized functions
def pos_to_screen(screen: pygame.Surface, pos: np.ndarray, boundary_scale: Optional[float] = 1.0, x_offset: Optional[int] = 0, y_offset: Optional[int] = 0) -> np.ndarray:
    screen_width, screen_height = screen.get_width(), screen.get_height()
    return pos_to_screen_jit(screen_width, screen_height, pos, boundary_scale, x_offset, y_offset)


def screen_to_pos(screen: pygame.Surface, screen_pos: np.ndarray, boundary_scale: Optional[float] = 1.0) -> np.ndarray:
    screen_width, screen_height = screen.get_width(), screen.get_height()
    return screen_to_pos_jit(screen_width, screen_height, screen_pos, boundary_scale)


def r_to_screen(screen: pygame.Surface, r: float, boundary_scale: float):
    return int(r * screen.get_width() * boundary_scale)


def color_to_rgb(color):
    return (color[0] * 255, color[1] * 255, color[2] * 255)
