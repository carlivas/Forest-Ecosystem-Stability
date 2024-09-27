# Import libraries
import sys
import numpy as np
import pygame
from pygame import gfxdraw
import pickle

# import quadS
from scipy.spatial import KDTree
from rendering import pos_to_screen, screen_to_pos, r_to_screen, color_to_rgb, show_points, show_circle


points = []
plants = []

for i in range(6250):
    pos = np.random.uniform(-0.5, 0.5, 2)
    data = i
    points.append(pos)
    plants.append(data)

# kwargs = {'capacity': 1}
# qt = quadS.QuadTree(points, **kwargs)
qt = KDTree(points)


def setup():
    global screen, boundary_scale, frame, fps, clock
    # Initialize Pygame & set up the display
    screen_width, screen_height = 700, 700
    boundary_scale = 0.8
    frame = 0

    screen = pygame.display.set_mode(
        (screen_width, screen_height))

    # Set the desired frame rate (e.g., 30 FPS)
    fps = 30
    clock = pygame.time.Clock()


def draw():
    global qt
    # Draw the background as a light grey with a yelowish tint
    background = (255, 255, 255)
    screen.fill(background)

    # Draw the quad tree
    qt_show_kwargs = {
        'color': (0, 0, 0),
        'color_point': (0, 0, 255),
        'boundary_scale': boundary_scale,
        'line_width': 1,
        'point_size': 2,
        'center_size': 2,
        'show_center': True
    }

    for point in points:
        point[0] += np.random.uniform(-1, 1)*0.002
        point[1] += np.random.uniform(-1, 1)*0.002
    # qt = quadS.QuadTree(points)
    qt = KDTree(points)
    show_points(screen, points, **qt_show_kwargs)

    bb_show_kwargs = {
        'color': (255, 0, 255),
        'show_center': True,
        'boundary_scale': boundary_scale,
        'line_width': 1
    }
    if pygame.mouse.get_pos() is not None:
        # Draw a boundary box around the mouse cursor
        mouse_pos_screen = np.array(pygame.mouse.get_pos())
        mouse_pos = screen_to_pos(screen, mouse_pos_screen, boundary_scale)

        bb_r = 0.01
        bb_c = mouse_pos
        # mouse_bb = quadS.BoundingCircle(bb_c, bb_r)
        indices = qt.query_ball_point(x=bb_c, r=bb_r, workers=-1)
        points_in_bb = [points[i] for i in indices]
       # Use a single draw call for all points
        points_screen = [pos_to_screen(
            screen, point, boundary_scale) for point in points_in_bb]
        for p_screen in points_screen:
            gfxdraw.aacircle(
                screen, p_screen[0], p_screen[1], 5, bb_show_kwargs['color'])
            gfxdraw.filled_circle(
                screen, p_screen[0], p_screen[1], 5, bb_show_kwargs['color'])
        # mouse_bb.show(screen, **bb_show_kwargs)
        show_circle(bb_c, bb_r, screen, **bb_show_kwargs)


        # Main loop
setup()

running = True
try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        draw()

        pygame.display.flip()
        clock.tick(fps)

except SystemExit:
    pass
finally:
    pygame.quit()
