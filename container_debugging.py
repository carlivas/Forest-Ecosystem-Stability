# Import libraries
import sys
import numpy as np
import pygame
from pygame import gfxdraw
import pickle

from container import Container
import quadT
from rendering import pos_to_screen, screen_to_pos, color_to_rgb


ct = Container(np.array([0, 0]), 0.5, 0.5, 4)

for i in range(2000):
    pos = np.random.rand(2) - 0.5
    ct.insert(quadT.Point(pos))


print(ct.__dict__)


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
    # Draw the background as a light grey with a yelowish tint
    background = (255, 255, 255)
    screen.fill(background)

    # Draw the quad tree
    show_kwargs = {
        'color': (255, 0, 0),
        'boundary_scale': boundary_scale,
        'line_width': 1,
        'show_center': False
    }
    ct.show(screen, **show_kwargs)

    if pygame.mouse.get_pos() is not None:
        # Draw a boundary box around the mouse cursor
        mouse_pos_screen = np.array(pygame.mouse.get_pos())
        mouse_pos = screen_to_pos(screen, mouse_pos_screen, boundary_scale)
        mouse_bb = quadT.BoundingBox(mouse_pos, 0.1, 0.1)
        points_in_mouse_bb = ct.query(mouse_bb, [])
        for point in points_in_mouse_bb:
            point_screen = pos_to_screen(
                screen, point.to_array(), boundary_scale)
            gfxdraw.aacircle(
                screen, point_screen[0], point_screen[1], 5, (0, 255, 0))
            gfxdraw.filled_circle(
                screen, point_screen[0], point_screen[1], 5, (0, 255, 0))
        mouse_bb.show(screen, color=(0, 255, 0),
                      boundary_scale=boundary_scale, line_width=1)


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
