# Import libraries
import numpy as np
import pygame
from pygame import gfxdraw
import pickle

from plant import Plant
from rendering import pos_to_screen, r_to_screen, color_to_rgb

file_path = 'Data\states_rep_p1e-1.pkl'

# Load simulation states
states = pickle.load(
    open(file_path, 'rb'))
print(f'Loaded {len(states)} simulation states.')


def setup():
    global screen, boundary_scale, frame, fps, clock, qt_show_kwargs
    screen_width, screen_height = 700, 700
    boundary_scale = 0.8
    frame = 0

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(
        (screen_width, screen_height))
    fps = 60
    clock = pygame.time.Clock()
    qt_show_kwargs = {
        'boundary_scale': 0.8,
        'show_children': True,
        'show_center': False,
        'show_points': False,
        'show_boundary': True
    }


color_background = (235, 235, 205)


def draw_loading_progress():
    screen.fill(color_background)
    font = pygame.font.Font(None, 36)
    text = font.render(
        f'Loading... {total_states} states loaded', True, (255, 255, 255))
    text_rect = text.get_rect(
        center=(screen.get_width() / 2, screen.get_height() / 2))
    screen.blit(text, text_rect)
    pygame.display.flip()


def draw():
    # Draw the background as a light grey with a yelowish tint
    screen.fill(color_background)

    # Draw the bounding box
    qt = states[frame]
    # qt.show(screen, **qt_show_kwargs)

    for point in qt.all_points():
        plant = point.data
        p_screen = pos_to_screen(screen, plant.pos, boundary_scale)
        r_screen = r_to_screen(screen, plant.r, boundary_scale)
        c_screen = plant.color

        if 0 <= p_screen[0] < screen.get_width() and 0 <= p_screen[1] < screen.get_height() and r_screen > 0:
            gfxdraw.filled_circle(
                screen, p_screen[0], p_screen[1], r_screen, c_screen)
            gfxdraw.aacircle(
                screen, p_screen[0], p_screen[1], r_screen, c_screen)


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
        frame += 1  # Update frame
        frame %= len(states)  # Loop back to the first frame

        # Control the frame rate
        clock.tick(fps)

except SystemExit:
    pass
finally:
    pygame.quit()
