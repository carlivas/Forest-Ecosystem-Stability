# Import libraries
import sys
import numpy as np
import pygame
from pygame import gfxdraw
import pickle

from plant import Plant
from rendering import pos_to_screen, r_to_screen, color_to_rgb

# Load simulation states
simulation_states_dicts = pickle.load(
    open('Data\simulation_states_temp.pkl', 'rb'))
simulation_states = [[Plant(**state) for state in states]
                     for states in simulation_states_dicts]
print(f'Loaded {len(simulation_states)} simulation states.')


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
    background = (235, 235, 205)
    screen.fill(background)

    # Draw the bounding box
    screen_center = pos_to_screen(screen, np.array([0,0]))
    bb_top_left = np.array([self.center[0] - self.half_width,
                            self.center[1] + self.half_height])
    screen_top_left = pos_to_screen(
        screen, bb_top_left, boundary_scale)
    width = 2 * self.half_width * screen.get_width() * boundary_scale
    height = 2 * self.half_height * screen.get_height() * boundary_scale
    bb_rect = pygame.Rect(
        screen_top_left[0], screen_top_left[1], width, height)
    pygame.draw.rect(
        screen, color, bb_rect, line_width)

    state = simulation_states[frame]
    for plant in state:
        p_screen = pos_to_screen(screen, plant.pos)
        r_screen = r_to_screen(plant.r, boundary_scale)
        c_screen = color_to_rgb(plant.color)
        gfxdraw.filled_circle(
            screen, p_screen[0], p_screen[1], r_screen, c_screen)
        gfxdraw.aacircle(screen, p_screen[0], p_screen[1], r_screen, c_screen)


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
        frame %= len(simulation_states)  # Loop back to the first frame

        # Control the frame rate
        clock.tick(fps)

except SystemExit:
    pass
finally:
    pygame.quit()
