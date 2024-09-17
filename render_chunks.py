# Import libraries
import numpy as np
import pygame
from pygame import gfxdraw
import pickle

from plant import Plant
from rendering import pos_to_screen, r_to_screen, color_to_rgb


def load_metadata(file_path):
    with open(file_path, 'rb') as file:
        metadata = pickle.load(file)
    return metadata


def load_states_in_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as file:
        while True:
            try:
                chunk = pickle.load(file)
                for i in range(0, len(chunk), chunk_size):
                    yield chunk[i:i + chunk_size]
            except EOFError:
                break


def setup():
    global screen, boundary_scale, x_offset, y_offset, speed_move, speed_zoom, frame, fps, clock, qt_show_kwargs
    screen_width, screen_height = 700, 700
    boundary_scale = 0.8
    x_offset, y_offset = 0, 0
    speed_move = 50
    speed_zoom = 0.2
    frame = 0

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(
        (screen_width, screen_height))
    fps = 30
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
    font = pygame.font.Font(None, 24)
    text = font.render(
        f'Loaded {last_chunk_loaded} / {n_iter} ({last_chunk_loaded * 100 // n_iter} %) simulation states.', True, (0, 0, 0))

    progress_bar_width = 300
    progress_bar_height = 20
    progress = last_chunk_loaded / n_iter

    # Draw the filled portion of the progress bar
    pygame.draw.rect(screen, (0, 128, 0),
                     (screen.get_width() / 2 - progress_bar_width / 2,
                      screen.get_height() / 2 - progress_bar_height / 2,
                      progress_bar_width * progress, progress_bar_height))

    # Draw the outline of the progress bar
    pygame.draw.rect(screen, (0, 0, 0),
                     (screen.get_width() / 2 - progress_bar_width / 2,
                      screen.get_height() / 2 - progress_bar_height / 2,
                      progress_bar_width, progress_bar_height), 2)

    text_rect = text.get_rect(
        center=(screen.get_width() / 2, screen.get_height() / 2 + 25))
    screen.blit(text, text_rect)


def draw():
    # Draw the background as a light grey with a yelowish tint
    screen.fill(color_background)

    # Draw the bounding box
    qt = states[frame]
    # qt.show(screen, **qt_show_kwargs)

    for point in qt.all_points():
        plant = point.data
        p_screen = pos_to_screen(
            screen, plant.pos, boundary_scale, x_offset, y_offset)
        r_screen = r_to_screen(screen, plant.r, boundary_scale)
        c_screen = plant.color

        if 0 <= p_screen[0] < screen.get_width() and 0 <= p_screen[1] < screen.get_height() and r_screen > 0:
            gfxdraw.filled_circle(
                screen, p_screen[0], p_screen[1], r_screen, c_screen)
            gfxdraw.aacircle(
                screen, p_screen[0], p_screen[1], r_screen, c_screen)


surfix = 'temp'
file_path_metadata = f'Data\metadata_{surfix}.pkl'
file_path_states = f'Data\states_{surfix}.pkl'

metadata = load_metadata(file_path_metadata)

total_states = n_iter = metadata['n_iter']
chunk_size = metadata['chunk_size']

# Initialize an empty list to hold all the QuadTree objects
states = []

# Main loop
setup()

loading = True
running = True
try:
    while loading:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loading = False
                running = False
        # Load the data in chunks and extend the states list
        i = 1
        for chunk in load_states_in_chunks(file_path_states, chunk_size):
            last_chunk_loaded = min(chunk_size * i, n_iter)
            states.extend(chunk)

            draw_loading_progress()
            i += 1
            pygame.display.flip()

        if last_chunk_loaded == n_iter:
            loading = False
            running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                loading = False
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    y_offset += speed_move
                elif event.key == pygame.K_s:
                    y_offset -= speed_move
                elif event.key == pygame.K_a:
                    x_offset += speed_move
                elif event.key == pygame.K_d:
                    x_offset -= speed_move

                if event.key == pygame.K_z:
                    boundary_scale -= speed_zoom
                elif event.key == pygame.K_x:
                    boundary_scale += speed_zoom
        draw()

        frame += 1  # Update frame
        frame %= len(states)  # Loop back to the first frame
        # Control the frame rate
        clock.tick(fps)
        pygame.display.flip()

except SystemExit:
    pass
finally:
    pygame.quit()
