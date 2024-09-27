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
    global screen, boundary_scale, x_offset, y_offset, speed_move, speed_zoom, frame, frame_dummy, fps, fps_max, fps_divisor, show_collisions, show_generation, clock, qt_show_kwargs, running, animating
    screen_width, screen_height = 700, 700

    boundary_scale = 0.8
    x_offset, y_offset = 0, 0
    speed_move = 50
    speed_zoom = 0.2
    frame = 0
    frame_dummy = 0
    fps = 240
    fps_max = 240
    fps_divisor = 6

    show_collisions = False
    show_generation = False

    pygame.init()
    pygame.font.init()

    screen = pygame.display.set_mode(
        (screen_width, screen_height))

    clock = pygame.time.Clock()
    qt_show_kwargs = {
        'boundary_scale': boundary_scale,
        'show_children': False,
        'show_center': False,
        'show_points': False,
        'show_boundary': True
    }

    running = True
    animating = True


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


def draw_buttons():
    font = pygame.font.Font(None, 24)
    start_text = font.render('>', True, (0, 0, 0))
    stop_text = font.render('||', True, (0, 0, 0))
    speed_up_text = font.render('+', True, (0, 0, 0))
    slow_down_text = font.render('-', True, (0, 0, 0))
    forward_text = font.render('>>', True, (0, 0, 0))
    backward_text = font.render('<<', True, (0, 0, 0))
    reset_text = font.render('0 <-', True, (0, 0, 0))

    width_screen, height_screen = screen.get_width(), screen.get_height()
    width_button, height_button = 50, 25
    margin = 10

    h = height_screen - height_button - margin
    start_stop_button = pygame.Rect(
        width_screen/2 - 0.5*width_button, h, width_button, height_button)
    slow_down_button = pygame.Rect(
        width_screen/2 - 1.5*width_button, h, width_button, height_button)
    speed_up_button = pygame.Rect(
        width_screen/2 + 0.5*width_button, h, width_button, height_button)
    forward_button = pygame.Rect(
        width_screen/2 + 1.5*width_button, h, width_button, height_button)
    backward_button = pygame.Rect(
        width_screen/2 - 2.5*width_button, h, width_button, height_button)
    reset_button = pygame.Rect(
        width_screen - width_button - margin, margin, width_button, height_button)

    if animating == False:
        pygame.draw.rect(screen, (220, 255, 220), start_stop_button)
        start_text_rect = start_text.get_rect(center=start_stop_button.center)
        screen.blit(start_text, start_text_rect)
    if animating == True:
        pygame.draw.rect(screen, (255, 220, 220), start_stop_button)
        stop_text_rect = stop_text.get_rect(center=start_stop_button.center)
        screen.blit(stop_text, stop_text_rect)

    pygame.draw.rect(screen, (255, 255, 255), speed_up_button)
    speed_up_text_rect = speed_up_text.get_rect(center=speed_up_button.center)
    screen.blit(speed_up_text, speed_up_text_rect)

    pygame.draw.rect(screen, (255, 255, 255), slow_down_button)
    slow_down_text_rect = slow_down_text.get_rect(
        center=slow_down_button.center)
    screen.blit(slow_down_text, slow_down_text_rect)

    pygame.draw.rect(screen, (255, 255, 255), forward_button)
    forward_text_rect = forward_text.get_rect(center=forward_button.center)
    screen.blit(forward_text, forward_text_rect)

    pygame.draw.rect(screen, (255, 255, 255), backward_button)
    backward_text_rect = backward_text.get_rect(center=backward_button.center)
    screen.blit(backward_text, backward_text_rect)

    pygame.draw.rect(screen, (255, 255, 255), reset_button)
    reset_text_rect = reset_text.get_rect(center=reset_button.center)
    screen.blit(reset_text, reset_text_rect)

    return start_stop_button, speed_up_button, slow_down_button, forward_button, backward_button, reset_button


def draw_simulation():
    # Draw the background as a light grey with a yelowish tint
    screen.fill(color_background)

    # Draw the bounding box
    qt = states[frame_dummy]
    qt.show(screen, **qt_show_kwargs)

    for point in qt.all_points():
        plant = point.data
        p_screen = pos_to_screen(
            screen, plant.pos, boundary_scale, x_offset, y_offset)
        r_screen = r_to_screen(screen, plant.r, boundary_scale)
        c_screen = plant.color
        if show_collisions:
            if plant.is_colliding:
                c_screen = (255, 0, 0)

        if 0 <= p_screen[0] < screen.get_width() and 0 <= p_screen[1] < screen.get_height() and r_screen > 0:
            gfxdraw.filled_circle(
                screen, p_screen[0], p_screen[1], r_screen, c_screen)
            gfxdraw.aacircle(
                screen, p_screen[0], p_screen[1], r_screen, c_screen)
            if show_generation:
                font = pygame.font.Font(None, r_screen)
                text = font.render(
                    str(plant.generation), True, (0, 0, 0))
                text_rect = text.get_rect(center=p_screen)
                screen.blit(text, text_rect)
    font = pygame.font.Font(None, 24)
    text = font.render(f't = {frame_dummy}', True, (0, 0, 0))
    screen.blit(text, (10, 10))


prefix = 'sim6'
file_path_metadata = f'Data\{prefix}_metadata.pkl'
file_path_states = f'Data\{prefix}_states.pkl'

metadata = load_metadata(file_path_metadata)

total_states = n_iter = metadata['n_iter']
chunk_size = metadata['chunk_size']

# Initialize an empty list to hold all the QuadTree objects
states = []

# Main loop
setup()

loading = True
running = False
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
                qt_show_kwargs['boundary_scale'] = boundary_scale
                qt_show_kwargs['x_offset'] = x_offset
                qt_show_kwargs['y_offset'] = y_offset

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                if start_stop_button.collidepoint(mouse_pos):
                    animating = not animating

                elif speed_up_button.collidepoint(mouse_pos):
                    fps_divisor /= 2
                    fps_divisor = max(1, fps_divisor)

                elif slow_down_button.collidepoint(mouse_pos):
                    fps_divisor *= 2
                    fps_divisor = min(fps_max, fps_divisor)

                elif forward_button.collidepoint(mouse_pos):
                    frame_dummy = (frame_dummy + 1) % len(states)

                elif backward_button.collidepoint(mouse_pos):
                    frame_dummy = (frame_dummy - 1) % len(states)

                elif reset_button.collidepoint(mouse_pos):
                    frame_dummy = 0

        draw_simulation()
        start_stop_button, speed_up_button, slow_down_button, forward_button, backward_button, reset_button = draw_buttons()

        if animating and frame % fps_divisor == 0:
            frame_dummy += 1  # Update frame
            frame_dummy %= len(states)  # Loop back to the first frame

        frame += 1
        clock.tick(fps)
        pygame.display.flip()

except SystemExit:
    pass
finally:
    pygame.quit()
