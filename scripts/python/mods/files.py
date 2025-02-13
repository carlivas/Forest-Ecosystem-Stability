import os


def move_to_previous_line(f):
    # MOVES THE POINTER TO THE START OF THE PREVIOUS LINE

    # Move the file pointer two bytes back from the current position
    if f.tell() == 0:
        f.seek(-2, os.SEEK_END)
    else:
        f.seek(-2, os.SEEK_CUR)

    # Loop until the start of the current line is reached
    while f.read(1) != b'\n' and f.tell() > 1:
        # Move the file pointer two bytes back from the current position
        f.seek(-2, os.SEEK_CUR)
