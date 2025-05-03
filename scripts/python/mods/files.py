import numpy as np
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

def find_all_lines_key_values_sorted(f, key, vals):
    # Ensure vals is a set for faster lookup
    vals_set = set(vals)
    
    # Find which column the key is in
    f.seek(0)
    line = f.readline().decode('utf-8').strip()
    print(f'files.find_all_lines_key_values_sorted(): Found keys: {line}')
    keys = line.split(',')
    print(f'files.find_all_lines_key_values_sorted(): Found keys: {keys}')
    key_col = keys.index(key)
    
    lines = []
    for val in vals:
        # Find the first line with the value
        line = find_first_value_sorted(f, key, val)
        print(f'files.find_all_lines_key_values_sorted(): Reading at stream position {f.tell()}: \'{key}\' = {val}', end=' '*10 + '\r')
        # Loop through the lines from the first line with the value
        f.seek(f.tell())
        while True:
            line = f.readline().decode('utf-8')
            if not line:
                break
            val_found = float(line.split(',')[key_col])
            if val_found != val:
                break
            print(f'files.find_all_lines_key_values_sorted(): Found value at stream position {f.tell()}: \'{key}\' = {val}', end=' '*10 + '\r')
            lines.append(line)
    return lines

# def find_all_lines_key_values(f, key, vals, assume_sorted=False):
#     # Ensure vals is a set for faster lookup
#     vals_set = set(vals)
#     min_val = min(vals)
#     max_val = max(vals)

#     # Find which column the key is in
#     f.seek(0)
#     line = f.readline().decode('utf-8').strip()
#     print(f'files.find_all_lines_key_values(): Found keys: {line}')
#     keys = line.split(',')
#     print(f'files.find_all_lines_key_values(): Found keys: {keys}')
#     key_col = keys.index(key)
    
#     # Create a list to store the lines
#     lines = []
    
#     if assume_sorted:
#         first_line_with_value = find_first_value_sorted(f, key, min_val)
        
#     # Loop through the remaining lines in the file
#     for i, line in enumerate(f):
#         # Check if the line contains any of the values
#         line = line.decode('utf-8')
#         val = float(line.split(',')[key_col])
#         if val in vals_set:
#             print(f'files.find_all_lines_key_values(): Found value at stream position {i}: \'{key}\' = {val}', end=' '*10 + '\r')
#             lines.append(line)
#         else:
#             print(f'files.find_all_lines_key_values(): Searching for value at stream position {i}', end=' '*10 + '\r') 
#         if assume_sorted and val > max_val:
#             print()
#             print(f'files.find_all_lines_key_values(): Assumed sorted, stopping search at line {i}, val > max_val: {val} > {max_val}')
#             break
#     return lines

def print_head(f, n=10):
    # PRINTS THE FIRST N LINES OF A FILE
    f.seek(0)
    for i in range(n):
        line = f.readline().decode('utf-8').strip()
        print(line)
        
def print_tail(f, n=10):
    # PRINTS THE LAST N LINES OF A FILE
    f.seek(0, os.SEEK_END)
    end_line = f.tell()
    f.seek(end_line - 1)
    for i in range(n):
        move_to_previous_line(f)
        line = f.readline().decode('utf-8').strip()
        print(line)
        if f.tell() == 0:
            break

def find_first_value_sorted(f, key, val):
    # Find which column the key is in
    f.seek(0)
    line = f.readline().decode('utf-8').strip()
    keys = line.split(',')
    key_col = keys.index(key)
    
    val_found = None
    start_line_bit = f.tell()
    
    # Find the line number of the last line in the file
    f.seek(0, os.SEEK_END)
    end_line_bit = f.tell()
    
    f.seek(start_line_bit)
    start_line = f.readline().decode('utf-8')
    val_at_start_line = float(start_line.split(',')[key_col])
    if val_at_start_line > val:
        print(f'files.find_first_value_sorted(): val_at_start_line > val: {val_at_start_line} > {val}')
        return None
    
    
    while val_found != val and start_line_bit < end_line_bit:    
        # Find the line number and value in the middle line
        f.seek((start_line_bit + end_line_bit) // 2)
        move_to_previous_line(f)
        mid_line = f.readline().decode('utf-8')
        val_found = float(mid_line.split(',')[key_col])
        
        # If the value is less than the target value, move the start line to the middle line
        if val_found < val:
            start_line_bit = f.tell()
        # If the value is greater than the target value, move the end line to the middle line
        elif val_found > val:
            end_line_bit = f.tell()
        # If the value is equal to the target value, return the line
        else:
            break
        # print(f'files.find_first_value_sorted(): Interval: {start_line = }, {val_found = }, {end_line = }', end=' '*10 + '\r')
        
        f.seek(start_line_bit)
        start_line = f.readline().decode('utf-8')
        if start_line != '':
            val_at_start_line = float(start_line.split(',')[key_col])
        else:
            val_at_start_line = -np.inf
        
        f.seek(end_line_bit)
        end_line = f.readline().decode('utf-8')
        if end_line != '':
            val_at_end_line = float(end_line.split(',')[key_col])
        else:
            val_at_end_line = np.inf
        
        if val_at_start_line > val:
            print(f'files.find_first_value_sorted(): val_at_start_line > val: {val_at_start_line} > {val}')
            return None
        elif val_at_end_line < val:
            print(f'files.find_first_value_sorted(): val_at_end_line < val: {val_at_end_line} < {val}')
            return None
        
    
    val = val_found
    # Loop back through the lines from the first line with the value
    while val_found == val:
        prev_line_num = f.tell()
        move_to_previous_line(f)
        line = f.readline().decode('utf-8')
        try:
            val_found = float(line.split(',')[key_col])
        except ValueError:
            break
        move_to_previous_line(f)
        print(f'files.find_first_value_sorted(): Found value at stream position {f.tell()}: \'{key}\' = {val}', end=' '*10 + '\r')
    # If the value is less than the minimum value, move the pointer to the next line
    if val_found < val:
        f.seek(prev_line_num)
    return line

# def read_lines(lines, f):
#     # READS SPECIFIC LINES FROM A FILE

#     # Create a list to store the lines
#     selected_lines = []

#     # Move the pointer to the start of the file
#     f.seek(0, os.SEEK_SET)

#     # Loop through each line in the file
#     for i, line in enumerate(f):
#         # Check if the line number is in the list of selected lines
#         if i in lines:
#             # Add the line to the list
#             selected_lines.append(line)

#     # Return the list of selected lines
#     return selected_lines