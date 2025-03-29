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

def find_all_lines_key_values(f, key, vals, assume_sorted=False):
    # Ensure vals is a set for faster lookup
    vals_set = set(vals)
    min_val = min(vals)
    max_val = max(vals)

    # Find which column the key is in
    f.seek(0)
    line = f.readline().decode('utf-8').strip()
    print(f'files.find_all_lines_key_values(): Found keys: {line}')
    keys = line.split(',')
    print(f'files.find_all_lines_key_values(): Found keys: {keys}')
    key_col = keys.index(key)
    
    # Create a list to store the lines
    lines = []
    
    if assume_sorted:
        first_line_with_value = find_first_value_sorted(f, key, min_val)
        val = min_val
        # Loop back through the lines from the first line with the value
        while val == min_val:
            prev_line_num = f.tell()
            move_to_previous_line(f)
            line = f.readline().decode('utf-8')
            try:
                val = float(line.split(',')[key_col])
            except ValueError:
                break
            move_to_previous_line(f)
            print(f'files.find_all_lines_key_values(): Found value at stream position {f.tell()}: \'{key}\' = {val}', end=' '*10 + '\r')
        
        # If the value is less than the minimum value, move the pointer to the next line
        if val < min_val:
            f.seek(prev_line_num)  
        print()       
        
    # Loop through the remaining lines in the file
    for i, line in enumerate(f):
        # Check if the line contains any of the values
        line = line.decode('utf-8')
        val = float(line.split(',')[key_col])
        if val in vals_set:
            print(f'files.find_all_lines_key_values(): Found value at stream position {i}: \'{key}\' = {val}', end=' '*10 + '\r')
            lines.append(line)
        else:
            print(f'files.find_all_lines_key_values(): Searching for value at stream position {i}', end=' '*10 + '\r') 
        if assume_sorted and val > max_val:
            print()
            print(f'files.find_all_lines_key_values(): Assumed sorted, stopping search at line {i}, val > max_val: {val} > {max_val}')
            break
    return lines

def find_first_value_sorted(f, key, val):
    # Find which column the key is in
    f.seek(0)
    line = f.readline().decode('utf-8').strip()
    keys = line.split(',')
    key_col = keys.index(key)
    
    found_val = None
    start_line = 0
    
    # Find the line number of the last line in the file
    f.seek(0, os.SEEK_END)
    end_line = f.tell()
    
    while found_val != val and start_line < end_line:
        # Find the line number and value in the middle line
        f.seek((start_line + end_line) // 2)
        move_to_previous_line(f)
        line = f.readline().decode('utf-8')
        found_val = float(line.split(',')[key_col])
        
        # If the value is less than the target value, move the start line to the middle line
        if found_val < val:
            start_line = f.tell()
        # If the value is greater than the target value, move the end line to the middle line
        elif found_val > val:
            end_line = f.tell()
        # If the value is equal to the target value, return the line
        else:
            break
        print(f'files.find_first_value_sorted(): Interval: {start_line = }, {found_val = }, {end_line = }', end=' '*10 + '\r')
    
    print(f'files.find_first_value_sorted(): Found value: {found_val} at line {f.tell()}' + ' '*10)
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