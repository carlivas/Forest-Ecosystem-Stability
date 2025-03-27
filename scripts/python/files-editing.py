import os
folder = 'Data/baseline/L2000'
# Path to the folder containing the buffers

alias1 = 'state_buffer_baseline_L2000_P5e_01'
alias2 = 'state_buffer-baseline_L2000_P5e-01'


file1 = folder + '/' + alias1 + '.csv'
file2 = folder + '/' + alias2 + '.csv'
response = input(f"This script will append the content of '{file2}' to '{file1}' and remove the content of '{file2}'. Do you want to continue? (Y/n): ").lower()
if response != 'y':
    print("Operation cancelled.")
else:
    # For each line in file2 that isn't the header, write the line at the end of file1 and remove it from file2
    with open(file2, 'r+') as f2, open(file1, 'a') as f1:
        lines = f2.readlines()
        f2.seek(0)  # Move to the start of file2 to overwrite
        f2.truncate()  # Clear file2
        f2.write(lines[0])  # Write back the header
        for i, line in enumerate(lines[1:]):  # Skip the header
            print(f'Processing line {i+1}/{len(lines[1:])}')
            f1.write(line)
