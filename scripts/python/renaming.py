import os
folder =  # Path to the folder containing the buffers

load_folder = os.path.abspath(folder)
print(f'\nload_folder: {load_folder}\n')

for root, dirs, files in os.walk(load_folder):
    print(f'root: {root}')
    print(f'dirs: {dirs}')
    print(f'files: {files}')
    print()
    for f in files:
        new_name = f.replace('-', '_')
        new_name = new_name.replace(' ', '-')
        new_name = new_name.replace('kwargs_','kwargs-')
        new_name = new_name.replace('buffer_','buffer-')
        new_name = new_name.replace('data_combined_', 'data_combined-')
                            
        old_file = os.path.join(root, f)
        new_file = os.path.join(root, new_name)
        if old_file == new_file:
            continue
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file.split('Code\\')[-1]}')
        print(f'to --->  {new_file.split('Code\\')[-1]}')