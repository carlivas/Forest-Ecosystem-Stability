import os
folder = 'Data/baseline/L1000'
# Path to the folder containing the buffers

load_folder = os.path.abspath(folder)
print(f'\nload_folder: {load_folder}\n')

for root, dirs, files in os.walk(load_folder):
    print(f'root: {root}')
    print(f'dirs: {dirs}')
    print(f'files: {files}')
    print()
    print("The following files will be renamed:")
    for old_file in files:
        new_file = old_file.replace('-', '_')
        new_file = new_file.replace('kwargs_', 'kwargs-')
        new_file = new_file.replace('buffer_', 'buffer-')
        new_file = new_file.replace('data_combined_', 'data_combined-')
        new_file = new_file.replace('_l', '_L')
        new_file = new_file.replace('_p', '_P')
        
        if old_file != new_file:
            print(f"{old_file:<50}-> {new_file}")
    
    confirm = input("Do you want to proceed with the renaming? (Y/n): ").lower()
    if confirm != 'y':
        print("Renaming aborted.")
        break
    for old_file in files:
        new_file = old_file.replace('-', '_')
        new_file = new_file.replace('kwargs_','kwargs-')
        new_file = new_file.replace('buffer_','buffer-')
        new_file = new_file.replace('data_combined_', 'data_combined-')
                            
        new_file = new_file.replace('_l', '_L')
        new_file = new_file.replace('_p', '_P')
        
        if old_file == new_file or new_file in files:
            continue

        old_file = os.path.join(root, old_file)
        new_file = os.path.join(root, new_file)
        os.rename(old_file, new_file)
        print(f'{old_file.split('Code\\')[-1]:<50} renamed to ->  {new_file.split('Code\\')[-1]:<50}')