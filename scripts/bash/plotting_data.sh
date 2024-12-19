#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit 1

# Activate the virtual environment
source ./scripts/bash/venv.sh

read -p "plotting_data.sh: Save generated plots? (1/0): " save_plot

# Run the plotting script
path="Data\modi_ensemble"

echo "plotting.sh: Loading from $path"

save_plot=${save_plot:-0}
converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python scripts/python/plotting_data.py $converted_path $save_plot