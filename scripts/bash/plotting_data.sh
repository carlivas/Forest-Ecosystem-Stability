#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit 1

# Activate the virtual environment
source ./scripts/bash/venv.sh

# Run the plotting script
path='Data\precipitation_experiments_L3000m\lognorm_disp30m\precipitation_1000e-4'

echo "plotting.sh: Loading from $path"

converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python scripts/python/plotting_data.py $converted_path