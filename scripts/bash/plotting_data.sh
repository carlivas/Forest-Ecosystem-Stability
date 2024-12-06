#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit 1

# Activate the virtual environment
source ./scripts/bash/venv.sh

# Run the plotting script
path='Data\precipitation_experiments\cauchy_disp30m\precipitation_6000e-5'

echo "plotting.sh: Loading from $path"

converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python scripts/python/plotting_data.py $converted_path