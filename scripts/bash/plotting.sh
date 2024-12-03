#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit 1

# Activate the virtual environment
source ./scripts/bash/venv.sh

# Run the plotting script
path='Data\init_density_experiment_SPH_L3000\lq100e-5_sg5500e-5_gaussian_combined'

echo "plotting.sh: Loading from $path"
echo "plotting.sh: Please enter arguments:"

# Prompt the user if they want to provide custom values or use defaults
read -p "plotting.sh: Do you want to provide custom values? (1/0): " custom_values

if [ "$custom_values" == "1" ]; then
    # Prompt the user for input
    read -p "plotting.sh: Print kwargs? (1/0): " print_kwargs
    read -p "plotting.sh: Plot data? (1/0): " plot_data
    read -p "plotting.sh: Plot states? (1/0): " plot_states
    read -p "plotting.sh: Plot density field? (1/0): " plot_density_field
    read -p "plotting.sh: Fast plot? (1/0): " fast_plot
else
    # Set default values
    print_kwargs=1
    plot_data=1
    plot_states=1
    plot_density_field=0
    fast_plot=1
fi

converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python scripts/python/plotting.py $converted_path $print_kwargs $plot_data $plot_states $plot_density_field $fast_plot