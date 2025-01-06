#!/bin/bash
#SBATCH --job-name=trees
#SBATCH --partition=modi_short
#SBATCH --output=./slurm_out/PLOT_%j.out
#SBATCH --exclusive

# Run the plotting script
path='../../Data/modi_ensemble_test/precipitation_64e-3'

echo "plotting.sh: Loading from $path"

# echo "plotting.sh: Please enter arguments:"
# read -p "plotting.sh: Detailed plot? (might take longer...) (1/0): " detailed_plot

# Set default values
print_kwargs=1
plot_data=1
plot_states=1
plot_density_field=1
detailed_plot=0

converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python ../python/plotting.py $converted_path $print_kwargs $plot_data $plot_states $plot_density_field $detailed_plot