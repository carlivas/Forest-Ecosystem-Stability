#!/bin/bash
#SBATCH --job-name=trees
#SBATCH --partition=modi_short
#SBATCH --output=./slurm_out/DATAPLOT_%j.out
#SBATCH --exclusive

# Run the plotting script
path="../../Data/modi_ensemble_L4500"

echo "plotting.sh: Loading from $path"

save_plot=1
converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python3 ../python/plotting_data.py $converted_path $save_plot