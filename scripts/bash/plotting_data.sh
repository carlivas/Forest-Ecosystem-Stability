#!/bin/bash
#SBATCH --job-name=TREESDATAPLOT
#SBATCH --partition=modi_short
#SBATCH --output=./slurm_out/TREESDATAPLOT_%j.out
#SBATCH --exclusive

# Run the plotting script
path="../../Data/modi_ensemble_test"

echo "plotting.sh: Loading from $path"

save_plot=1
converted_path=$(echo "$path" | sed 's|\\\\|/|g')

echo -ne "\n------------------------------------------------------------\n"

# Run the plotting Python script
python3 ../python/plotting_data.py $converted_path $save_plot