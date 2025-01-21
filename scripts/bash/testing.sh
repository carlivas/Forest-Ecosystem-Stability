#!/bin/bash
#SBATCH --job-name=trees
#SBATCH --partition=modi_long
#SBATCH --output=./slurm_out/TESTING_%j.out
#SBATCH --exclusive

num_plantss=(500 1000 1500 2000 2500 3000)
precipitations=(60e-3 61e-3 62e-3 63e-3 64e-3 65e-3 66e-3 67e-3 68e-3 69e-3 70e-3)
n_searches_per_parameter_set=5
len_precipitations=${#precipitations[@]}
len_num_plants=${#num_plantss[@]}

for ((i = 1; i <= len_precipitations; i++)); do
    for ((j = 1; j <= len_num_plants; j++)); do
        for ((k = 1; k <= n_searches_per_parameter_set; k++)); do
            precipitation=${precipitations[$(($i - 1))]}
            num_plants=${num_plantss[$(($j - 1))]}
            
            save_path="../../Data/modi_ensemble_L4500/precipitation_${precipitation}"
            
            echo -ne "\n\n------------------------------------------------------------------------------------------------------------------------\n"
            echo -ne "testing.sh: Run ($i, $j, $k) with precipitation = ${precipitation}, num_plants = ${num_plants}\n"
            python3 ../python/testing_bash.py $save_path $num_plants $precipitation
        done
    done
done