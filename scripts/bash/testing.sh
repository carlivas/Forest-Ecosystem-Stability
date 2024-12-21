#!/bin/bash
#SBATCH --job-name=TREESTEST
#SBATCH --partition=modi_long
#SBATCH --output=./slurm_out/TREESTEST_%j.out
#SBATCH --exclusive

num_plantss=(500 1000 1500 2000 2500 3000 3500 4000)
precipitations=(51e-3 52e-3 53e-3 54e-3 55e-3 56e-3 57e-3 58e-3 59e-3 60e-3 61e-3 62e-3 63e-3 64e-3 65e-3 66e-3 67e-3 68e-3 69e-3 70e-3)
n_searches_per_parameter_set=5
len_precipitations=${#precipitations[@]}
len_num_plants=${#num_plantss[@]}

for ((i = 1; i <= len_precipitations; i++)); do
    for ((j = 1; j <= len_num_plants; j++)); do
        for ((k = 1; k <= n_searches_per_parameter_set; k++)); do
            precipitation=${precipitations[$(($i - 1))]}
            num_plants=${num_plantss[$(($j - 1))]}
            
            save_path="../../Data/modi_ensemble_test/precipitation_${precipitation}"
            
            echo -ne "\n\n------------------------------------------------------------------------------------------------------------------------\n"
            echo -ne "testing.sh: Run ($i, $j, $k) with precipitation = ${precipitation}, num_plants = ${num_plants}\n"
            python3 ../python/testing_bash.py $save_path $num_plants $precipitation
        done
    done
done