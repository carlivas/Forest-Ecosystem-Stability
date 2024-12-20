#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit

source ./scripts/bash/venv.sh

num_plantss=(500 1000 1500 2000 2500 3000)
precipitations=(10e-3 20e-3 30e-3 40e-3 50e-3 60e-3 70e-3 80e-3 90e-3)
n_searches_per_parameter_set=5
len_precipitations=${#precipitations[@]}
len_num_plants=${#num_plantss[@]}

for ((i = 1; i <= len_precipitations; i++)); do
    for ((j = 1; j <= len_num_plants; j++)); do
        for ((k = 1; k <= n_searches_per_parameter_set; k++)); do
            precipitation=${precipitations[$(($i - 1))]}
            num_plants=${num_plantss[$(($j - 1))]}
            
            save_path="Data/modi_ensemble_test/precipitation_${precipitation}"
            
            echo -ne "\n\n------------------------------------------------------------------------------------------------------------------------\n"
            echo -ne "testing.sh: Run ($i, $j, $k) with precipitation = ${precipitation}, num_plants = ${num_plants}\n"
            python scripts/python/testing_bash.py $save_path $num_plants $precipitation
        done
    done
done