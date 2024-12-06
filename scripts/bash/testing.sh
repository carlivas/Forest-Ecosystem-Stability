#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit

source ./scripts/bash/venv.sh


precipitations=(6000e-5 6250e-5 6500e-5 6750e-5 7000e-5)
num_plantss=(50 100 250 500 750 1000 1250 1500 1750 2000 2500 3000)
# num_plantss=(2000 2200 2400 2600 2800 3000)
# num_plantss=(555 1111 2222 5555 8333 11111 16666 22222 38888)

len_precipitations=${#precipitations[@]}
len_num_plants=${#num_plantss[@]}

for ((i = 1; i <= len_precipitations; i++)); do
    for ((j = 1; j <= len_num_plants; j++)); do
        precipitation=${precipitations[$(($i - 1))]}
        num_plants=${num_plantss[$(($j - 1))]}
        
        save_path="Data/precipitation_experiments/cauchy_disp30m/precipitation_${precipitation}"
        
        echo -ne "\n\n------------------------------------------------------------------------------------------------------------------------\n"
        echo -ne "testing.sh: Run ($i, $j) with precipitation = ${precipitation}, num_plants = ${num_plants}\n"
        python scripts/python/testing_bash.py $save_path $L $num_plants $precipitation
    done
done
