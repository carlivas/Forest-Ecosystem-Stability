#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit

source ./scripts/bash/venv.sh

L=3000
num_plantss=(1500 2250 3000 3750 4500)
precipitations=(800e-4 1000e-4 1200e-4)
dispersal_range=30

len_precipitations=${#precipitations[@]}
len_num_plants=${#num_plantss[@]}

for ((i = 1; i <= len_precipitations; i++)); do
    for ((j = 1; j <= len_num_plants; j++)); do
        precipitation=${precipitations[$(($i - 1))]}
        num_plants=${num_plantss[$(($j - 1))]}
        
        save_path="Data/precipitation_experiments_L${L}m/lognorm_disp${dispersal_range}m/precipitation_${precipitation}"
        
        echo -ne "\n\n------------------------------------------------------------------------------------------------------------------------\n"
        echo -ne "testing.sh: Run ($i, $j) with precipitation = ${precipitation}, num_plants = ${num_plants}\n"
        python scripts/python/testing_bash.py $save_path $L $num_plants $precipitation $dispersal_range
    done
done
