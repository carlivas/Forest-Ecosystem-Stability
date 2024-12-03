#!/bin/bash

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit

source ./scripts/bash/venv.sh

# Prompt the user for input
# read -p "Enter the number of runs: " n_runs
# read -p "Enter the number of iterations: " n_iter
# read -p "Enter the land quality: " lq
# read -p "Enter the species germination chance: " sg
n_iter=10000
lq=100e-5
sg=5500e-5
L=3000


echo "testing.sh: Running testing_bash.py script $n_runs times with the following arguments:"
echo "testing.sh: Number of iterations: $n_iter"
echo "testing.sh: Land quality: $lq"
echo "testing.sh: Species germination chance: $sg"

# Create a list of plant numbers to iterate through
num_plantss=(50 100 200 500 750 1000 1500 2000 3500 5000)
# num_plantss=(555 1111 2222 5555 8333 11111 16667 22222 38888 55555)
len_num_plants=${#num_plantss[@]}

for ((i = 1; i <= 5; i++)); do 
    save_path="Data/init_density_experiment_SPH_L${L}/lq${lq}_sg${sg}_gaussian_$(($i + 5))"
    for ((j = 1; j <= len_num_plants; j++)); do
        num_plants=${num_plantss[$(($j - 1))]}
        echo -ne "\n\n------------------------------------------------------------------------------------------------------------------------\n"
        echo -ne "testing.sh: Run ($i, $j)\n"
        echo -ne "testing.sh: Number of plants: $num_plants \n\n"
        python scripts/python/testing_bash.py $save_path $L $num_plants $n_iter $lq $sg
    done
done
