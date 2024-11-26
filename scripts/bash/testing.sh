#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 5 ]; then
    echo "Usage: $0 <number_of_runs> <num_plants> <n_iter> <lq> <sgc>"
    exit 1
fi

# Navigate to the vscode directory
cd C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code || exit

# Activate the virtual environment
source venv/Scripts/activate

# Print the Python executable being used using the echo command
echo -ne "Using Python: $(which python):\n\n"

# Run the python script multiple times with different arguments
n_runs=$1
echo "Running the script $n_runs times with the following arguments:"
echo "Number of plants: $2"
echo "Number of iterations: $3"
echo "Land quality: $4"
echo "Species germination chance: $5"

for ((i = 1; i <= n_runs; i++)); do # C-style for loop because n_runs is a variable
    echo -ne "Run $i:\r"
    python scripts/python/testing_bash.py $2 $3 $4 $5 &
    # The & operator runs the command in the background making the for loop run faster by running multiple commands at the same time
done