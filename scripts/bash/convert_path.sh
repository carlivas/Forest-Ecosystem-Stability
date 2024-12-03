#!/bin/bash


path = '.\Data\init_density_experiment_SPH\lq_2e-1_sg3e-2'
# Convert backslashes to forward slashes
converted_path=$(echo "$path" | sed 's|\\\\|/|g')

# Print the converted path
echo "$converted_path"