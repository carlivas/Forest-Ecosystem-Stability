#!/bin/bash
#SBATCH --job-name=trees
#SBATCH --partition=modi_short
#SBATCH --output=../../slurm_out/TESTING_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000MB

python3 ../python/testing.py