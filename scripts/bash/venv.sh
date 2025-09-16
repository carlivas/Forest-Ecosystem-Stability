#!/bin/bash
'''
This bash script activates the venv in the venv directory when run. 
'''

VENV_DIR=$"C:/Users/carla/Dropbox/_CARL/UNI/KANDIDAT/PROJEKT/Code/scripts/python/trees_venv/Scripts/activate"

# Activate the virtual environment
source ${VENV_DIR}

# Print the Python executable being used
echo ''
echo "Using Python: $(which python)"
echo ''
