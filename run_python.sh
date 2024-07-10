#!/bin/bash
#SBATCH -p a6000
#SBATCH -w mp-gpu4-a6000-1
#SBATCH --job-name=simon_python_job
#SBATCH --output=output.txt
#SBATCH --time=01:00:00

# Load any required modules, if needed
# module load python/3.8

# Execute the Python script
python create_data_astra.py