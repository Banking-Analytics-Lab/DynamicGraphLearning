#!/bin/bash
#SBATCH --account=def-cbravo
#SBATCH --mem-per-cpu=10G      # increase as needed
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00

module load python/3.8.10
source $HOME/rappi_venv/bin/activate
python3 /home/emiliano/projects/def-cbravo/emiliano/Rappi/main.py\
        --data_output="./edge.edg" --data_input="./referidos.csv" --buildData=True        --run_Algo=True \
        --alg_inp="./edge.edg" --alg_output="output.txt"