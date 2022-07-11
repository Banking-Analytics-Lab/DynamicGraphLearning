#!/bin/bash
#SBATCH --account=def-cbravo
#SBATCH --mem-per-cpu=10G      # increase as needed
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00

rappi_folder=$HOME/projects/def-cbravo/rappi_data
output_folder=$HOME/projects/def-cbravo/rappi_data/output/node2vect
module load python/3.8.10
source $HOME/node2vect/bin/activate
cd /home/$USER/projects/def-cbravo/$USER/RappiAnomalyDetection
python3 ./methods/node2vect/main.py\
        --data_output="${output_folder}/edge.edg" --data_input="${rappi_folder}/referidos.csv" --buildData=True\
        --run_Algo=True --alg_inp="${output_folder}/edge.edg" --alg_output="${output_folder}/output.txt"