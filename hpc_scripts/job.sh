#!/bin/bash

#PBS -l select=1:ncpus=32:mem=32gb
#PBS -l walltime=08:00:00
#PBS -N dend_sims
#PBS -J 1-2
 
module load anaconda3/personal

source ~/venv/bin/activate
cd ~/projects/synergisticDendrites/hpc_scripts
python main.py ${HOME} ${PBS_ARRAY_INDEX}
