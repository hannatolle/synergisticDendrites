#!/bin/bash
#SBATCH -J PhiID_run   # Job name
#SBATCH -e PhiID_run.%j.err    # Name of stderr output file(%j expands to jobId)
#SBATCH --partition=AMD
#SBATCH -c 20
#SBATCH --mem 50000
#SBATCH --time=24:00:00

export JULIA_NUM_THREADS=20

julia PIC_cont.jl

