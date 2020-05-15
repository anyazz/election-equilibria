#!/bin/bash
#SBATCH --test-only
#SBATCH -n 1 # Number of cores requested
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 1-00:00 # Runtime in days-hours:minutes
#SBATCH -p shared # Partition to submit to
#SBATCH --mem=100 # Memory per node in MB (see also --mem-per-cpu)
#SBATCH --open-mode=write # Append when writing files
#SBATCH -o pov_data_%j.out # Standard out goes to this file
#SBATCH -e pov_data_%j.err # Standard err goes to this filehostname
python3 run_pov_stdout.py