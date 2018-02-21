#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J dust_sph
#SBATCH --mail-user=dmuley@berkeley.edu
#SBATCH --mail-type=ALL
#SBATCH -t 03:00:00
#SBATCH -A m2218

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
module load python/2.7-anaconda
srun -n 10 -c 6 --cpu_bind=cores python code_running.py
