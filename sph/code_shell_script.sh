#!/bin/bash
for number in{1..5}
do
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -t 03:00:00
#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
#run the application:
srun -n 64 -c 1 --cpu_bind=threads code_running.py
done
Exit 0
