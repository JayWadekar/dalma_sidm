#!/usr/bin/env python

#SBATCH -p parallel
#SBATCH -o "/scratch/dsw310/sidm_data/surface/output_del.log"
#SBATCH -e "/scratch/dsw310/sidm_data/surface/error_del.log"
#SBATCH -n 56 
# My commands


import sys
import os
import multiprocessing

print multiprocessing.cpu_count()
print os.environ['SLURM_JOB_CPUS_PER_NODE']

# Necessary to add cwd to path when script run
# by SLURM (since it executes a copy)
sys.path.append(os.getcwd())

def hello():
    print("Hello World")

pool = multiprocessing.Pool() 
jobs = [] 
for j in range(25):
    p = multiprocessing.Process(target = hello)
    jobs.append(p)
    p.start() 
    
# daya #SBATCH --partition=whatever_the_name_of_your_SLURM_partition_is
# daya #SBATCH -m abe
# daya#SBATCH -M dsw310@nyu.edu
