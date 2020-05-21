#!/bin/bash

#SBATCH --partition=skylake
#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time=3:00:00
#SBATCH --account=oz071

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load git/2.16.0
module load hdf5/1.10.1

mpirun -n 25 python ../pipeline_step05_generate_slice_hist.py ../param_files/pipeline_zinterp.params
