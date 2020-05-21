#!/bin/bash
#SBATCH --ntasks=25
#SBATCH --mem=40G
#SBATCH --time=6:00:0

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load openmpi/3.0.0
module load hdf5/1.10.1

mpirun -n 25 python ../pipeline_step04_perform_submap_sum.py ../param_files/pipeline_interp_redshift.params
