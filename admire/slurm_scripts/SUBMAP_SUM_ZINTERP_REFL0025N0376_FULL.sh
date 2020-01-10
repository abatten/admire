#!/bin/bash
#SBATCH --ntasks=25
#SBATCH --mem=15G
#SBATCH --time=6:00:00

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load openmpi/3.0.0
module load hdf5/1.10.1




mpirun -n 25 python ../step4_perform_submap_sum.py ../param_files/pipeline_zinterp_RefL0025N0376_full.param
