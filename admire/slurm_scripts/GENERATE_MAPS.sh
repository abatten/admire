#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=60G
#SBATCH --time=2:00:0

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load git/2.16.0
module load openmpi/3.0.0
module load hdf5/1.10.1

#python ../pipeline_step02_generate_interpolated_maps.py ../param_files/pipeline.params
