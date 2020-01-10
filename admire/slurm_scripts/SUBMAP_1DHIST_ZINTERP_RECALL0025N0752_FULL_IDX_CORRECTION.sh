#!/bin/bash

#SBATCH --partition=skylake
#SBATCH --ntasks=25
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --time=3:30:00
#SBATCH --account=oz071
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abatten@swin.edu.au

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load git/2.16.0
#module load hdf5/1.10.1
module load openmpi/3.0.0
module load hdf5/1.10.1

mpirun -n 25 python ../step5_generate_slice_hist.py ../param_files/pipeline_zinterp_RecalL0025N0752_full_idx_correction.param
