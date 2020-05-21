#!/bin/bash -l

#SBATCH --ntasks=1                     # 1
#SBATCH --cpus-per-task=1              # 25
#SBATCH --mem=60                        # 15G
#SBATCH --time=3:00:00                      # [d-]hh:mm:ss

#SBATCH --chdir=

#SBATCH --job-name=calc_var_correlations # blank_slurm_project
#SBATCH --output=stdout.%J.%A                     # stdout.%J.%A
#SBATCH --error=stderr.%J.%A                      # stderr.%J.%A

#SBATCH --mail-type=ALL
#SBATCH --mail-user=abatten@swin.edu.au


###################
# LOAD MODULES
##################

module purge
module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load hdf5/1.10.1

python ../step9_calculate_variance_interpolated_maps.py ../param_files/pipeline_zinterp_RefL0100N1504_full_idx_correction.param

#load the modules used to build your program.
#module load git/2.16.0 
#module load gcc/6.4.0
#module load openmpi/3.0.0
#module load python/3.6.4
#module load hdf5/1.10.1
#module load numpy/1.14.1-python-3.6.4
#module load ipython/5.5.0-python-3.6.4
#module load matplotlib/2.2.2-python-3.6.4
#module load h5py/2.7.1-python-3.6.4
#module load tkinter/3.6.4-python-3.6.4
