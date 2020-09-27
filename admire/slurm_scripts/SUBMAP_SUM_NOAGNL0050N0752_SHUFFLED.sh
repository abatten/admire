#!/bin/bash
#SBATCH --ntasks=25
#SBATCH --mem=15G
#SBATCH --time=18:00:00
#SBATCH --tmp=1500G
#SBATCH --account=oz071
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abatten@swin.edu.au

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load openmpi/3.0.0
module load hdf5/1.10.1


cp /fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_NoAGNL0050N0752/all_snapshot_data/shuffled_output/shuffled_interpolated_dm_map_*.hdf5 $JOBFS

python ../pipeline_step03_generate_interpolated_master.py ../param_files/pipeline_NoAGNL0050N0752_shuffled_jobfs.param

wait

mpirun -n 25 python ../pipeline_step04_perform_submap_sum.py ../param_files/pipeline_NoAGNL0050N0752_shuffled_jobfs.param

wait

cp $JOBFS/sum_* /fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_NoAGNL0050N0752/all_snapshot_data/shuffled_output/
