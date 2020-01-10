#!/bin/bash
#SBATCH --ntasks=25
#SBATCH --mem=15G
#SBATCH --time=6:00:00
#SBATCH --tmp=1500G

module load anaconda3/5.0.1

source activate py3

module load gcc/7.3.0
module load openmpi/3.0.0
module load hdf5/1.10.1


cp /fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/output/T4EOS/interpolated_dm_map_*.hdf5 $JOBFS

python ../step3_generate_interpolated_master.py ../param_files/pipeline_zinterp_RefL0025N0376_full_jobfs.param

wait

mpirun -n 25 python ../step4_perform_submap_sum.py ../param_files/pipeline_zinterp_RefL0025N0376_full_jobfs.param

wait

cp $JOBFS/sum_dm_maps_jobfs* /fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0025N0376/all_snapshot_data/output/T4EOS