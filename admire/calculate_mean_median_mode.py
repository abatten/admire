import h5py
import numpy as np
import scipy
from pyx import math_tools, str_tools
import astropy.table as atab
from glob import glob
import os

master_hdf5_file = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/coarse_snapshot_data/linear_redshift_interp/output/T4EOS/master_dm_maps.hdf5"


perform_sum_of_stats = False
perform_stats_of_sum = True


num_slices = 65

#redshift_list = ["Redshift"]
#mean_list = ["Mean"]
#median_list = ["Median"]
#mode_list = ["Mode"]

redshift_list = []
mean_list = []
median_list = []
mode_list = []

if perform_sum_of_stats:
    output_filename = "mean_median_mode_interpolated_redshift.txt"

    with h5py.File(master_hdf5_file, "r") as masterfile:
        for i in range(num_slices):
            name = f"slice_{i:03d}"
            oned_array = math_tools.reshape_to_1D(masterfile["dm_maps"][name]["DM"][:])
            mean = np.mean(oned_array)
            median = np.median(oned_array)
            mode = scipy.stats.mode(oned_array)[0][0]
            redshift = masterfile["dm_maps"][name]["HEADER"].attrs["Redshift"]
    
            redshift_list.append(redshift)
            mean_list.append(mean)
            median_list.append(median)
            mode_list.append(mode)
    
    
        #np.savetxt(out, ["Redshift", "Mean", "Median", "Mode"])
    #    np.savetxt(out, [redshift_list, mean_list, median_list, mode_list], fmt=[".5f", ".5f", ".5f", ".5f"])
    
    
elif perform_stats_of_sum:
    output_filename = "mean_median_mode_interpolated_redshift_summed.txt"

    #map_idx = [4, 6, 13, 15, 17, 24]
    map_idx = [7, 14, 16]
    mapdir = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/coarse_snapshot_data/linear_redshift_interp/output/T4EOS"
    map_filename = "sum_dm_maps"

    files = sorted(glob(os.path.join(mapdir, map_filename) + "*.hdf5"))

    redshift_vals = np.array(
    [0.        , 0.02271576, 0.04567908, 0.06890121, 0.09239378,
     0.1161688 , 0.1402387 , 0.16461634, 0.18931503, 0.21434853,
     0.23973108, 0.26547743, 0.29160285, 0.31812316, 0.34505475,
     0.37241459, 0.40022031, 0.42849013, 0.45724301, 0.48649858,
     0.51627725, 0.54660017, 0.57748936, 0.60896766, 0.64105881,
     0.67378753, 0.7071795 , 0.74126146, 0.77606125, 0.81160784,
     0.84793147, 0.88506357, 0.92303701, 0.96188603, 1.00164638,
     1.04235539, 1.08405204, 1.12677711, 1.1705732 , 1.21548489,
     1.26155887, 1.30884399, 1.35739143, 1.40725491, 1.45849065,
     1.51115776, 1.56531826, 1.62103729, 1.6783833 , 1.73742834,
     1.79824817, 1.86092257, 1.92553561, 1.99217588, 2.06093685,
     2.13191717, 2.20522104, 2.2809586 , 2.35924626, 2.44020723,
     2.52397206, 2.61067901, 2.70047481, 2.79351511, 2.88996519])


    with h5py.File(files[map_idx[0]], "r") as ds1:#, \
         #h5py.File(files[map_idx[1]], "r") as ds2, \
         #h5py.File(files[map_idx[2]], "r") as ds3:#, \
         #h5py.File(files[map_idx[3]], "r") as ds4, \
         #h5py.File(files[map_idx[4]], "r") as ds5, \
         #h5py.File(files[map_idx[5]], "r") as ds6:

        for idx in range(num_slices):
            print(f"{idx+1} / {num_slices}")
            data1 = ds1["DM"][:, :, idx]
            #data2 = ds2["DM"][:, :, idx]
            #data3 = ds3["DM"][:, :, idx]
            #data4 = ds4["DM"][:, :, idx]
            #data5 = ds5["DM"][:, :, idx]
            #data6 = ds6["DM"][:, :, idx]
            combined_data = np.array([data1])#, data2, data3])#, data4, data5, data6])

            mean = np.mean(combined_data)
            median = np.median(combined_data)
            mode = scipy.stats.mode(combined_data, axis=None)[0][0]
            redshift = redshift_vals[idx]

            redshift_list.append(redshift)
            mean_list.append(mean)
            median_list.append(median)
            mode_list.append(mode)





data_table = atab.Table(data=[redshift_list, mean_list, median_list, mode_list], names=("Redshift", "Mean", "Median", "Mode"))

data_table.write(output_filename, format='ascii')

