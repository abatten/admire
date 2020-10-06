import numpy as np
from glob import glob
import os
import h5py
from pyx.math_tools import cosmology as pyxcosmo

output_dir_name = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_FBconstL0050N0752/all_snapshot_data/shuffled_output/"
#output_dir_name = "/fred/oz071/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0025"
#output_dir_name = "/fred/oz071/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0100"

#submap_file_name = "admire_output_DM_z_hist_unnormed_*"

submap_file_name = "admire_output_DM_z_hist_unnormed_fixed_mean_fixed_sigma_corrected_*"
#submap_file_name = "admire_output_DM_z_hist_unnormed_divided_100_0*"

#submap_file_name = "Batten2020_EAGLE_unnormed"
#submap_file_name = "

files = glob(os.path.join(output_dir_name, submap_file_name))

PDF_direction = "DM"

print(files)

for i, f in enumerate(files):
    with h5py.File(f, "r") as ds:
        print(f"{i+1}")
        if i == 0:
            data = ds["DMz_hist"][:]
            redshifts = ds["Redshifts"][:]
            DM_Bin_Edges = ds["Bin_Edges"][:]
            DM_Bin_Centres = ds["Bin_Centres"][:]
        else:
            data += ds["DMz_hist"][:]

Redshifts = np.zeros(len(redshifts) + 1)
Redshifts[0] = pyxcosmo.cMpc_to_z(0)
Redshifts[1:] = redshifts + pyxcosmo.cMpc_to_z(50)



#Redshifts[:-1] = redshifts + pyxcosmo.cMpc_to_z(100)
#Redshifts[-1] = redshifts[-1] + pyxcosmo.cMpc_to_z(200)

print(Redshifts)

data_norm = np.zeros(data.shape)

Redshift_Bin_Widths = np.diff(Redshifts)

DM_Bin_Widths = np.diff(DM_Bin_Edges)
DM_Bin_Centres = DM_Bin_Edges[:-1] + DM_Bin_Widths / 2

if PDF_direction == "z":
    print(data.shape)
    for idx, hist in enumerate(data):
        if np.sum(hist.T) < 1e-16:
            pdf = np.zeros(len(hist))
        else:
            pdf = hist/Redshift_Bin_Widths/np.sum(hist.T)

        # Prints out the area under the PDF. If all went well this should be 1.0
        print(np.sum(pdf * Redshift_Bin_Widths))

        data_norm[idx, :] = pdf


if PDF_direction == "DM":
    for idx, hist in enumerate(data.T):
        pdf = hist/DM_Bin_Widths/np.sum(hist)
        print(np.sum(pdf * DM_Bin_Widths))

        data_norm[:, idx] = pdf


if PDF_direction == "None":
    data_norm = data

output_name = "admire_output_DM_z_hist_total_DM_normed_newkeys.hdf5"

output_filename = os.path.join(output_dir_name, output_name)

with h5py.File(output_filename, "w") as output:
    output.create_dataset("DMz_hist", data=data_norm, dtype=np.float)
    output.create_dataset("DM_Bin_Edges", data=DM_Bin_Edges, dtype=np.float)
    output.create_dataset("DM_Bin_Centres", data=DM_Bin_Centres, dtype=np.float)
    output.create_dataset("DM_Bin_Widths", data=DM_Bin_Widths, dtype=np.float)
    output.create_dataset("Redshifts_Bin_Edges", data=Redshifts, dtype=np.float)
    output.create_dataset("Redshift_Bin_Widths", data=Redshift_Bin_Widths, dtype=np.float)

