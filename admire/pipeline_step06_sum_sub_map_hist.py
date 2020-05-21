import numpy as np
from glob import glob
import os
import h5py


output_dir_name = "/fred/oz071/abatten/ADMIRE_ANALYSIS/ADMIRE_RefL0100N1504/all_snapshot_data/output/T4EOS"

submap_file_name = "admire_output_DM_z_hist_unnormed_0*"

files = glob(os.path.join(output_dir_name, submap_file_name))

print(files)

for i, f in enumerate(files):
    with h5py.File(f, "r") as ds:
        print(f"{i+1}")
        if i == 0:
            data = ds["DMz_hist"][:]
            redshifts = ds["Redshifts"][:]
            Bin_Edges = ds["Bin_Edges"][:]
            Bin_Centres = ds["Bin_Centres"][:]
        else:
            data += ds["DMz_hist"][:]

data_normed = data / np.sum(data, axis=0)
    
    
output_name = "admire_output_DM_z_hist_total_normed_idx_corrected.hdf5"
output_filename = os.path.join(output_dir_name, output_name)

with h5py.File(output_filename, "w") as output:
    output.create_dataset("DMz_hist", data=data_normed, dtype=np.float)
    output.create_dataset("Redshifts", data=redshifts, dtype=np.float)
    output.create_dataset("Bin_Edges", data=Bin_Edges, dtype=np.float)
    output.create_dataset("Bin_Centres", data=Bin_Centres, dtype=np.float)
