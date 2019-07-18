import numpy as np
import h5py
import os
import sys
import dictconfig
import matplotlib as mpl
from glob import glob
import matplotlib.pyplot as plt
import utilities as utils
from tqdm import tqdm


def sum_sub_map_slice_hist(sub_map_paths, slice_idx, bins):
    total_hist = 0
    edges = 0
    for i, sub_map in enumerate(sub_map_paths):
        with h5py.File(sub_map, "r") as f:
            hist, e = np.histogram(np.log10(f["DM"][:, :, slice_idx]), bins)
            total_hist += hist
            edges = e

    return (total_hist, edges)





def run(params):

    # Create log binning and find the bin centre
    bins = np.linspace(params["min_bin"], params["max_bin"], params["num_bins"])
    #bincentre = (10**bins[:-1] + 10**bins[1:]) / 2
    #bins = np.linspace(0, 10000, 1000)

    sub_map_filename = os.path.join(params["datadir"], params["sub_map_name"])

    sub_map_paths = sorted(glob(f"{sub_map_filename}_*.hdf5"))

    twod_array = np.empty((65, 999))

    for slc in tqdm(range(params["num_slices"])):
        hist, edges = sum_sub_map_slice_hist(sub_map_paths, slc, bins)
        #plt.bar(bins[:-1], hist, width=1)
        #plt.savefig("{}_{:03}.png".format(params["plot_name"], slc))
        #np.savez("{}_{:03d}.npz".format(params['hist_file'], slc), hist=hist, edges=edges)
        twod_array[slc, :] = hist / sum(hist)
    np.savez("2darray_logged.npz", twod_array)






if __name__ == "__main__":
    params = dictconfig.read(sys.argv[1], "1DHist")

    run(params)
