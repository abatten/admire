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
from mpi4py import MPI
import pyx


def sum_sub_map_slice_hist(sub_map_path, slice_idx, bins):
    total_hist = 0
    edges = 0
    #for i, sub_map in enumerate(sub_map_paths):
    with h5py.File(sub_map_path, "r") as f:
        hist, edges = np.histogram(np.log10(f["DM"][:, :, slice_idx]), bins)
        total_hist += hist

    return (total_hist, edges)


def run(params):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    controller = True if not rank else False
    worker = True if rank else False


    # Create log binning and find the bin centre
    bins = np.linspace(params["min_bin"], params["max_bin"], params["num_bins"])
    #bincentre = (10**bins[:-1] + 10**bins[1:]) / 2

    # Controller get the file paths and scatter them to the remaining cores.
    # The idea is that each core performs the histogram on a single sub map.
    if controller:
        sub_map_filename = os.path.join(params["datadir"], params["sub_map_name"])
        sub_map_paths = sorted(glob(f"{sub_map_filename}_*.hdf5"))
        print(sub_map_paths)

    elif worker:
        sub_map_paths = None

    sub_map_path = comm.scatter(sub_map_paths, root=0)

    # Initialise an empty 2D array that will be the 2D histogram.
    # The reason that the size is num_bins - 1 is to account for the edges of
    # the bins. i.e There is 1 more bin than data.
    twod_array = np.empty((params["num_slices"], params["num_bins"] - 1))

    for slc in tqdm(range(params["num_slices"]), disable=not params["verbose"]):
        hist, edges = sum_sub_map_slice_hist(sub_map_path, slc, bins)
        plt.bar(bins[:-1], hist, width=1)
        output_plot_name = os.path.join(params["plotdir"],"{}_unnormed_{:03}.png".format(params["plot_name"], slc))
        plt.savefig(output_plot_name)
        output_hist_file = os.path.join(params["outputdir"],"{}_{:03d}.npz".format(params['hist_file'], slc))
        np.savez(output_hist_file, hist=hist, edges=edges)
        twod_array[slc, :] = hist #/ sum(hist)

    output_filename = os.path.join(params["outputdir"],
        f"admire_output_2D_array_logged_unnormed_{rank:03d}.npz" )

    np.savez(output_filename, twod_array)






if __name__ == "__main__":
    pyx.decoprint.header("ADMIRE PIPELINE")
    params = dictconfig.read(sys.argv[1], "1DHist")
    run(params)
    pyx.decoprint.footer()
