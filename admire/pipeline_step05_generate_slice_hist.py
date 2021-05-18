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
from pyx import print_tools
import fast_histogram


def sum_sub_map_slice_hist(sub_map_path, slice_idx, bins):
    with h5py.File(sub_map_path, "r") as f:
        #edges = bins

        #new_dm = np.log10(f["DM"][:, :, slice_idx])# - 940)
        #hist = fast_histogram.histogram1d(new_dm, range=[0, 5], bins=1000)
        hist, edges = np.histogram(f["DM"][:, :, slice_idx] , bins)

    return (hist, edges)


def run(params):
    print(h5py.__version__)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    controller = True if not rank else False
    worker = True if rank else False


    # Create log binning and find the bin centre
    bins = 10**np.linspace(params["min_bin"], params["max_bin"], params["num_bins"])

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
    twod_array = np.empty((params["num_bins"] - 1, params["num_slices"]))

    redshifts = []
    for slc in tqdm(range(params["num_slices"]), disable=not params["verbose"]):
        hist, edges = sum_sub_map_slice_hist(sub_map_path, slc, bins)
        #plt.bar(bins[:-1], hist, width=1)
        #output_plot_name = os.path.join(params["plotdir"],"{}_unnormed_{:03}.png".format(params["plot_name"], slc))
        #plt.savefig(output_plot_name)
        #output_hist_file = os.path.join(params["outputdir"],"{}_{:03d}.npz".format(params['hist_file'], slc))
        #np.savez(output_hist_file, hist=hist, edges=edges)
        twod_array[:, slc] = hist


    output_filename = os.path.join(params["outputdir"],
        f"admire_output_interpolated_maps_histogram_{rank:03d}.hdf5")
        #f"admire_output_DM_z_hist_unnormed_fixed_mean_fixed_sigma_corrected_{rank:03d}.hdf5" )

    with h5py.File(output_filename, "w") as output, h5py.File(sub_map_path, "r") as submap:
        output.create_dataset("DMz_hist", data=twod_array, dtype=np.float)
        output.create_dataset("Redshifts", data=submap["Redshifts"][:], dtype=np.float)
        output.create_dataset("Bin_Edges", data=edges, dtype=np.float)

        bin_centres = bins[:-1] + np.diff(bins) / 2
        output.create_dataset("Bin_Centres", data=bin_centres, dtype=np.float)
    #np.savez(output_filename, "hist_data"=twod_array, "edges"=edges)


if __name__ == "__main__":
    print_tools.script_info.print_header("ADMIRE PIPELINE")
    params = dictconfig.read(sys.argv[1], "1DHist")
    run(params)
    print_tools.script_info.print_footer()
