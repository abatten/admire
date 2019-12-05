import numpy as np
import h5py
import os
import sys
import dictconfig
from pyx import print_tools
from mpi4py import MPI


def calc_idx_ranges(num_cores, num_rows, num_cols, array_size):
    """
    """
    col = 0
    idx_ranges = []
    for i in range(num_cores):

        # The start and end indicies for each row
        row_start = int(i % num_rows * (array_size / num_rows))
        row_end = int((i % num_rows + 1) * (array_size / num_rows))

        # Move to next col when end of row
        if i % num_cols == 0:
            col += 1

        col_start = int((col - 1) * (array_size / num_cols))
        col_end = int(col * (array_size / num_cols))

        row_ranges = (row_start, row_end)
        col_ranges = (col_start, col_end)

        idx_ranges.append((row_ranges, col_ranges))

    return idx_ranges


def run(params):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    controller = True if not rank else False
    worker = True if rank else False


    # Divide up the maps into sum maps
    array_size = params["numpixels"]
    num_rows = int(np.sqrt(size))
    num_cols = num_rows

    # The number of pixels that each column and row will have
    num_pixels_per_row = int(array_size / num_rows)
    num_pixels_per_col = int(array_size / num_cols)

    if controller:
        idx_ranges = calc_idx_ranges(size, num_rows, num_cols, array_size)

    elif worker:
        idx_ranges = None

    row_ranges, col_ranges = comm.scatter(idx_ranges, root=0)


    masterfile = os.path.join(params["masterdir"], params["masterfilename"])
    print(f"\nResults from rank {rank} \n row {row_ranges} col {col_ranges}")
    fn = params["sumfilename"]

    outputfile = os.path.join(params["outputdir"], f"{fn}_{rank:03d}.hdf5")

    num_slices = params["num_slices"]

    with h5py.File(masterfile, mode="r") as master, h5py.File(outputfile, mode="w") as output:

        array_shape = (num_pixels_per_row, num_pixels_per_col, num_slices)

        output.create_dataset("DM", shape=array_shape, dtype=np.float)
        output.create_dataset("Redshifts", shape=(num_slices,), dtype=np.float)

        for index in range(num_slices):
            print(f"Rank: {rank} Slice: {index}")
            data = master["dm_maps"][f"slice_{index:03d}"]["DM"]
            
            output["Redshifts"][index] = master["dm_maps"][f"slice_{index:03d}"]["HEADER"].attrs["Redshift"]

            if index == 0:
                output["DM"][:, :, index] = \
                    data[row_ranges[0]: row_ranges[1],
                         col_ranges[0]: col_ranges[1]]

            # If not the first submap, add the current supmap to the previous
            # output file. 
            else:
                output["DM"][:, :, index] = \
                    (output["DM"][:, :, index - 1] +
                    data[row_ranges[0]: row_ranges[1], 
                         col_ranges[0]: col_ranges[1]])

#                woutput["DM"][:, :, index] = (output["DM"][:, :, index] *
#                    data[row_ranges[0]: row_ranges[1], col_ranges[0]: col_ranges[1]] / normalisation)


if __name__ == "__main__":
    print_tools.script_info.print_header("ADMIRE PIPELINE")

    if len(sys.argv) == 2:
        params = dictconfig.read(sys.argv[1], "SumMaster")

    elif len(sys.argv) == 1:
        print("Please provide parameter file")
        sys.exit(1)

    else:
        print("Too many command line arguments!")
        sys.exit(1)

    for key, value in params.items():
        print(f"{key:<16}: {value}")
    run(params)

    print_tools.script_info.print_footer()

