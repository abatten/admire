import numpy as np
import h5py
import configparser as cp
import os
import sys
from glob import glob

from utilities import get_file_paths


def read_user_params(param_path):
    """

    """
    if not os.path.exists(param_path):
        raise OSError(f"Could not find parameter file: {param_path}")

    params = {}

    params["ParamPath"] = param_path
    params["ParamName"] = os.path.basename(param_path)

    config = cp.ConfigParser()
    config.read(param_path)

    # Output Parameters
    params["Verbose"] = config.getboolean("InterpMaster", "Verbose")
    params["ProgressBar"] = config.getboolean("InterpMaster", "ProgressBar")

    # Data Location
    params["MapDir"] = config.get("InterpMaster", "MapDir")
    params["OutputDir"] = config.get("InterpMaster", "OutputDir")

    # Data FileNames
    params["InterpFileName"] = config.get("InterpMaster", "InterpFileName")
    params["MasterFileName"] = config.get("InterpMaster", "MasterFileName")
    params["Dataset"] = config.get("InterpMaster", "Dataset")
    params["Header"] = config.get("InterpMaster", "Header")

    return params


def create_master(params):

    # Get path to create the master file
    master_file = os.path.join(params["OutputDir"], params["MasterFileName"])
    master_file = ".".join([master_file, "hdf5"])

    # Get the list of interpolated files
    interp_prefix = os.path.join(params["OutputDir"], params["InterpFileName"])
    interp_files = sorted(glob(interp_prefix + "*"))

    # Open master file
    with h5py.File(master_file, mode="w") as mf:
        group = mf.create_group("dm_maps")
        group.attrs["desc"] = "This the the master file for the interpolated redshfits"

        # Link each interpolated file to master file
        for i, fn in enumerate(interp_files):
            print(i)
            with h5py.File(fn, mode="w") as file_i:

                redshift = file_i[params["Header"]].attrs["Redshift"]

                # Create sub_group
                group_name = f"z_{redshift}"
                sub_group = group.create_group(group_name)

                # Save the Header and DM dataset to the subgroup
                sub_group[params["Header"]] = h5py.ExternalLink(fn, params["Header"])
                sub_group[params["Dataset"]] = h5py.ExternalLink(fn, params["Dataset"])






if __name__ == "__main__":
    params = read_user_params(sys.argv[1])
    create_master(params)

