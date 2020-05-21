import numpy as np
import h5py
from glob import glob
import astropy.units as u
import json

import os
import sys
from tqdm import tqdm
import configparser as cp


import pyx
from utilities import vprint


def read_user_params(param_path):
    """

    """
    if not os.path.exists(param_path):
        raise OSError("Could not find parameter file: {}".format(param_path))

    params = {}

    params["ParamPath"] = param_path
    params["ParamName"] = param_path.split("/")[-1]

    config = cp.ConfigParser()
    config.read(param_path)

    # Output Parameters
    params["Verbose"] = config.getboolean("Conversion", "Verbose")
    params["ProgressBar"] = config.getboolean("Conversion", "ProgressBar")

    # Data Parameters
    params["DataDir"] = config.get("Conversion", "DataDir")
    params["OutputDir"] = config.get("Conversion", "OutputDir")

    # Simulation Parameters
    params["SimName"] = config.get("Conversion", "SimName")
    params["EOS"] = config.get("Conversion", "EOS")
    params["ProjectionAxis"] = config.get("Conversion", "ProjectionAxis")
    params["Boxsize"] = config.get("Conversion", "Boxsize")
    params["NumPixels"] = config.getint("Conversion", "NumPixels")
    params["MapzVals"] = json.loads(config.get("Conversion", "MapzVals"))

    vprint("READING PARAMETER FILE FOR CONVERSION", params["Verbose"])

    for key, val in params.items():
        vprint("{:<20} {}".format(key, val), params["Verbose"])

    return params

def hdf5_create_dataset(file, name, data, attributes):
    """

    Parameters
    ----------
    file

    name

    data: The data

    attributes : dictionary
        A dictionary containing the attributes and description of the dataset.

    """
    file.create_dataset(name, data=data, dtype=np.float)

    for key, val in attributes.items():
        file[name].attrs[key] = val



def hdf5_create_group_attributes(file, name, attributes):
    """
    """
    group = file.create_group(name)

    for key, val in attributes.items():
        group.attrs[key] = val

def convert_npz_map_to_hdf5(npz_file, params, redshift=None):
    """
    Converts a .npz file to a .hdf5 file with all the data attributes


    Parameters
    ----------
    npz_file : str
        The path of the .npz file.

    params : 
        The output file name.

    redshift : float or None, optional
        The redshift of the npz map.
    """

    filename = "_".join(["dispersion_measure", params["SimName"],
                         params["EOS"], "comoving", "z{}p{}".format(str(redshift).split(".")[0], str(redshift).split(".")[1])])

    filename = os.path.join(params["OutputDir"], ".".join([filename, "hdf5"]))

    with np.load(npz_file, "r") as ds, h5py.File(filename, "w") as h5:

        # Get list of parameters from npz file name
        file_name_values = npz_file.split("/")[-1].split("_")

        # Get parameters and values from file name
        snapshot_number = int(file_name_values[3])
        code_version = file_name_values[4]
        metal_abundances_type = file_name_values[5]
        kernel_shape = file_name_values[6]
        slice_length = file_name_values[8][:4] + " Mpc"

        # Convert electron column density to DM
        dm_data = convert_col_density_to_pc_cm3(ds["arr_0"], redshift=redshift)

        dm_data_attributes = {
            "Units": "pc cm**-3",
            "VarDescription": "Dispersion Measure. Electron column density"
        }

        header_attributes = {
            "SimName": params["SimName"],
            "Snapshot": snapshot_number,
            "Redshift": redshift,
            "EOS": params["EOS"],
            "ProjectionAxis": params["ProjectionAxis"],
            "NumPixels": params["NumPixels"],
            "Boxsize": params["Boxsize"],
            "SliceLength": slice_length,
            "CodeVersion": code_version,
            "MetalAbundancesType": metal_abundances_type,
            "KernelShape": kernel_shape,
        }

        hdf5_create_group_attributes(h5, "HEADER", header_attributes)
        hdf5_create_dataset(h5, "DM", dm_data, dm_data_attributes)

def convert_col_density_to_pc_cm3(data, redshift=None):
    """
    Converts column density [cm**-2] into dispersion measure [pc cm**-3]

    DM = (1 + z)**-1 * CD * pc/cm

    Parameters
    ----------
    data : numpy.ndarray
        Column density data.

    redshift: float, optional
        The redshift of the data. Default: 0

    Returns
    -------
    dm : numpy.ndarray
        The data converted to dispersion measure.

    """
    if redshift is None:  # If no redshift is passed, assume redshift is zero.
        redshift = 0.0

    unit_col_dens = 1 * u.cm**-2
    unit_dm = unit_col_dens.to("pc cm**-3")


    # The factor of (1 + z) was removed after a discussion with Chris Blake.
    # It looks like we should be correct for it later in the pipeline.
    dm = (10**data) * unit_dm #* (1 + redshift)**-1

    return dm.value

def get_file_paths(loc="", suffix=".npz"):
    """

    """

    if suffix[0] == ".":
        suffix = "".join(["*", suffix])
    else:
        suffix = "".join(["*.", suffix])

    return sorted(glob(os.path.join(loc, suffix)))


if __name__ == "__main__":
    pyx.decoprint.header()
    params = read_user_params(sys.argv[1])

    data_files = get_file_paths(loc=params["DataDir"])

    for fn in tqdm(data_files, desc="npz --> hdf5", disable=not params["ProgressBar"]):
        convert_npz_map_to_hdf5(fn, params, redshift=params["MapzVals"][data_files.index(fn)])

    pyx.decoprint.footer()
