import numpy as np
import h5py
import glob
import os
import sys

import time
from tqdm import tqdm
import astropy.units as u

import configparser as cp


import transformation
from utilities import vprint, z_to_mpc, mpc_to_z, get_file_paths


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
    params["Verbose"] = config.getboolean("Interpolation", "Verbose")
    params["ProgressBar"] = config.getboolean("Interpolation", "ProgressBar")

    # Data Parameters
    params["MapDir"] = config.get("Interpolation", "MapDir")
    params["DataDir"] = config.get("Interpolation", "DataDir")
    params["OutputDir"] = config.get("Interpolation", "OutputDir")

    # Interpolation Parameters
    params["RedshiftMin"] = config.getfloat("Interpolation", "RedshiftMin")
    params["RedshiftMax"] = config.getfloat("Interpolation", "RedshiftMax")
    params["DistSpacing"] = config.getfloat("Interpolation", "DistSpacing")
    params["MinTransLength"] = config.getfloat("Interpolation", "MinTransLength")

    # Simulation Parameters
    params["Boxsize"] = config.getfloat("Interpolation", "Boxsize")
    params["NumPixels"] = config.getint("Interpolation", "NumPixels")

    # Output file names
    params["InterpFileName"] = config.get("Interpolation", "InterpFileName")
    params["MasterFileName"] = config.get("Interpolation", "MasterFileName")


    vprint("READING PARAMETER FILE FOR INTERPOLATION", params["Verbose"])

    for key, val in params.items():
        vprint("{:<20} {}".format(key, val), params["Verbose"])

    return params



def num_intervals_between_redshifts(zmin, zmax, interval):
    """
    Calculate the number of distance intervals that are required to span
    a redshift range.

    Parameters
    ----------
    zmin : float
        The minumum redshift value.

    zmax : float
        The maximum redshift value.

    interval : float
        A distance interval in Mpc.

    units : str, optional
        The units of the distance interval. Default: "Mpc"

    Returns
    -------
    num_intervals : float
        The number of distance intervals required to span the min and
        max redshift.

    """

    # If interval is not a quantity convert it.
    if not isinstance(interval, u.Quantity):
        interval = interval * u.Mpc

    dist_min = z_to_mpc(zmin)
    dist_max = z_to_mpc(zmax)

    return int((dist_max - dist_min) / interval)


def get_redshifts_with_interval(zmin, zmax, interval):
    """
    Get the list of redshifts with a fixed comoving distance between them.

    Parameters
    ----------
    zmin, zmax : float
        The redshifts of the lower and upper limits

    interval : float or astropy.units.quantity.Quantity
        The comoving distance between redshifts

    Return
    ------
    redshifts: numpy.darray
        An array of redshifts with a fixed comoving distance between them.

    """

    # If interval is not a quantity convert it
    if not isinstance(interval, u.Quantity):
        interval = interval * u.Mpc

    num_intervals = num_intervals_between_redshifts(zmin, zmax, interval)
    dist_min = z_to_mpc(zmin)
    redshifts = np.empty(num_intervals)

    for i in range(num_intervals):
        total_dist = dist_min + i * interval
        redshifts[i] =  mpc_to_z(total_dist)

    return redshifts


def linear_interp2d(z, map_lower, map_higher):
    """
    Peforms a linear interpolation between two dispersion measure maps.

    Parameters
    ----------
    z : float
        The redshift of the interpolated slice.


    Returns
    -------
    """

    with h5py.File(map_lower, "r") as ds1, h5py.File(map_higher, "r") as ds2:
        y2 = ds2["DM"][:]
        y1 = ds1["DM"][:]

        x2 = z_to_mpc(ds2["HEADER"].attrs["Redshift"])
        x1 = z_to_mpc(ds1["HEADER"].attrs["Redshift"])

        grad = (y2 - y1)/ (x2 - x1)

        dist = z_to_mpc(z) - x1

        return grad * dist + y1


def calc_min_max_translate(min_length, boxsize, num_pixels):
    """
    Calculate the minimum and maximum possible pixels for translation based
    on a minimum length translation.

    The maximum translation is equivalent to a minimum translation in the
    opposite direction.

    Parameters
    ----------
    min_length : float
        The minimum length in the map to translate.

    boxsize : float
        The total length of one size of the box.

    num_pixels : int
        The number of pixels along one size of the box.

    Returns
    -------
    trans_min_pix : int
        The number of pixels that corresponds to min_length in the box.

    trans_max_pix : int
        The number of pixels that corresponds to boxsize-min_length.
        Translating by this number of pixels correcponds to translating by
        min_length in the negative direction.

    """
    trans_min_pix = length_to_pixels(min_length, boxsize, num_pixels)

    max_length = boxsize - min_length
    trans_max_pix = length_to_pixels(max_length, boxsize, num_pixels)


    return trans_min_pix, trans_max_pix


def length_to_pixels(length, boxsize, num_pixels):
    """
    Calculates the number of 'pixels' that corresponds to a given length in
    a snapshot.

    Parameters
    ----------
    length : float
        The length in the snapshot to convert in to pixels.

    boxsize : float
        The total length of the box in the same units of length.

    num_pixels : int
        The number of pixels along an axis of the box.

    Return
    ------
    pixels : int
        The number of pixels that corresponds to the specified length in the
        boxsize

    """
    pixels = int((length / boxsize) * num_pixels)

    return pixels


def get_redshift_from_header(path):
    """
    Parameters
    ----------
    path: str
        The path of the hdf5 file containing "Redshift" as a keyword in the
        Group "HEADER".

    Returns
    -------
    redshift: float
        The redshift value of the file

    """
    with h5py.File(path, "r") as ds:
        return ds["HEADER"].attrs["Redshift"]


def get_map_redshifts(paths):
    """
    Parameters
    ----------
    paths: list of strings
        A list containing the paths to hdf5 files.

    Returns
    -------
    redshifts: numpy.ndarray
        A 1D array containing the redshifts of the maps.

    """

    redshifts = np.empty(len(paths))

    for i, path in enumerate(paths):
        redshifts[i] = get_redshift_from_header(path)

    return redshifts


def get_adjacent_idxs(sample, array):
    """
    Get the indicies in an array that are below and above a sample value.

    Parameters
    ----------
    sample: float or int

    array: array or array like

    Returns
    -------
    idx_lower :

    idx_higher :

    """
    state = sample >= array #  boolean array
    # Find the index where the first "True"
    # This is the first idx lower than the sample
    idx_lower = np.where(state)[0][0]

    # Find the index where the last "False"
    # This is the first idx higher than the sample
    idx_higher = np.where(np.logical_not(state))[0][-1]

    return idx_lower, idx_higher


def get_adjacent_maps(z, z_maps, maps_paths):
    """
    Returns the paths of maps adjacent to the sample redshift.
    """
    idx_lower, idx_higher = get_adjacent_idxs(z, z_maps)
    map_lower, map_higher = maps_paths[idx_lower], maps_paths[idx_higher]
    return map_lower, map_higher


def create_interpolated_maps(z_interp, z_maps, transform_seq, filename,
                            verbose=True, pbar_on=True):
    """

    """

    pbar = tqdm(z_interp, disable=not pbar_on)

    for z in pbar:
        i = np.where(z_interp == z)[0][0]

        map_lower, map_higher = get_adjacent_maps(z, z_maps, maps_paths)

        pbar.set_description("Interpolating z = {:.2f}".format(z))
        interp_map = linear_interp2d(z, map_lower, map_higher)

        pbar.set_description("Transforming z = {:.2f}".format(z))
        transformed_map = transformation.perform_transform(interp_map, transform_seq[i])



if __name__ == "__main__":
    params = read_user_params(sys.argv[1])

    maps_paths = get_file_paths(loc=params["MapDir"])
    maps_redshifts = get_map_redshifts(maps_paths)

    vprint("\nRedshifts of Maps", params["Verbose"])
    vprint(maps_redshifts, params["Verbose"])

    # Generate list of redshifts with fixed distance spacing
    z_interp = get_redshifts_with_interval(params["RedshiftMin"],
                                           params["RedshiftMax"],
                                           params["DistSpacing"])

    vprint("\nRedshifts of Maps to Generate with Interpolation", params["Verbose"])
    vprint(z_interp, params["Verbose"])

    # Calculate the minimum and maximum number of pixels to translate
    trans_min, trans_max = calc_min_max_translate(min_length=params["MinTransLength"],
                                                  boxsize=params["Boxsize"],
                                                  num_pixels=params["NumPixels"])

    # Generate the transformation sequence
    transform_sequence  = transformation.gen_transform_sequence(
                                seq_length=len(z_interp),
                                min_trans=trans_min,
                                max_trans=trans_max)


    vprint("\nTransformation Sequence", params["Verbose"])
    for i, transform in enumerate(transform_sequence):
        vprint("{:<8.3f} {}".format(z_interp[i], transform), params["Verbose"])

    create_interpolated_maps(z_interp, maps_redshifts, transform_sequence,
                             params["InterpFileName"])
