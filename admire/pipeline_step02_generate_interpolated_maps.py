import numpy as np
import h5py
import glob
import os
import sys

import time
from tqdm import tqdm
import astropy.units as u

import configparser as cp


from pyx import print_tools, math_tools

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

    params["NewProjected"] = config.getboolean("Interpolation", "NewProjected")


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

    return int((dist_max - dist_min) / interval) + 1


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


def linear_interp2d(z, map_lower, map_higher, comoving_dist=False, NewProjected=False):
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
        if NewProjected:
            dm_name = "map"
            header_name = "Header"
        else:
            dm_name = "DM"
            header_name = "HEADER"

        y2 = ds2[dm_name][:]
        y1 = ds1[dm_name][:]

        if comoving_dist:
            x2 = z_to_mpc(ds2[header_name].attrs["Redshift"])
            x1 = z_to_mpc(ds1[header_name].attrs["Redshift"])
            dist = z_to_mpc(z) - x1
        else:
            x2 = ds2[header_name].attrs["Redshift"]
            x1 = ds1[header_name].attrs["Redshift"]
            dist = z - x1

        grad = (y2 - y1)/ (x2 - x1)

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


def pixels_to_length(pixels, boxsize, num_pixels):
    """
    Calculates the length that corresponds to a given number of 'pixels' in
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
    length : float
        The length that corresponds to a given number of pixels in the
        boxsize

    """
    length = (boxsize * pixels) / num_pixels

    return length


def get_redshift_from_header(path, NewProjected=False):
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
        if NewProjected:
            header_name = "Header"
        else:
            header_name = "HEADER"

        redshift = ds[header_name].attrs["Redshift"]
        if redshift < 1e-10:
            redshift = 0.0
        return redshift


def get_map_redshifts(paths, NewProjected=False):
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
        redshifts[i] = get_redshift_from_header(path, NewProjected)

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
    # Find the index where the last "True"
    # This is the idx lower than the sample
    idx_lower = np.where(state)[0][-1]

    # Find the index where the last "False"
    # This is the first idx lower than the sample
    idx_higher = np.where(np.logical_not(state))[0][0]

    return idx_lower, idx_higher


def get_adjacent_maps(z, z_maps, maps_paths):
    """
    Returns the paths of maps adjacent to the sample redshift.
    """
    idx_lower, idx_higher = get_adjacent_idxs(z, z_maps)
    map_lower, map_higher = maps_paths[idx_lower], maps_paths[idx_higher]
    return map_lower, map_higher


def hdf5_create_dataset(file, name, data, attributes):
    """

    Parameters
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


def get_header_attributes(z, map_lower, map_higher, transformation, NewProjected):
    """

    """
    with h5py.File(map_lower, "r") as ds1, h5py.File(map_higher, "r") as ds2:
        if NewProjected:
            header_name = "Header"
        else:
            header_name = "HEADER"

        ds1_attrs = dict(ds1[header_name].attrs)
        ds2_attrs = dict(ds2[header_name].attrs)

        if NewProjected:
            boxsize_str = ds1_attrs["Boxsize"].decode("utf-8")
        else:
            boxsize_str = ds1_attrs["Boxsize"]

        boxsize = float(boxsize_str.split(" ")[0])
        numpixels = ds1_attrs["NumPixels"]

        header_attributes = {}
        header_attributes.update(ds1_attrs)
        del header_attributes["Snapshot"]

        mirrored = True if transformation[0] == 1 else False
        rotated = True if transformation[1] == 1 else False
        x_translate = pixels_to_length(transformation[2], boxsize, numpixels)
        y_translate = pixels_to_length(transformation[3], boxsize, numpixels)

        new_attributes = {
            "Redshift": z,
            "SnapshotLower": ds1_attrs["Snapshot"],
            "SnapshotHigher": ds2_attrs["Snapshot"],
            "RedshiftLower": ds1_attrs["Redshift"],
            "RedshiftHigher": ds2_attrs["Redshift"],
            "TransformTuple": transformation,
            "Mirrored": mirrored,
            "Rotated": rotated,
            "XTranslate": "{} Mpc".format(x_translate),
            "YTranslate": "{} Mpc".format(y_translate),
        }

        header_attributes.update(new_attributes)

    return header_attributes

def convert_col_density_to_dm(data, redshift=None):
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
    dm = data * unit_dm * (1 + redshift)**-1

    return dm.value

def create_interpolated_maps(z_interp, z_maps, maps_paths,
                            transform_seq, params):
    """

    """

    pbar = tqdm(z_interp, disable=not params["ProgressBar"])

    for z in pbar:
        i = np.where(z_interp == z)[0][0]

        map_lower, map_higher = get_adjacent_maps(z, z_maps, maps_paths)

        # Perform a 2D linear interpolation from neighbouring maps
        # Also note that there is a factor of (1 + z).
        pbar.set_description(f"Interpolating z = {z:.2f}")

        # This is in units of column density not DM, need to convert
        output_map = linear_interp2d(z, map_lower, map_higher, NewProjected=params["NewProjected"])

        # Convert to DM after interpolating Column Density since CD is linear with z.
        output_map = convert_col_density_to_dm(output_map, redshift=z)

        # Perform transformation on map
        #pbar.set_description(f"Transforming z = {z:.2f}")
        #output_map = transformation.perform_transform(output_map, transform_seq[i])

        pbar.set_description(f"Shuffling z = {z:.2f}")
        output_map = math_tools.mixup(output_map)


        pbar.set_description(f"Saving HDF5 z = {z:.2f}")

        fn = "{}_z{:.4f}.hdf5".format(params["InterpFileName"], z)
        output_fn = os.path.join(params["OutputDir"], fn)

        with h5py.File(output_fn, "w") as h5:

            header_attributes = get_header_attributes(z, map_lower, map_higher,
                                                      transform_seq[i], params["NewProjected"])

            output_map_attributes = {
                "Units": "pc cm**-3",
                "VarDescription": "Dispersion Measure. Electron column density"
            }

            hdf5_create_group_attributes(h5, "HEADER", header_attributes)
            hdf5_create_dataset(h5, "DM", output_map, output_map_attributes)


def run():
    params = read_user_params(sys.argv[1])

    maps_paths = get_file_paths(loc=params["MapDir"], reverse=True)
    print(maps_paths)
    maps_redshifts = get_map_redshifts(maps_paths, params["NewProjected"])

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

    create_interpolated_maps(z_interp, maps_redshifts, maps_paths, transform_sequence, params)



if __name__ == "__main__":
    print_tools.script_info.print_header()
    run()
    print_tools.script_info.print_footer()
