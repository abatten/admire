import numpy as np
import os
import time
import yt
import glob
import h5py as h5
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value


def wlog(text, log, verbose=False, t=False, u=False):

    chars = len(text)
    if t:
        t_special = "*" * chars
        out = "\n{0}\n{1}\n{0}".format(t_special, text)
    elif u:
        u_special = "-" * chars
        out = "\n{0}\n{1}".format(text, u_special)

    else:
        out = text

    log.write("{0}\n".format(out))
    vprint(out, verbose=verbose)
    return None



def vprint(out, verbose=True):
    """
    Acts the same as the print function but with a verbose option. Setting
    verbose to `False` will ignore the print statement.

    Parameters
    ----------
    out : object
        The object to potentially be printed
    verbose : boolean
        Whether the print function should actually occur.
    """

    if verbose:
        print(out)
    else:
        pass


def z_to_mpc(redshift):
    """
    Convert a redshift to a comoving distance

    Parameters
    ----------
    redshift : float

    Returns
    -------
    Comoving distance of redshift in Mpc

    """

    if redshift <= 1e-10:
        return 0 * u.Mpc
    else:
        return cosmo.comoving_distance(redshift)


def mpc_to_z(mpc):
    """
    Convert a comoving distance to a redshift.

    Parameters
    ----------
    mpc : 

    Returns
    -------
    The redshift of the comiving distance.
    """

    if mpc <= 1e-10 * u.mpc:
        return 0
    else:
        return z_at_value(cosmo.comoving_distance, mpc)


def num_slices_between_redshifts(z_low, z_high, dist_spacing, logfile=None):
    """
    Calculate the number of slices that are required to span
    a redshift range with a set distance spacing between
    slices.

    Parameters
    ----------
    z_low, z_high : float
        The redshift values

    dist_spacing : float or astropy.units.quantity.Quantity
        The comoving distance between slices
    """
    if logfile:
        logfile.write("Calculating the number of required samples between z_low and z_high")
    if isinstance(dist_spacing, float):
        dist_spacing = dist_spacing * u.Mpc

    dist_low = z_to_mpc(z_low)
    dist_high = z_to_mpc(z_high)

    return int(((dist_high - dist_low) / dist_spacing).round())


def get_redshifts_with_dist_spacing(z_low, z_high, dist_spacing, 
                                    logfile=None, verbose=False):
    """
    Get the list of redshifts with a fixed comoving distance between them.

    Parameters
    ----------
    z_low, z_high : float
        The redshifts of the lower and upper limits

    dist_spacing : float or astropy.units.quantity.Quantity
        The comoving distance between redshifts

    Return
    ------
    redshifts: numpy.darray
        An array of redshifts with a fixed comoving distance between them.

    """

    if logfile:
        wlog("Getting redshifts between z = {0} and z = {1} seperated by {2} Mpc".format(z_low, z_high, dist_spacing), logfile, verbose, u=True)

    if isinstance(dist_spacing, float):
        dist_spacing = dist_spacing * u.Mpc

    num_samples = num_slices_between_redshifts(z_low, z_high, dist_spacing)
    if logfile:
        wlog("Num sample redshifts: {0}".format(num_samples), logfile, verbose)

    dist_low = z_to_mpc(z_low)

    redshifts = np.empty(0)
    
    if logfile:
        wlog("{0:16}{1:16}".format("Redshift", "Comoving Dist"), logfile, verbose)


    for i in range(num_samples):
        dist = dist_low + i * dist_spacing
        redshifts = np.append(redshifts, mpc_to_z(dist))
        if logfile:
            wlog("\n{0:<16.5f}{1:<16.2f}".format(mpc_to_z(dist), dist), logfile, verbose)

    return redshifts


def get_redshift_from_snapshot(snapshot):
    """
    Extract the redshift from a snapshot

    Parameters
    ----------
    snapshot : string
        The snapshot file

    Return
    ------
    redshift : float
        The redshift of the snapshot
    """

    if isinstance(snapshot, str):
        ds = yt.load(snapshot)
        redshift = ds["Redshift"]
    else:
        raise TypeError("Snapshot is not a string")

    return redshift


def get_idx(sample_z, z_arr):
    """
    Get the indicies in the redshift array that are below
    and above the sample_z
    """

    state = sample_z >= z_arr  # boolean array
    # Find the index where the first "True"
    # This is the first redshift idx lower than the sample_z
    lower_z_idx = np.where(state)[0][0]

    # Find the index where the first "False"
    # This is the first redshift idx lower than the sample_z
    higher_z_idx = np.where(np.logical_not(state))[0][-1]

    return lower_z_idx, higher_z_idx


def join_path(directory, filename):
    """
    Joins the path of a directory and the filename of a file
    """
    return glob.glob(os.path.join(directory, filename))


def create_log_file(log_dir, name):
    """
    Creates the log file for the analysis
    """
    
    fn = os.path.join(log_dir, name)

    # Check if LOG file already exists
    # If so change file name to a unique name
    if os.path.exists(fn):
        index_suffix = 0
        new_fn = fn + "{0:03d}".format(index_suffix)
        while os.path.exists(new_fn):
            index_suffix += 1
            new_fn = fn + "{0:03d}".format(index_suffix)
        fn = new_fn
    log = open(fn, "w")

    # Write header and date/time to the file
    wlog("ADMIRE PIPELINE LOG FILE", log, u=True)
    wlog(time.asctime(time.localtime(time.time())), log, t=True)

    return log



def convert_npz_to_h5(npz_file, fn, z=0, logfile=None, verbose=False):
    npz_fn = npz_file.split("/")[-1]
    wlog("Converting {0} to .h5 file".format(npz_fn), log=logfile, verbose=verbose)

    with np.load(npz_file, "r") as ds:

        DM = ds["arr_0"] 
        with h5.File(fn, "w") as h5_file:
            h5_file.create_dataset("DM", data=DM)
            h5_file.create_dataset("Redshift", data=z)
            h5_file["DM"].attrs["units"] = "pc cm**-3"
            h5_file["DM"].attrs["VarDescription"] = "Dispersion Measure. The electron column density DM = n_e * dl"
            h5_file["Redshift"].attrs["VarDescription"] = "Redshift of the snapshot"

def reshape_2D_to_1D(data, logfile=None, verbose=False):
    if logfile:
        wlog("Reshaping 2D data to 1D data", logfile, verbose)

    xsize, ysize = data.shape[0], data.shape[1]
    return np.reshape(data, (xsize * ysize))
