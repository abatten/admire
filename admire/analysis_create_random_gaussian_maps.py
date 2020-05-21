import numpy as np
import os
import sys
import h5py
from scipy import stats
from astropy import units as u
from pyx.math_tools import cosmology as cosmo

def hdf5_create_group_attributes(file, name, attributes):
    """
    """
    group = file.create_group(name)

    for key, val in attributes.items():
        group.attrs[key] = val

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
    dist_min = cosmo.z_to_cMpc(zmin)
    redshifts = np.empty(num_intervals)

    for i in range(num_intervals):
        total_dist = dist_min + i * interval
        redshifts[i] =  cosmo.cMpc_to_z(total_dist)

    return redshifts

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

    dist_min = cosmo.z_to_cMpc(zmin)
    dist_max = cosmo.z_to_cMpc(zmax)

    return int((dist_max - dist_min) / interval) + 1


if __name__ == "__main__":
    mean = 1000
    std = 10

    num_pixels_per_side = 16000

    num_maps = 131

    z_list = get_redshifts_with_interval(0, 3.016, 50) + cosmo.cMpc_to_z(50)
    print(z_list)
    for idx in range(num_maps):
        map_array = stats.norm.rvs(loc=mean, scale=std, size=(num_pixels_per_side, num_pixels_per_side))

        filename = f"/home/abatten/ADMIRE_ANALYSIS/Random_Gaussian_Maps/RandL0050/random_gaussian_fixed_mean_fixed_std_{idx:03d}.hdf5"
        print(filename)
        with h5py.File(filename, "w") as output:
                output.create_dataset("dm", data=map_array, dtype=np.float)
                header_attributes = {'Redshift': z_list[idx]}
                hdf5_create_group_attributes(output, "Header", header_attributes)
