
import numpy as np
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value


def z_to_mpc(redshift):
    """
    Convert a redshift to a comoving distance
    """

    if redshift == 0:
        return 0 * u.Mpc
    else:
        return cosmo.comoving_distance(redshift)


def mpc_to_z(mpc):
    """
    Convert a comoving distance to a redshift
    """

    if mpc == 0:
        return 0
    else:
        return z_at_value(cosmo.comoving_distance, mpc)


def num_slices_between_redshifts(z_low, z_high, dist_spacing):
    """
    Calculate the number of slices that are required to span
    a redshift range with a set distance spacing between
    slices.

    Parameters
    ----------
    z_low, z_high : float
        The redshift values

    dist_spacing : astropy.units.quantity.Quantity
        The comoving distance between slices
    """

    dist_low = z_to_mpc(z_low)
    dist_high = z_to_mpc(z_high)

    return int(((dist_high - dist_low) / dist_spacing).round())


def get_redshifts_with_dist_spacing(z_low, z_high, dist_spacing):
    """
    Get the list of redshifts with a fixed comoving distance between them.

    Parameters
    ----------
    z_low, z_high : float
        The redshifts of the lower and upper limits

    dist_spacing : astropy.units.quantity.Quantity
        The comoving distance between redshifts

    Return
    ------
    redshifts: numpy.darray
        An array of redshifts with a fixed comoving distance between them.

    """
    num_samples = num_slices_between_redshifts(z_low, z_high, dist_spacing)

    dist_low = z_to_mpc(z_low)

    redshifts = np.empty(0)

    for i in range(num_samples):
        dist = dist_init + i * dist_spacing
        np.append(redshifts, mpc_to_z(dist))

    return redshifts


def get_idx(sample_z, z_arr):
    """
    Get the indicies in the redshift array that are below
    and above the sample_z
    """

    state = sample_z > z_arr  # boolean array

    # Find the index where the first "True"
    # This is the first redshift idx lower than the sample_z
    lower_z_idx = np.where(state)[0][0]

    # Find the index where the first "False"
    # This is the first redshift idx lower than the sample_z
    higher_z_idx = np.where(np.logical_not(state))[0][-1]

    return lower_z_idx, higher_z_idx
