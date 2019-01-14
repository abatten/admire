
import astropy.units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value


def z_to_mpc(redshift):

    if redshift == 0:
        return 0 * u.Mpc
    else:
        return cosmo.comoving_distance(redshift)


def mpc_to_z(mpc):

    if mpc == 0:
        return 0
    else:
        return z_at_value(cosmo.comoving_distance, mpc)


def num_slices_between_redshifts(z_low, z_high, spacing):
    """
    Calculate the number of slices that are required to span
    a redshift range with a set distance spacing between
    slices.

    Parameters
    ----------
    z_low, z_high : float
        The redshift values

    spacing : astropy.units.quantity.Quantity
        The comoving distance between slices
    """

    dist_low = z_to_mpc(z_low)
    dist_high = z_to_mpc(z_high)

    return int(((dist_high - dist_low) / spacing).round())
