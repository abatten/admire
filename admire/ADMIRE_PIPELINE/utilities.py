"""
Utility functions for the ADMIRE pipeline
"""
import os
from glob import glob
import astropy.units as u
from astropy.cosmology import Planck15 as P15, z_at_value

def vprint(msg, verbose=True, *args, **kwargs):
    """
    Acts the same as the print function but with a verbose option. Setting
    verbose to `False` will ignore the print statement.

    Parameters
    ----------
    msg : anything
        The message or object to be printed.

    verbose : boolean, optional
        Whether the pring function should actually occur.

    """

    if verbose:
        print(msg, *args, **kwargs)
    else:
        pass


def z_to_mpc(redshift):
    """
    Convert a redshift to a comoving distance

    Parameters
    ----------
    redshift : float
        A redshift value

    Returns
    -------
    Comoving distance of redshift in Mpc

    """

    if redshift <= 1e-10:
        return 0 * u.Mpc
    else:
        return P15.comoving_distance(redshift)


def mpc_to_z(mpc):
    """
    Convert a comoving distance to a redshift.

    Parameters
    ----------
    mpc : float
        A comoving distance in Mpc.

    Returns
    -------
    The redshift of the comiving distance.
    """

    if mpc <= 1e-10 * u.mpc:
        return 0
    else:
        return z_at_value(P15.comoving_distance, mpc)



def get_file_paths(loc="", suffix=".hdf5"):
    """
    """

    if suffix[0] == ".":
        suffix = "".join(["*", suffix])
    else:
        suffix = "".join("*.", suffix)

    return sorted(glob(os.path.join(loc, suffix)))

