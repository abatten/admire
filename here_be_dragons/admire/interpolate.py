import h5py
import utils

def linear_interp_2D(z_sample, map_low, map_high, logfile=None, verbose=False):
    """
    Performs a linear interpolation between two column density maps.

    Parameters
    ----------
    z_sample : float

    map_low :


    Returns:
    --------
    Interpolated Map
    """

    if logfile:
        utils.wlog("Performing Linear Interpolation: z = {0:.5f}\n".format(z_sample), logfile,
        verbose)

    with h5py.File(map_low, "r") as ds1, h5py.File(map_high, "r") as ds2:
        y2 = ds2["DM"][:]
        y1 = ds1["DM"][:]

        x2 = utils.z_to_mpc(ds2["Header"].attrs["Redshift"])
        x1 = utils.z_to_mpc(ds1["Header"].attrs["Redshift"])

        grad = (y2 - y1) / (x2 - x1)

        dist = utils.z_to_mpc(z_sample) - x1
        return grad * dist + y1


