import utils

def linear_interp_2D(z_sample, data_low, dist_low, data_high, dist_high, 
                     logfile=None, verbose=False):
    """
    Performs a linear interpolation between two snapshots
    """

    if logfile:
        utils.wlog("Performing Linear Interpolation: z = {0:.5f}\n".format(z_sample), logfile, verbose=verbose)
        
    
    y2 = data_high["DM"][:]
    y1 = data_low["DM"][:]
    x2 = dist_high
    x1 = dist_low
    
    grad = (y2 - y1) / (x2 - x1)
    
    dist = utils.z_to_mpc(z_sample) - x1
    return grad * dist + y1


