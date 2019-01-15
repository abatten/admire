import utils

def linear_interp_2D(z_sample, data_low, dist_low, data_high, dist_high):
    """
    Performs a linear interpolation between two snapshots
    """
    
    y2 = data_high["grid"]["density"][:]
    y1 = data_low["grid"]["density"][:]
    x2 = dist_high
    x1 = dist_low
    
    grad = (y2 - y1) / (x2 - x1)
    
    dist = utils.z_to_mpc(z_sample) - x1
    return grad * dist + y1
