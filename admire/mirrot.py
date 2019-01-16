import numpy as np

def mirrot(array, mir=0, rot=0):
    """
    Mirrors and/or rotates an image array

    Parameters
    ----------

    array : 2D array

    mir : int

    rot : int

    Returns
    -------
    mirrot_array : 2D array
        The mirrored and rotated array
    """

    if mir == 0:
        mirrot_array = array
    elif mir == 1:
        mirrot_array = np.fliplr(array)
    else:
        raise ValueError("Value for mir is not 0 or 1")

    if rot <= 3:
        mirrot_array = np.rot90(mirrot_array, rot)
    else:
        raise ValueError("Value for rot is not 0, 1, 2 or 3")

    return mirrot_array
