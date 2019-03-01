import numpy as np
import random
from collections import Counter

def options():
    options = {'r90': (0, 1),
               'r180': (0, 2),
               'r270': (0, 3),
               'm': (1, 0),
               'mr90': (1, 1),
               'mr180': (1, 2),
               'mr270': (1, 3)
              }
    return options


def mirror_rotate_options():
    options = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)]
    return options



def gen_mirrot_sequence(length):

    ops = options()

    seq = np.empty(length)


    return seq

def mirrot(array, mir=0, rot=0, transx=0, transy=0):
    """
    Mirrors and/or rotates an image array

    Parameters
    ----------

    array : 2D array

    mir : int

    rot : int

    transx : int

    transy : int

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


def perform_transform(array, transformation=(0, 0, 0, 0)):
    """
    Performs mirror, rotation and translation of array in that order

    Parameters
    ----------
    array: 2D numpy.ndarray
        A 2D array to transform

    transformation: tuple of ints
        A tuple containing 4 integers representing the transformation. Each
        tuple represents: (mirror, rotate, translate_x, translate_y).
        mirror can be either 0 or 1 (no mirror and mirror around vertical axis
        respectively). rotate can be 0, 1, 2 or 3 (for 0, 90, 180 and 270 deg
        rotation respectfully), translate_x and translate_y are the number of
        positions to translate along the x and y axes.
        Default:(0, 0, 0, 0)
    """

    # Check the transformation tuples contains the correct number of values.
    if len(transformation) is not 4:
        raise ValueError("""transformation is expecting a length 4 tuple.
            Instead got {}""".format(transformation))

    # Ensure that all the values in the transformation tuple are integers.
    for val in transformation:
        if type(val) is not int:
            raise ValueError("""transformation needs to be a tuple of ints.
                Instead tuple contains value of type: {}""".format(type(val)))

    # Extract the transformation values from the tuple
    mir, rot, transx, transy = transformation

    # Peform the mirroring of the array
    if mir == 0:
        pass
    elif mir == 1:
        array = _mirror(array)
    else:
        raise ValueError("""Value for mirror must be 0 or 1. Instead recieved
            mirror value: {0}""".format(mir))

    # Peform the rotation of the array
    if rot <= 3:
        array = _rotate(array, rot)
    else:
        raise ValueError("""Value for rotation must be an integer less than 3.
            Instead recieved rotation value: {0}""".format(rot))

    # Peform the translation of the array
    array = _translate(array, transx, transy)

    return array


def _mirror(array):
    """
    Mirrors a 2D numpy array around the vertical axis

    Parameters
    ----------
    array: 2D numpy array

    Returns
    -------
    numpy.ndarray
        The mirrored array
    """
    mirrored_array = np.fliplr(array)
    return mirrored_array


def _rotate(array, rot=0):
    """
    Rotate a 2D numpy array anti-clockwise in a multiple of 90 degrees.

    Parameters
    ----------
    array: 2D numpy array

    rot: int
        The multiple of 90 degrees with which to rotate the array. i.e rot=2
        implies a 2 x 90 = 180 degree rotation. Default: 0

    Returns
    -------
    numpy.ndarray
        The rotated array
    """
    rotated_array = np.rot90(array, rot)
    return rotated_array


def _translate(array, transx=0, transy=0):
    """
    Translates a 2D array in the x and y directions

    Parameters
    ----------
    array: 2D numpy array

    transx: int
        The number of positions to translate along the x axis. Defaut: 0

    transy: int
        The number of positions to translate along the y axis. Default: 0

    Returns
    -------
    numpy.ndarray
        The translated array
    """
    translated_array = np.roll(array, (transx, transy), axis=(1, 0))
    return translated_array


def gen_random_transform(min_trans, max_trans):
    """
    Generates a random transformation

    Parameters
    ----------
    min_trans: int
        The minimum value for the x and y translations.

    max_trans: int
        The maximum value for the x and y translations

    Returns
    -------
    tuple of ints
        A tuple containing 4 integers representing the transformation. Each
        tuple represents: (mirror, rotate, translate_x, translate_y).
        mirror can be either 0 or 1 (no mirror and mirror around vertical axis
        respectively). rotate can be 0, 1, 2 or 3 (for 0, 90, 180 and 270 deg
        rotation respectfully), translate_x and translate_y are the number of
        positions to translate the array along the x and y axis.

    """
    mir, rot = random.choice(mirror_rotate_options())
    transx = random.randint(min_trans, max_trans)
    transy = random.randint(min_trans, max_trans)

    transform = (mir, rot, transx, transy)

    return transform

def gen_transform_sequence(seq_length, min_trans, max_trans):
    """
    Generates a sequence of transformations of a specified length.

    Parameters
    ----------
    seq_length: int
        The number of transformations to in the sequence to generate.

    min_trans: int
        The minimum value for the x and y translations.

    max_trans: int
        The maximum value for the x and y translations

    Returns
    -------
    list of tuples
        A list containing a sequence of transformations
    """

    if type(seq_length) is not int:
        raise ValueError("""seq_length must be of type int. Instead seq_length
            hase type: {0}""".format(type(seq_length)))

    seq = []
    for i in range(seq_length):
        seq.append(gen_random_transform(min_trans, max_trans))

    return seq
