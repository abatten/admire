from admire import utils
from admire.mirrot import mirrot 
import numpy as np


def test_mirrot_r90():
    A = np.array([[1,2,3], [4,5,6], [7,8,9]])
    B = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])

    assert (mirrot(A, mir=0, rot=1) == B).all, "Mirrot is broken for rotations"

def test_mirrot_m():
    A = np.array([[1,2,3], [4,5,6], [7,8,9]])
    B = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    assert (mirrot(A, mir=1, rot=0) == B).all, "Mirrot is broken for mirror"
