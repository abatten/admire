import os
from os import path

import numpy as np
from admire import utils
from admire.mirrot import mirrot


dirpath = path.dirname(__file__)

np.random.seed(42)

simple_matrix = path.join(dirpath, "data/simple_matrix.txt")

#matrix = np.loadtxt(simple_matrix, dtype="i", delimiter=" ")
def matrix():
    return np.loadtxt(simple_matrix, dtype="i", delimiter=" ")



def test_mirrot_r90():
    A = np.array([[1,2,3],[4,5,6],[7,8,9]])
    B = np.array([[3, 6, 9], [2, 5, 8], [1, 4, 7]])
    assert (mirrot(A, mir=0, rot=1) == B).all, "Broken"

def test_mirrot_m():
    A = matrix()
    B = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    assert (mirrot(A, mir=1, rot=0) == B).all, "Broken"
