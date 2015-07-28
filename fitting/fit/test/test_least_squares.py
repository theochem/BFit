import numpy as np
from fitting.fit.least_squares import *
import math
import os


def test_model():
    file_path =  os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/examples/be.slater'

    be = DensityModel(element_name='be', file_path=file_path, grid=np.array([[1], [2], [3], [4], [5]]))
    exp_array = np.array([1, 2, 3, 4, 5])
    coeff_array = np.array([2, 2, 2, 4, 5])
    model = be.model(coeff_array, exp_array)
    first_value_calc = 2 * np.exp(-1) + 2 * np.exp(-2) + 2 * np.exp(-3) + 4 * np.exp(-4) + 5 * np.exp(-5)
    secon_value_calc = 2 * np.exp(-1 * 2**2) + 2 * np.exp(-2 * 2**2) + 2 * np.exp(-3 * 2**2) + 4 * np.exp(-4 * 2**2) + 5 * np.exp(-5 * 2**2)
    third_value_calc = 2 * np.exp(-1 * 3**2) + 2 * np.exp(-2 * 3**2) + 2 * np.exp(-3 * 3**2) + 4 * np.exp(-4 * 3**2) + 5 * np.exp(-5 * 3**2)
    assert model[0] == first_value_calc
    assert model[1] == secon_value_calc
    assert model[2] == third_value_calc


def test_cofactor_matrix():
    file_path =  os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/examples/be.slater'

    grid = np.array([[5], [6], [7], [8], [9]])
    be = DensityModel(element_name='be', file_path=file_path, grid=grid)
    exp_array = np.array([5, 6, 7, 8, 9])

    cofactor_matrix = be.cofactor_matrix(exp_array, change_exponents=True)

    one_one_value = np.exp(-2 * 5 * 5**2); one_two_value = np.exp(-2 * 6 * 5**2); one_three_value = np.exp(-2 * 7 * 5**2)

    assert cofactor_matrix[0,0] == one_one_value and cofactor_matrix[0, 1] == one_two_value and cofactor_matrix[0,2] == one_three_value
    assert np.shape(cofactor_matrix) == (np.shape(grid)[0], len(exp_array))
    last_value = np.exp(-2 * 9 * 9**2)
    assert cofactor_matrix[4, 4] == last_value


def test_integration():
    pass


test_model()
test_cofactor_matrix()