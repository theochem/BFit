import numpy as np
from fitting.kl_divergence.least_squares import *
import math
import os

"""
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

def test_cost_function():
    file_path =  os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/examples/be.slater'

    #Test Float Numbers
    coeff = np.array([2.0])
    be = DensityModel(element_name="be", file_path=file_path, grid=np.array([[1.0]]), exponents=np.array([2.0]), change_exponents=True)
    value = be.cost_function(coeff)
    density = be.true_model
    calc_value = np.array([[2.0 * np.exp(-1.0 * 2.0 * 1.0**2)]])
    assert value == np.sum((density - calc_value)**2)


    #Test Array Numbers
    coeff = np.array([1.0, 2.0])
    be = DensityModel(element_name='be', file_path=file_path, grid=np.array([[1.0], [2.0]]), exponents=np.array([3.0, 4.0]), change_exponents=True)
    value = be.cost_function(coeff)
    density = be.true_model
    calc_value = np.array([ [1.0 * np.exp(-1.0 * 3.0 * 1.0**2) + 2.0 * np.exp(-1.0 * 4.0 * 1.0 **2) ], [1.0 * np.exp(-1.0 * 3.0 * 2.0**2) + 2.0 * np.exp(-1.0 * 4.0 * 2.0**2)]])
    assert value == np.sum((density - calc_value)**2)

def test_derivative_cost_function():
    file_path =  os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/examples/be.slater'


    #Test Float Value
    coeff = np.array([2.0])
    be = DensityModel(element_name="be", file_path=file_path, grid=np.array([[1.0]]), exponents=np.array([2.0]), change_exponents=True)
    value = be.derivative_cost_function(coeff)
    density = be.true_model

    exponential = np.exp(-1.0 * 2.0 * 1.0**2)
    calc_value = -2.0 * np.array([[exponential]]) * (density -  2.0 * exponential)
    assert value == calc_value

    #Test List
    coeff = np.array([1.0, 2.0])
    be = DensityModel(element_name='be', file_path=file_path, grid=np.array([[1.0], [2.0]]), exponents=np.array([3.0, 4.0]), change_exponents=True)
    value = be.derivative_cost_function(coeff)
    density = be.true_model

    calc_value = -2.0 * (density - np.array([ [1.0 * np.exp(-1.0 * 3.0 * 1.0**2) + 2.0 * np.exp(-1.0 * 4.0 * 1.0 **2) ], [1.0 * np.exp(-1.0 * 3.0 * 2.0**2) + 2.0 * np.exp(-1.0 * 4.0 * 2.0**2)]]))
    derivative_first_coeff = calc_value * np.array([ [np.exp(-1.0 * 3.0 * 1.0**2)], [np.exp(-1.0 * 3.0 * 2.0**2)]])
    derivative_sec_coeff = calc_value * np.array([ [np.exp(-1.0 * 4.0 * 1.0**2)], [np.exp(-1.0 * 4.0 * 2.0**2)]])
    assert value[0] == np.ravel(derivative_first_coeff)[0] + np.ravel(derivative_first_coeff)[1]
    assert value[1] == np.ravel(derivative_sec_coeff)[0] + np.ravel(derivative_sec_coeff)[1]

    # Test Derivative with scipy approximation function
    be = DensityModel(element_name="be", file_path=file_path, grid=np.array([[1.0]]), exponents=np.array([float(x) for x in range(0,5)]), change_exponents=True)
    coeff = np.array([float(x) for x in [  5.0,   0.0,   0.0, 2.44, 5.6]])

    proximation = scipy.optimize.approx_fprime(coeff, be.cost_function, epsilon=1e-5)
    derivative = be.derivative_cost_function(coeff)
    assert np.absolute(proximation[0] - derivative[0]) < 1e-5
    assert np.absolute(proximation[1] - derivative[1]) < 1e-5
    assert np.absolute(proximation[-1] - derivative[-1]) < 1e-5

"""