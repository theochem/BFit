import os
import numpy as np
import math
from fitting.fit.GaussianBasisSet import *

file_path =  os.path.dirname(__file__).rsplit('/', 2)[0] + '/data/examples/be.slater'

def test_create_model():
    # One Value Test
    grid = 1.0
    coefficient = 3.0
    exponent = 2.0
    exponential = math.exp(-1 * exponent * 1.0**2) * 3.0
    parameters = np.array([3.0, 2.0])
    grid = np.array([[1.0]])
    model_object = GaussianTotalBasisSet("be", grid, file_path)
    model = model_object.create_model(parameters, 1)
    assert math.fabs(exponential - model )< 1e-13


    # Multiple Value Test
    grid = np.array([[1], [2], [3], [4], [5]])
    model_object = GaussianTotalBasisSet("be", grid, file_path)
    parameters = np.array([2, 2, 2, 4, 5, 1, 2, 3, 4, 5])
    model = model_object.create_model(parameters, 5)
    first_value_calc = 2 * np.exp(-1) + 2 * np.exp(-2) + 2 * np.exp(-3) + 4 * np.exp(-4) + 5 * np.exp(-5)
    secon_value_calc = 2 * np.exp(-1 * 2**2) + 2 * np.exp(-2 * 2**2) + 2 * np.exp(-3 * 2**2) + 4 * np.exp(-4 * 2**2) + 5 * np.exp(-5 * 2**2)
    third_value_calc = 2 * np.exp(-1 * 3**2) + 2 * np.exp(-2 * 3**2) + 2 * np.exp(-3 * 3**2) + 4 * np.exp(-4 * 3**2) + 5 * np.exp(-5 * 3**2)
    assert model[0] == first_value_calc
    assert model[1] == secon_value_calc
    assert model[2] == third_value_calc

def test_cost_function():
    # Test Value
    grid = 3.0
    coefficient = 4.0
    exponent = 5.0
    model = 4.0 * math.exp(-exponent * grid**2)


    parameters = np.array([4.0, 5.0])
    grid = np.array([[3.0]])
    model_object = GaussianTotalBasisSet("be", grid, file_path)
    electron_density = model_object.electron_density
    manual_cost_function = (electron_density - model)**2
    cost_function = model_object.cost_function(parameters, 1)

    assert math.fabs(cost_function - manual_cost_function) < 1e-13


    #Multiply Array
    grid = np.array([[1], [2]])
    model_object = GaussianTotalBasisSet("be", grid, file_path)
    parameters = np.array([1.0, 2.0, 3.0, 4.0])
    cost_function = model_object.cost_function(parameters, 2)
    calc_value = np.array( [1.0 * np.exp(-3.0 * 1.0**2) + 2.0 * np.exp(-4.0 * 1.0**2) ,1.0 * np.exp(-3.0 * 2.0**2) + 2.0 * np.exp(-4.0 * 2.0**2)])
    manual_cost_function = np.ravel(model_object.electron_density) - calc_value

    assert cost_function[0] - manual_cost_function[0] < 1e-13
    assert cost_function[1] - manual_cost_function[1] < 1e-13

def test_derivative_cost_function():
    #Test Float Value
    parameters = np.array([2.0, 3.0])
    grid = np.array([[4.0]])
    model_object = GaussianTotalBasisSet("be", grid, file_path)
    calculated_derivative = model_object.derivative_of_cost_function(parameters, 1)

    grid = 4.0
    coefficient = 2.0
    exponent = 3.0
    exponential = math.exp(-3.0 * 4.0**2)
    derivative_of_cost_function_wrt_coefficient = -2.0 * (model_object.electron_density - 2.0 * exponential) * exponential
    derivative_of_cost_function_wrt_exponent = 2.0 * (model_object.electron_density - 2.0 * exponential) * -2.0 * exponential * -1 * grid**2
    both_derivatives = np.append(derivative_of_cost_function_wrt_coefficient, derivative_of_cost_function_wrt_exponent)

    assert calculated_derivative[0] == both_derivatives[0]
    assert calculated_derivative[1] == both_derivatives[1]

    #Test With Array WRT TO ONLY COEFFICIENT
    #TODO Test with derivative wrt to exponent too
    grid = np.array([[1.0], [2.0]])
    parameters = np.array([1.0, 2.0, 3.0 ,4.0])
    model_object = GaussianTotalBasisSet("be", grid, file_path)
    derivative = model_object.derivative_of_cost_function(parameters, 2)
    density = model_object.electron_density

    calc_value = -2.0 * (density - np.array([ [1.0 * np.exp(-1.0 * 3.0 * 1.0**2) + 2.0 * np.exp(-1.0 * 4.0 * 1.0 **2) ], [1.0 * np.exp(-1.0 * 3.0 * 2.0**2) + 2.0 * np.exp(-1.0 * 4.0 * 2.0**2)]]))
    derivative_first_coeff = calc_value * np.array([ [np.exp(-1.0 * 3.0 * 1.0**2)], [np.exp(-1.0 * 3.0 * 2.0**2)]])
    derivative_sec_coeff = calc_value * np.array([ [np.exp(-1.0 * 4.0 * 1.0**2)], [np.exp(-1.0 * 4.0 * 2.0**2)]])
    assert derivative[0] == np.ravel(derivative_first_coeff)[0] + np.ravel(derivative_first_coeff)[1]
    assert derivative[1] == np.ravel(derivative_sec_coeff)[0] + np.ravel(derivative_sec_coeff)[1]

    #Test with Scipy
    grid = np.array([[1.0]])
    exponents = np.array([float(x) for x in range(0,5)])
    coefficient = np.array([float(x) for x in [  5.0,   3.0,  2.0, 2.44, 5.6]])
    parameters = np.append(coefficient, exponents)
    model_object = GaussianTotalBasisSet("be", grid, file_path)

    approximation = scipy.optimize.approx_fprime(parameters, model_object.cost_function, 1e-5, 5)
    derivative = model_object.derivative_of_cost_function(parameters, 5)
    assert np.absolute(approximation[0] - derivative[0]) < 1e-5
    assert np.absolute(approximation[1] - derivative[1]) < 1e-5
    assert np.absolute(approximation[-1] - derivative[-1]) < 1e-5


test_derivative_cost_function()
test_create_model()
test_cost_function()