from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import numpy as np

def update_coefficients(initial_coeffs, constant_exponents, electron_density, grid):
    assert np.all(initial_coeffs > 0) == True
    assert len(initial_coeffs) == len(constant_exponents)
    assert len(np.ravel(electron_density)) == len(np.ravel(grid))

    exponential = np.exp(-constant_exponents * np.power(grid, 2.))
    assert exponential.shape[1] == len(constant_exponents)
    assert exponential.shape[0] == len(np.ravel(grid))
    gaussian_density = np.dot(exponential, initial_coeffs)
    assert gaussian_density.shape[0] == len(np.ravel(grid))

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))

    new_coefficients = np.empty(len(initial_coeffs))
    for i in range(0, len(initial_coeffs)):
        factor = initial_coeffs[i] * (constant_exponents[i] / np.pi)**(3/2)
        integrand = masked_electron_density * np.ravel(np.ma.asarray(np.exp(- constant_exponents[i] * np.power(grid, 2.))))\
                    / masked_gaussian_density
        new_coefficients[i] = factor * np.trapz(y=integrand, x=np.ravel(grid))
    return new_coefficients

def measure_error_by_integration_of_difference(true_model, approximate_model, grid):
        error = np.trapz(y=np.ravel(grid**2) * (np.absolute(np.ravel(true_model) - np.ravel(approximate_model))), x=np.ravel(grid))
        return error

def iterative_MBIS_method(initial_coeffs, constant_exponents, electron_density, grid, error=1e-5):
    current_error = 1e10
    old_coefficients = np.copy(initial_coeffs)
    new_coefficients = np.copy(initial_coeffs)
    print(error * len(initial_coeffs))

    #while current_error > error * len(initial_coeffs):
    for x in range(0, 100):
        temp = np.copy(new_coefficients)
        new_coefficients = update_coefficients(old_coefficients, constant_exponents, electron_density, grid)
        old_coefficients = np.copy(temp)
        current_error = np.sum(np.abs((old_coefficients - new_coefficients)))
        model = np.dot(np.exp(-constant_exponents * grid**2), new_coefficients)
        print(current_error,  measure_error_by_integration_of_difference(model, np.ravel(electron_density), np.ravel(grid)),"\n")

    return new_coefficients, current_error

if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 1000; NUMBER_OF_DIFFUSED_PTS = 1000
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)







