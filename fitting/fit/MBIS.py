from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import decimal
import numpy as np

d = decimal.Decimal
#Absolute difference= 314245.645194 15667.5795399 [ 329913.2247343]

def update_coefficients(initial_coeffs, constant_exponents, electron_density, grid, masked_value=1e-6):
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
    masked_gaussian_density[masked_gaussian_density <= masked_value] = masked_value

    ratio = masked_electron_density / masked_gaussian_density

    new_coefficients = np.empty(len(initial_coeffs))
    for i in range(0, len(initial_coeffs)):
        factor = initial_coeffs[i] * (constant_exponents[i] / np.pi)**(1/2) * 2.
        integrand = ratio * np.ravel(np.ma.asarray(np.exp(- constant_exponents[i] * np.power(grid, 2.))))
        new_coefficients[i] = factor * np.trapz(y=integrand, x=np.ravel(grid))
    return new_coefficients

def measure_error_by_integration_of_difference(true_model, approximate_model, grid):
        error = np.trapz(y=np.ravel(grid**2) * (np.absolute(np.ravel(true_model) - np.ravel(approximate_model))), x=np.ravel(grid))
        return error

def update_exponents(constant_coeffs, initial_exponents, electron_density, grid, masked_value=1e-6):
    assert np.all(constant_coeffs > 0) == True
    assert len(constant_coeffs) == len(initial_exponents)
    assert len(np.ravel(electron_density)) == len(np.ravel(grid))

    exponential = np.exp(-initial_exponents * np.power(grid, 2.))
    assert exponential.shape[1] == len(initial_exponents)
    assert exponential.shape[0] == len(np.ravel(grid))
    gaussian_density = np.dot(exponential, constant_coeffs)
    assert gaussian_density.shape[0] == len(np.ravel(grid))

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    masked_gaussian_density[masked_gaussian_density <= masked_value] = masked_value

    ratio = masked_electron_density / masked_gaussian_density
    new_exponents = np.empty(len(initial_exponents))
    for i in range(0, len(initial_exponents)):
        factor = 12. * np.sqrt(initial_exponents[i]) / np.sqrt(np.pi)
        integrand = ratio * np.ravel(np.ma.asarray(np.exp(- initial_exponents[i] * np.power(grid, 2.)))) *\
                    np.ravel(np.power(grid, 2.))
        new_exponents[i] = 3. / ( factor * np.trapz(y=integrand, x=np.ravel(grid)) )
    return new_exponents


def iterative_MBIS_method(initial_coeffs, constant_exponents, electron_density, grid, error=1e-5, masked_value=1e-6):
    current_error = 1e10
    old_coefficients = np.copy(initial_coeffs)
    new_coefficients = np.copy(initial_coeffs)

    return new_coefficients, current_error



if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 300; NUMBER_OF_DIFFUSED_PTS = 400
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)
    exps = be.UGBS_s_exponents
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    parameters = fitting_obj.optimize_using_slsqp(np.append(coeffs, exps), len(exps))

    coeffs[coeffs == 0] = 1E-6






