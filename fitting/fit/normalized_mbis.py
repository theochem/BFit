from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from fitting.fit.multi_objective import GaussianSecondObjTrapz, GaussianSecondObjSquared
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

import numpy as np

def normalized_gaussian_model(coefficients, exponents, grid):
    exponential = np.exp(-exponents * np.power(grid, 2.))
    normalization_constants = np.array([(exponents[x] / np.pi)**(1/2)  for x in range(0, len(exponents))])
    normalized_coeffs = normalization_constants * coefficients
    gaussian_density = np.dot(exponential, normalized_coeffs)
    return gaussian_density

def integrate_error(coefficients, exponents, electron_density, grid):
    gaussian_model = normalized_gaussian_model(coefficients, exponents, grid)
    return np.trapz(y=np.ravel(np.power(grid, 2.)) * np.abs(gaussian_model - np.ravel(electron_density)) ,x=
            np.ravel(grid))

def integrate_normalized_model(coefficients, exponents, electron_density, grid):
    gaussian_model = normalized_gaussian_model(coefficients, exponents, grid)
    return np.trapz(y=np.power(np.ravel(grid), 2.) * gaussian_model , x=np.ravel(grid))

def KL_objective_function(coefficients, exponents, true_density, grid):
    gaussian_density = normalized_gaussian_model(coefficients, exponents, grid)
    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(true_density))
    #masked_gaussian_density[masked_gaussian_density <= 1e-6] = 1e-12

    ratio = masked_electron_density / masked_gaussian_density
    return np.trapz(y=masked_electron_density * np.log(ratio), x=np.ravel(grid))


def update_coefficients(initial_coeffs, constant_exponents, electron_density, grid, masked_value=1e-6):
    assert np.all(initial_coeffs > 0) == True, "Coefficinets should be positive non zero, " + str(initial_coeffs)
    assert len(initial_coeffs) == len(constant_exponents)
    assert len(np.ravel(electron_density)) == len(np.ravel(grid))

    gaussian_density = normalized_gaussian_model(initial_coeffs, constant_exponents,  grid)

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

def update_exponents(constant_coeffs, initial_exponents, electron_density, grid, masked_value=1e-6):
    gaussian_density = normalized_gaussian_model(constant_coeffs, initial_exponents, grid)

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    masked_gaussian_density[masked_gaussian_density <= masked_value] = masked_value

    ratio = masked_electron_density / masked_gaussian_density
    new_exponents = np.empty(len(initial_exponents))
    for i in range(0, len(initial_exponents)):
        factor = 2. * ((initial_exponents[i]) / np.sqrt(np.pi))**(3/2)
        integrand = ratio * np.ravel(np.ma.asarray(np.exp(- initial_exponents[i] * np.power(grid, 2.)))) *\
                    np.ravel(np.power(grid, 2.))
        new_exponents[i] = 3. / ( factor * np.trapz(y=integrand, x=np.ravel(grid)) )
    return new_exponents

def fixed_iteration_MBIS_method(coefficients, exponents,  electron_density, grid, num_of_iterations=800,masked_value=1e-6, iprint=False):
    current_error = 1e10
    old_coefficients = np.copy(coefficients)
    new_coefficients = np.copy(coefficients)
    counter = 0
    error_array = [ [], [], [], [], [] ,[]]

    for x in range(0, num_of_iterations):
        temp_coefficients = new_coefficients.copy()
        new_coefficients = update_coefficients(old_coefficients, exponents,electron_density,
                                            grid, masked_value=masked_value)
        exponents = update_exponents(old_coefficients, exponents, electron_density, grid)
        old_coefficients = temp_coefficients.copy()

        approx_model = normalized_gaussian_model(new_coefficients, exponents, grid)
        inte_error = integrate_error(new_coefficients, exponents, electron_density, grid)
        integration_model = integrate_normalized_model(new_coefficients, exponents, electron_density, grid)
        obj_func_error = KL_objective_function(new_coefficients, exponents, electron_density, grid)

        if iprint:
            print(counter, integration_model,
               inte_error,
               obj_func_error)
        counter += 1

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

    exps = np.array([  2.50000000e-02,   5.00000000e-02,   1.00000000e-01,   2.00000000e-01,
                   4.00000000e-01,   8.00000000e-01,   1.60000000e+00,   3.20000000e+00,
                   6.40000000e+00,   1.28000000e+01,   2.56000000e+01,   5.12000000e+01,
                   1.02400000e+02,   2.04800000e+02,   4.09600000e+02,   8.19200000e+02,
                   1.63840000e+03,   3.27680000e+03,   6.55360000e+03,   1.31072000e+04,
                   2.62144000e+04,   5.24288000e+04,   1.04857600e+05,   2.09715200e+05,
                   4.19430400e+05,   8.38860800e+05,   1.67772160e+06,   3.35544320e+06,
                   6.71088640e+06,   1.34217728e+07,   2.68435456e+07,   5.36870912e+07,
                   1.07374182e+08,   2.14748365e+08,   4.29496730e+08,   8.58993459e+08,
                   1.71798692e+09,   3.43597384e+09])

    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    coeffs[coeffs == 0.] = 1e-12

    fixed_iteration_MBIS_method(coeffs, exps, be.electron_density, be.grid, num_of_iterations=10000000,iprint=True)

