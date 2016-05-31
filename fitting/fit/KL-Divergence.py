from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from scipy.integrate import simps, trapz
import numpy as np

class KLDivergence(DensityModel):
    def __init__(self, element_name, grid, file_path):
        DensityModel.__init__(self, element_name, grid, file_path)

def update_coefficients(electron_density, coefficients_array, exponents_array, grid):
    assert len(coefficients_array) == len(exponents_array)

    exponential = np.exp(-exponents_array * np.power(grid, 2.0))
    gaussian_density = np.dot(exponential, coefficients_array)

    updated_coefficients = np.zeros((len(coefficients_array)))
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    for i in range(0, len(coefficients_array)):
        slater_orbital = coefficients_array[i] * np.exp(- exponents_array[i] * np.power(np.ravel(grid), 2.0))
        assert slater_orbital.ndim == 1

        integrand = electron_density * (np.divide(np.ravel(slater_orbital) , np.ravel(gaussian_density)))
        print(integrand)
        integrand = np.nan_to_num(integrand)
        updated_coefficients[i] = np.trapz(y=integrand,  x=np.ravel(grid))


    return(updated_coefficients)


def update_exponents(electron_density, electrons_per_shell_array, coefficients_array, exponents_array, grid):
    exponential = np.exp(-exponents_array * np.power(grid, 2.0))
    gaussian_density = np.dot(exponential, coefficients_array)
    updated_exponents = np.zeros((len(exponents_array)))
    old_err_state = np.seterr(divide='raise')
    ignored_states = np.seterr(**old_err_state)
    for i in range(0, len(exponents_array)):
        slater_orbital = coefficients_array[i] * np.exp(- exponents_array[i] * np.power(np.ravel(grid), 2.0))
        integrand = electron_density * np.ravel(grid) * (np.divide(np.ravel(slater_orbital) , np.ravel(gaussian_density)))
        integrand = np.nan_to_num(integrand)

        updated_exponents[i] = (1/(3 * electrons_per_shell_array[i])) * np.trapz(y=integrand, x=np.ravel(grid))

    return(updated_exponents)

if __name__ == "__main__":
    ELEMENT_NAME = "c"
    ATOMIC_NUMBER = 6

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 300; NUMBER_OF_DIFFUSED_PTS = 400
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)

    NUMBER_OF_BASIS_FUNCS = 3

    NUMBER_OF_ELECTRON_SHELL_1S = 2
    NUMBER_OF_ELECTRON_SHELL_2S = 2
    NUMBER_OF_ELETRON__SHELL_2P = 2
    NUMBER_OF_ELETRON_PER_SHELL = np.array([NUMBER_OF_ELECTRON_SHELL_1S, NUMBER_OF_ELECTRON_SHELL_2S, NUMBER_OF_ELETRON__SHELL_2P])

    COEFFICIENT1 = 2.0
    COEFFICIENT2 = 2.0
    COEFFICIENT3 = 2.0
    coefficient_array = np.array([COEFFICIENT1,COEFFICIENT2, COEFFICIENT3])

    BOHR_RADIUS_AU_UNITS = 1
    BOHR_RADIUS_A_UNITS = 0.529

    EXPONENTS_1 = BOHR_RADIUS_A_UNITS / (2 * ATOMIC_NUMBER)
    EXPONENTS_2 = BOHR_RADIUS_A_UNITS / (2)
    EXPONENTS_3 = BOHR_RADIUS_A_UNITS / 2 * (ATOMIC_NUMBER **(1 - (1 / (NUMBER_OF_BASIS_FUNCS - 1))))
    exponent_array = np.array([EXPONENTS_1,EXPONENTS_3, EXPONENTS_2])

    print(1, coefficient_array, exponent_array)

    fitting_object = Fitting(be)

    for i in range(1, 10):
        original_coeff = np.copy(coefficient_array)
        original_exp = np.copy(exponent_array)

        coefficient_array = update_coefficients(np.ravel(be.electron_density), original_coeff, original_exp, column_grid_points)
        exponent_array = update_exponents(np.ravel(be.electron_density), NUMBER_OF_ELETRON_PER_SHELL, original_coeff, original_exp, column_grid_points)

        parameters = np.concatenate((coefficient_array, exponent_array))

        parameters2 = (fitting_object.optimize_using_l_bfgs(parameters, NUMBER_OF_BASIS_FUNCS))
        model2 = be.create_model(parameters2, NUMBER_OF_BASIS_FUNCS)
        print(be.integrate_model_using_trapz(model2))

        model = be.create_model(parameters, NUMBER_OF_BASIS_FUNCS)
        print(i, coefficient_array, exponent_array)
        print("Approx ", be.integrate_model_using_trapz(model), "True", be.integrated_total_electron_density)
        print("")

