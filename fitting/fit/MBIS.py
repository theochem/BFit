from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import numpy as np

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
    masked_gaussian_density[masked_gaussian_density <= masked_value] = 0.0

    ratio = masked_electron_density / masked_gaussian_density
    #print(masked_electron_density[200:250], masked_gaussian_density[200:250])
    #print(np.where(masked_gaussian_density < masked_value))
    #print(ratio[len(ratio) - 10:])
    #print(masked_gaussian_density[len(ratio) - 10:])
    #print("max and min", np.max(ratio), np.min(ratio))
    max_val = np.max(ratio)
    min_val = np.min(ratio)
    index_max, index_min = (np.argmax(ratio), np.argmin(ratio))
    #index_max, index_min = (np.where(ratio == max_val), np.where(ratio==min_val))
    #print("index_max", index_max, "index_min", index_min)
    print("max",ratio[index_max], "=", masked_electron_density[index_max],"/", masked_gaussian_density[index_max])
    print("min",ratio[index_min], "=", masked_electron_density[index_min],"/", masked_gaussian_density[index_min])

    new_coefficients = np.empty(len(initial_coeffs))
    for i in range(0, len(initial_coeffs)):
        factor = initial_coeffs[i] * (constant_exponents[i] / np.pi)**(3/2)
        integrand = ratio * np.ravel(np.ma.asarray(np.exp(- constant_exponents[i] * np.power(grid, 2.))))
        new_coefficients[i] = factor * np.trapz(y=integrand, x=np.ravel(grid))
    return new_coefficients

def measure_error_by_integration_of_difference(true_model, approximate_model, grid):
        error = np.trapz(y=np.ravel(grid**2) * (np.absolute(np.ravel(true_model) - np.ravel(approximate_model))), x=np.ravel(grid))
        return error

def iterative_MBIS_method(initial_coeffs, constant_exponents, electron_density, grid, error=1e-5, masked_value=1e-6):
    current_error = 1e10
    old_coefficients = np.copy(initial_coeffs)
    new_coefficients = np.copy(initial_coeffs)

    #while current_error > error * len(initial_coeffs):
    list_of_obj_func = []
    for x in range(0, 100):
        temp = np.copy(new_coefficients)
        new_coefficients = update_coefficients(old_coefficients, constant_exponents, electron_density, grid)
        old_coefficients = np.copy(temp)
        current_error = np.sum(np.abs((old_coefficients - new_coefficients)))
        model = np.dot(np.exp(-constant_exponents * grid**2), new_coefficients)
        print("Currenterror", current_error,  measure_error_by_integration_of_difference(model, np.ravel(electron_density), np.ravel(grid)),"\n")

        exponential = np.exp(-constant_exponents * np.power(grid, 2.))
        assert exponential.shape[1] == len(constant_exponents)
        assert exponential.shape[0] == len(np.ravel(grid))
        gaussian_density = np.dot(exponential, new_coefficients)
        assert gaussian_density.shape[0] == len(np.ravel(grid))

        masked_gaussian_density = np.ma.asarray(gaussian_density)
        masked_electron_density = np.ma.asarray(np.ravel(electron_density))
        masked_gaussian_density[masked_gaussian_density <= masked_value] = 0.0

        log_ratio = np.log(masked_electron_density / masked_gaussian_density)
        list_of_obj_func.append(np.trapz(y=masked_electron_density * log_ratio, x=np.ravel(grid)))
    print(list_of_obj_func)
    return new_coefficients, current_error


from scipy.optimize import minimize


if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 200; NUMBER_OF_DIFFUSED_PTS = 300
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)

    def objective_func(initial_coefficients):
        exponential = np.exp(-be.UGBS_s_exponents * np.power(be.grid, 2.))
        assert exponential.shape[1] == len(be.UGBS_s_exponents)
        assert exponential.shape[0] == len(np.ravel(be.grid))
        gaussian_density = np.dot(exponential, initial_coefficients)
        assert gaussian_density.shape[0] == len(np.ravel(be.grid))

        masked_gaussian_density = np.ma.asarray(gaussian_density)
        masked_electron_density = np.ma.asarray(np.ravel(be.electron_density))
        masked_gaussian_density[masked_gaussian_density <= 1e-12] = 0.0

        log_ratio = np.log(masked_electron_density / masked_gaussian_density)
        result = (np.trapz(y=masked_electron_density * log_ratio, x=np.ravel(be.grid)))
        return result

    def jacobian(initial_coefficients):
        pass
    initial_coefficients = np.array([100. for x in range(0, 25)])
    EXPONENTS = be.UGBS_s_exponents.copy()

    cons = ({'type': 'eq'})
    bnds = tuple((0., np.inf) for x in initial_coefficients)

    new_coefficients = minimize(objective_func, initial_coefficients, method='SLSQP', bounds=bnds)
    print(new_coefficients)

    model = be.create_model(np.append(new_coefficients['x'], be.UGBS_s_exponents), 25)

    print(be.integrate_model_using_trapz(model))





