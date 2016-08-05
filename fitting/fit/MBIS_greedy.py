from fitting.fit.GaussianBasisSet import *
from fitting.fit.multi_objective import *
import numpy as np
from scipy.integrate import simps, trapz
from fitting.fit.MBIS import *

def next_list_of_exponents(exponents_array, factor):
    assert exponents_array.ndim == 1
    size = exponents_array.shape[0]
    all_choices_of_exponents = []

    for index, exp in np.ndenumerate(exponents_array):
        if index[0] == 0:
            exponent_array = np.insert(exponents_array, index, exp / factor )

        elif index[0] <= size:
            exponent_array = np.insert(exponents_array, index, (exponents_array[index[0] - 1] + exponents_array[index[0]])/2)
        all_choices_of_exponents.append(exponent_array)

        if index[0] == size - 1:
            exponent_array = np.append(exponents_array, np.array([ exp * factor] ))
            all_choices_of_exponents.append(exponent_array)
    return(all_choices_of_exponents)

def add_only_to_ends(exponents_array, factor, coefficient_array):
    assert exponents_array.ndim == 1
    size = exponents_array.shape[0]
    all_choices_of_exponents = []
    all_choices_of_coefficients = []

    all_choices_of_exponents.append(np.insert(exponents_array, 0, exponents_array[0]/factor))
    all_choices_of_exponents.append(np.append(exponents_array, exponents_array[size - 1] * factor))

    all_choices_of_coefficients.append(np.insert(coefficient_array, 0, coefficient_array[0]/factor))
    all_choices_of_coefficients.append(np.append(coefficient_array, coefficient_array[size - 1] * factor))

    return all_choices_of_exponents, all_choices_of_coefficients

def nnls_optimization_routine(exponents, fitting_obj):
    cofactor_matrix = fitting_obj.model_object.create_cofactor_matrix(exponents)
    coeffs = fitting_obj.optimize_using_nnls(cofactor_matrix)
    coeffs[coeffs == 0.] = 1e-6
    return coeffs

def analytically_solve_objective_function(electron_density, grid, weight):
    grid = np.copy(np.ravel(grid))
    grid_squared = np.copy(np.power(grid, 2.0))
    ln_of_electron_density = np.ma.log(np.ravel(electron_density))

    if(np.isnan(ln_of_electron_density).any()):
        print("ISNAN in the electron density")
    a = 2.0 * np.sum(weight)
    b = 2.0 * np.sum(weight * grid_squared)
    c = 2.0 * weight * np.ravel(ln_of_electron_density)

    c = c.sum()
    if(np.isnan(c)):
        print("C IS THE ISNAN")
    d = np.copy(b)
    e = 2.0 * np.sum(weight * np.power(grid, 4.0))
    f = 2.0 * weight * grid_squared * ln_of_electron_density

    f = f.sum()
    if(np.isinf(f)):
        print("F IS THE ISNAN")

    A = (b * f - c * e) / (b * d - a * e)
    B = (a * f - c * d) / (a * e - b * d)
    coefficient = np.exp(A)
    exponent = -B
    if exponent < 0:
        exponent = 0.0
    if(np.isnan(exponent)):
        print("ISNAN DETECTED In The Exponent")
    return(np.array([coefficient, exponent]))

def KL_objective_function(parameters, electron_density, grid):
    coeffs = parameters[:len(parameters)//2]
    exponents = parameters[len(parameters)//2:]
    exponential = np.exp(exponents * np.power(grid, 2.))
    gaussian_density = np.dot(exponential, coeffs)

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    masked_gaussian_density[masked_gaussian_density <= 1e-6] = 1e-6
    ratio = masked_electron_density / masked_gaussian_density
    return np.trapz(y=np.ravel(electron_density) * np.log(ratio), x=np.ravel(grid))

def gaussian_model(parameters, grid):
    coeffs = parameters[:len(parameters)//2]
    exponents = parameters[len(parameters)//2:]

    exponential = np.exp(-exponents * np.power(grid, 2.))
    gaussian_density = np.dot(exponential, coeffs)
    return gaussian_density

def analytically_solve_with_weights(electron_density, grid):
    WEIGHTS = [np.ones(np.shape(grid)[0]),
               np.ravel(electron_density),
               np.ravel(np.power(electron_density, 2.0))]
    best_error = 1e10
    best_parameter = None
    for weight in WEIGHTS:
        parameter = analytically_solve_objective_function(electron_density, grid, weight)
        objective_func = KL_objective_function(parameter, electron_density, grid)
        if objective_func < best_error:
            best_error = objective_func
            best_parameter = parameter

    return best_parameter

def join_parameters_together(parameters, added_parameters):
    coeffs = parameters[:len(parameters)//2]
    exponents = parameters[len(parameters)//2:]

    coeffs = np.append(coeffs, added_parameters[0])
    exponents = np.append(exponents, added_parameters[1])
    return np.append(coeffs, exponents)

def finding_first_five_parameters(electron_density, grid):

    first_parameter = analytically_solve_with_weights(electron_density, grid)
    print(first_parameter, electron_density[0])
    model = gaussian_model(first_parameter, grid)
    print(np.trapz(y=np.ravel(np.power(grid, 2.)) * model, x=np.ravel(grid)))

    new_electron_density = electron_density.copy() - np.reshape(model, (len(model), 1))
    best_params = first_parameter.copy()
    for x in range(0, 4):
        next_params = analytically_solve_with_weights(new_electron_density, grid)
        best_params = join_parameters_together(best_params, next_params)
        model = gaussian_model(first_parameter, grid)
        print(np.trapz(y=np.ravel(np.power(grid, 2.)) * model, x=np.ravel(grid)))
        new_electron_density = new_electron_density.copy() - np.reshape(model, (len(model), 1))
    print(len(best_params), best_params)
    return best_params





def greedy_MBIS_method(factor, desired_accuracy, fitting_obj):
    assert isinstance(fitting_obj, Fitting), "fitting object should be of type Fitting. It is instead %r" % type(fitting_obj)

    #Start out with 5 Coefficients
    print("STARTING WITH FIVE GAUSSIAN FUNCTIONS")
    five_exponents = fitting_obj.model_object.UGBS_s_exponents[0:5]
    five_coefficients = fitting_obj.optimize_using_nnls(fitting_obj.model_object.create_cofactor_matrix(five_exponents))
    five_parameters = np.append(five_coefficients, five_exponents)
    optimized_params = fitting_obj.optimize_using_slsqp(five_parameters, 5)
    approx_model = fitting_obj.model_object.create_model(optimized_params,  5)

    coefficients = optimized_params[0:5]
    exponents = optimized_params[5:]
    current_error = fitting_obj.model_object.measure_error_by_integration_of_difference(
        fitting_obj.model_object.electron_density,
        approx_model
    )
    print("Initial Error - ", current_error)
    print("\n START GREEDY METHOD")
    global_best_error = current_error
    global_best_parameters = optimized_params.copy()
    local_parameters = optimized_params.copy()
    counter = 6
    while current_error > desired_accuracy:
        next_set_of_exps, next_set_of_coeffs = add_only_to_ends(local_parameters[len(local_parameters)//2:], factor,
                                                                local_parameters[0:len(local_parameters)//2])

        local_best_error = 1e10
        local_best_parameters = None
        i = 0
        for exp in next_set_of_exps:
            coeffs = next_set_of_coeffs[i]
            print(coeffs)
            parameters, error  = fixed_iteration_MBIS_method(coeffs, exp, fitting_obj, num_of_iterations=2000, iprint=False)
            i += 1
            if error < local_best_error:
                local_best_error = error
                local_best_parameters = parameters.copy()

        if local_best_error < global_best_error:
            global_best_error = local_best_error
            global_best_parameters = parameters.copy()

        local_parameters = local_best_parameters.copy()
        model = fitting_obj.model_object.create_model(local_parameters, len(next_set_of_exps[0]))
        int = fitting_obj.model_object.integrate_model_using_trapz(model)
        int_error = fitting_obj.model_object.measure_error_by_integration_of_difference(
                                                    fitting_obj.model_object.electron_density,
                                                    model)
        print("Current best error for iteration - ", int_error, " Number of Functions - ", counter, " Integration - ", int)
        counter += 1
    print("\n BEST RESULTS ")
    print(global_best_parameters)
    print("BEST ERROR - ", global_best_error)
    return global_best_parameters, global_best_error

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
    be_squared = GaussianSecondObjSquared(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=0.1, lam2=0.)
    be_abs = GaussianSecondObjTrapz(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=1., lam2=0.)
    fitting_squared = Fitting(be_squared)
    fitting_abs = Fitting(be_abs)
    #params = fitting_squared.forward_greedy_algorithm(10000., 0.0001, be.electron_density)
    #print(params)
    fitting_obj = Fitting(be)
    #greedy_MBIS_method(1000., 0.0045, fitting_obj)
    finding_first_five_parameters(be.electron_density, be.grid)