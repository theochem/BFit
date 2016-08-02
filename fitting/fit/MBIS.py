from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from fitting.fit.multi_objective import GaussianSecondObjTrapz, GaussianSecondObjSquared
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

def fixed_iteration_MBIS_method(coefficients, exponents,  fitting_obj, num_of_iterations=800,masked_value=1e-6, iprint=False):
    assert isinstance(fitting_obj, Fitting), "fitting object should be of type Fitting. It is instead %r" % type(fitting_obj)

    current_error = 1e10
    old_coefficients = np.copy(coefficients)
    new_coefficients = np.copy(coefficients)
    counter = 0

    for x in range(0, num_of_iterations):
        temp_coefficients = new_coefficients.copy()
        new_coefficients = update_coefficients(old_coefficients, exponents, fitting_obj.model_object.electron_density,
                                            fitting_obj.model_object.grid, masked_value=masked_value)

        exponents = update_exponents(old_coefficients, exponents, fitting_obj.model_object.electron_density
                                     , fitting_obj.model_object.grid, masked_value=masked_value)
        old_coefficients = temp_coefficients.copy()

        parameters = np.append(new_coefficients, exponents)
        approx_model = fitting_obj.model_object.create_model(parameters, len(exponents))
        current_error = fitting_obj.model_object.measure_error_by_integration_of_difference(fitting_obj.model_object.electron_density,
                                                                      approx_model)
        density_integration = fitting_obj.model_object.integrate_model_using_trapz(approx_model)
        diff_in_integration_error = fitting_obj.model_object.measure_error_by_difference_of_integration(
                                fitting_obj.model_object.electron_density,
                                approx_model)
        least_squares_error = fitting_obj.model_object.cost_function(parameters, len(exponents))
        if iprint:
            print(counter, density_integration,
                           least_squares_error,
                           current_error,
                           diff_in_integration_error)
        counter += 1

    return np.append(new_coefficients, exponents), current_error

def iterative_MBIS_method(coefficients, exponents, fitting_obj, threshold=1e-5, masked_value=1e-6, iprint=False):
    assert isinstance(fitting_obj, Fitting), "fitting object should be of type Fitting. It is instead %r" % type(fitting_obj)

    current_error = 1e10
    old_coefficients = np.copy(coefficients)
    new_coefficients = np.copy(coefficients)
    counter = 0
    error_array = [ [], [], [], [] ]
    if iprint:
        print("Counter", "Density Integration", "Least Squares Error", "Integration of the Difference", "Difference in Integration")
    while current_error > threshold:
        temp_coefficients = new_coefficients.copy()
        new_coefficients = update_coefficients(old_coefficients, exponents, fitting_obj.model_object.electron_density,
                                            fitting_obj.model_object.grid, masked_value=masked_value)

        exponents = update_exponents(old_coefficients, exponents, fitting_obj.model_object.electron_density
                                     , fitting_obj.model_object.grid, masked_value=masked_value)
        old_coefficients = temp_coefficients.copy()

        parameters = np.append(new_coefficients, exponents)
        approx_model = fitting_obj.model_object.create_model(parameters, len(exponents))
        current_error = be.measure_error_by_integration_of_difference(fitting_obj.model_object.electron_density,
                                                                      approx_model)

        density_integration = fitting_obj.model_object.integrate_model_using_trapz(approx_model)
        diff_in_integration_error = fitting_obj.model_object.measure_error_by_difference_of_integration(
                                fitting_obj.model_object.electron_density,
                                approx_model)
        least_squares_error = fitting_obj.model_object.cost_function(parameters, len(exponents))
        error_array[0].append(density_integration)
        error_array[1].append(least_squares_error)
        error_array[2].append(current_error)
        error_array[3].append(diff_in_integration_error)

        if iprint:
            print(counter, density_integration,
                           least_squares_error,
                           current_error,
                           diff_in_integration_error)
        counter += 1
    return parameters, np.array(error_array)


import matplotlib.ticker as mtick
from matplotlib import rcParams


def plot_density_sections(dens, prodens, points, title='None'):
    '''
    '''
    # choose fonts
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']

    # plotting intervals
    sector = 1.e-5
    conditions = [points <= sector,
                  np.logical_and(points > sector, points <= 100 * sector),
                  np.logical_and(points > 100 * sector, points <= 1000 * sector),
                  np.logical_and(points > 1000 * sector, points <= 1.e5 * sector),
                  np.logical_and(points > 1.e5 * sector, points <= 2.e5 * sector),
                  np.logical_and(points > 2.e5 * sector, points <= 5.0),
                  np.logical_and(points > 5.0, points <= 10.0)]

    # plot within each interval
    for cond in conditions:
        # setup figure
        fig, axes = plt.subplots(2, 1)
        fig.suptitle(title, fontsize=12, fontweight='bold')

        # plot true & model density
        ax1 = axes[0]
        ax1.plot(points[cond], dens[cond], 'ro', linestyle='-', label='True')
        ax1.plot(points[cond], prodens[cond], 'bo', linestyle='--', label='Approx')
        ax1.legend(loc=0, frameon=False)
        xmin, xmax = np.min(points[cond]), np.max(points[cond])
        ax1.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        ymin, ymax = np.min(dens[cond]), np.max(dens[cond])
        ax1.set_yticks(ticks=np.linspace(ymin, ymax, 5))
        ax1.set_ylabel('Density')
        if np.any(points[cond] < 1.0):
            ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # Hide the right and top spines
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        ax1.grid(True, zorder=0, color='0.60')

        # plot difference of true & model density
        ax2 = axes[1]
        ax2.plot(points[cond], dens[cond] - prodens[cond], 'ko', linestyle='-')
        ax2.set_xticks(ticks=np.linspace(xmin, xmax, 5))
        ax2.set_ylabel('True - Approx')
        ax2.set_xlabel('Distance from the nueleus')
        ymin, ymax = np.min(dens[cond] - prodens[cond]), np.max(dens[cond] - prodens[cond])
        ax2.set_yticks(ticks=np.linspace(ymin, ymax, 5))
        if np.any(points[cond] < 1.0):
            ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        # Hide the right and top spines
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.grid(True, zorder=0, color='0.60')

        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    ELEMENT_NAME = "ag"
    ATOMIC_NUMBER = 47
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


    be_abs = GaussianSecondObjTrapz(ELEMENT_NAME, column_grid_points, file_path, atomic_number=47, lam=1., lam2=0.)
    fitting_abs = Fitting(be_abs)
    be_squared = GaussianSecondObjSquared(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=0.1, lam2=0.)
    be_abs = GaussianSecondObjTrapz(ELEMENT_NAME, column_grid_points, file_path, atomic_number=4, lam=1., lam2=0.)
    fitting_squared = Fitting(be_squared)

    parameters = fitting_squared.optimize_using_slsqp(np.append(coeffs, exps), len(exps), additional_constraints=True)
    coeffs = parameters[0:len(parameters)//2]
    exps = parameters[len(parameters)//2:]
    print(ELEMENT_NAME + " True Density Integration " + str(be.integrated_total_electron_density))

    coeffs[coeffs == 0.] = 1E-6
    exps[exps == 0.] = 1E-6

    parameters, error_array = iterative_MBIS_method(coeffs, exps, fitting_obj, threshold=0.046, iprint=True)
    approx_model = be.create_model(parameters, len(exps))
    plot_density_sections(np.ravel(be.electron_density), approx_model, np.ravel(be.grid), title="Silver  MBIS Method")

    fig, axes = plt.subplots(2, 1)
    fig.suptitle("Integration and Cost Function Value", fontsize=12, fontweight='bold')
    ax1 = axes[0]
    ax1.plot(error_array[0], 'r')
    ax1.set_ylabel("Density Integration of Approx")
    ax1.set_xlabel("Number of Iteration")

    ax2 = axes[1]
    ax2.plot(error_array[1], 'r')
    ax2.set_ylabel("Least Squares Value of Carbon")
    ax2.set_xlabel("Number of Iteration")
    plt.savefig(ELEMENT_NAME + " NNLS_inte_and_Cost_func.png")
    plt.show()

    fig, axes = plt.subplots(2, 1)
    fig.suptitle("Different Integration Error Measures", fontsize=12, fontweight='bold')
    ax1 = axes[0]
    ax1.plot(error_array[2], 'r')
    ax1.set_ylabel("Int. of the Diff. ")

    ax2=axes[1]
    ax2.plot(error_array[3], 'r')
    ax2.set_ylabel("Diff In Density Int. ")
    plt.show()
    plt.savefig(ELEMENT_NAME + " NNLS_Integration_Error_Measures.png")
    plt.show()




