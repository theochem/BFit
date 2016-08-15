from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from fitting.fit.multi_objective import GaussianSecondObjTrapz, GaussianSecondObjSquared
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import matplotlib.ticker as mtick
from matplotlib import rcParams
import numpy as np

four_pi_exp = 1
def get_normalized_constants(exponent):
    return 4 * (exponent**(3/2))/(np.sqrt(np.pi))#(exponent / np.pi)**(3/2) #

def normalized_gaussian_model(coefficients, exponents, grid):
    assert len(coefficients) == len(exponents)
    exponential = np.exp(-exponents * np.power(grid, 2.))
    assert exponential.ndim == 2.
    assert exponential.shape[0] == grid.shape[0]
    assert exponential.shape[1] == len(exponents)
    normalized_constants = np.array([get_normalized_constants(exponents[x]) for x in range(0,len(coefficients))])
    normalized_coefficients = coefficients * normalized_constants
    assert len(normalized_coefficients) == len(normalized_constants) == len(coefficients)
    gaussian_density = np.dot(exponential, normalized_coefficients)

    return gaussian_density

def objective_function(coefficients, exponents, weights, electron_density, grid):
    normalized_gaussian_density = normalized_gaussian_model(coefficients, exponents, grid)
    masked_gaussian_density = np.ma.asarray(normalized_gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    #plt.semilogy(np.ravel(grid), normalized_gaussian_density, 'ro')
    #plt.semilogy(np.ravel(grid), electron_density)
    #plt.show()
    masked_grid_squared = np.ma.asarray(np.ravel(np.power(grid, 2.)))
    ratio = masked_electron_density / masked_gaussian_density
    log_ratio = np.log(ratio)
    return (4 * np.pi)**four_pi_exp * np.trapz(y=weights * masked_electron_density * log_ratio * masked_grid_squared , x=np.ravel(grid))

def lagrange_multiplier(atomic_number, weights, electron_density, grid):
    masked_grid_squared = np.ma.asarray(np.ravel(np.power(grid, 2.)))
    masked_grid = np.ma.asarray(np.ravel(grid))
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))

    return ((4 * np.pi)**four_pi_exp * np.trapz(y=np.ravel(weights) * masked_electron_density * masked_grid_squared, x=masked_grid)) / atomic_number

def update_coefficients(coefficients, exponents, weights, electron_density, grid, lam):
    assert np.all(coefficients > 0) == True, "Coefficinets should be positive non zero, " + str(coefficients)
    assert np.all(exponents > 0) == True, "Coefficinets should be positive non zero, " + str(coefficients)
    masked_grid_squared = np.ma.asarray(np.ravel(np.power(grid, 2.)))

    gaussian_density = normalized_gaussian_model(coefficients, exponents, grid)

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    ratio = np.ravel(weights) * masked_electron_density /  masked_gaussian_density

    new_coefficients = np.empty(len(coefficients))
    for i in range(0, len(coefficients)):
        prefactor = (4 * np.pi)**four_pi_exp * get_normalized_constants(exponents[i])
        integrand = ratio *  np.exp(-exponents[i] * np.ravel(masked_grid_squared)) * np.ravel(masked_grid_squared)
        new_coefficients[i] = new_coefficients[i] * prefactor * np.trapz(y=integrand, x=np.ravel(grid))  #/ lam
    return new_coefficients

def update_exponents(coefficients, exponents, weights, electron_density, grid, lam):
    assert np.all(coefficients > 0) == True, "Coefficinets should be positive non zero, " + str(coefficients)
    assert np.all(exponents > 0) == True, "Coefficinets should be positive non zero, " + str(coefficients)
    grid_quadr = np.ma.asarray(np.ravel(np.power(grid, 4.)))
    masked_grid_squared = np.ma.asarray(np.ravel(np.power(grid, 2.)))

    gaussian_density = normalized_gaussian_model(coefficients, exponents, grid)

    masked_gaussian_density = np.ma.asarray(gaussian_density)
    masked_electron_density = np.ma.asarray(np.ravel(electron_density))
    ratio_with_weights = np.ravel(weights) * masked_electron_density /  masked_gaussian_density

    new_exponents = np.empty(len(exponents))
    for i in range(0, len(exponents)):
        prefactor_coeff = 4 * np.pi * get_normalized_constants(exponents[i])
        integrand_coeff = ratio_with_weights *  np.exp(-exponents[i] * np.ravel(masked_grid_squared)) * np.ravel(masked_grid_squared)
        coeff =  prefactor_coeff * np.trapz(y=integrand_coeff, x=np.ravel(grid))

        prefactor = 4 * np.pi * get_normalized_constants(exponents[i])
        integrand = ratio_with_weights * grid_quadr * np.ma.asarray(np.exp(-exponents[i] * np.ravel(np.power(grid, 2.))))
        new_exponents[i] = 3 * coeff  / (2 * prefactor * np.trapz(y=integrand, x=np.ravel(grid)))
    return new_exponents

def integrate_error(coefficients, exponents, grid):
    model = normalized_gaussian_model(coefficients, exponents, grid)
    return  (4 * np.pi)**0 * np.trapz(y=model * np.ravel(np.power(grid, 2.)), x=np.ravel(grid))

def integrate_error2(coefficients, exponents, electron_density, grid):
    gaussian_model = normalized_gaussian_model(coefficients, exponents, grid)
    return np.trapz(y=np.ravel(np.power(grid, 2.)) * np.abs(gaussian_model - np.ravel(electron_density)) ,x=
                    np.ravel(grid))

def MBIS_method(threshold, coefficients, exponents, weights, electron_density, grid, atomic_number):
    lam = lagrange_multiplier(atomic_number, weights, electron_density, grid)
    print(lam)
    counter = 0
    old_coeffs = coefficients.copy()
    new_coeffs = coefficients.copy() + 1.
    old_exps = exponents.copy()
    new_exps = exponents.copy() + 1
    for x in range(0, 800):

        while np.any(np.abs((new_coeffs - old_coeffs)) > threshold):
            temp_coeffs = new_coeffs.copy()
            new_coeffs = update_coefficients(old_coeffs, exponents, weights, electron_density, grid, lam)
            new_coeffs[new_coeffs == 0.] = 1e-6
            assert np.all(new_coeffs > 0) == True, "Coefficinets should be positive non zero, " + str(new_coeffs)
            old_coeffs = temp_coeffs


            print(counter, integrate_error(new_coeffs, exponents, grid), objective_function(new_coeffs, exponents,
                                                                                              weights, electron_density,
                                                                                              grid), np.sum(new_coeffs),
                                                        integrate_error2(new_coeffs, exponents, electron_density, grid),
                                                        np.trapz(y=np.ravel(electron_density) * np.ravel(np.power(grid, 2.)),x=np.ravel(grid)))

            counter += 1
        while np.any(np.abs((new_exps - old_exps) > threshold)):
            temp_exps = new_exps.copy()
            exponents = update_exponents(old_coeffs, exponents, weights, electron_density, grid, lam)
            old_exps = temp_exps
            temp_exps[temp_exps == 0.] = 1e-6
            print(counter, integrate_error(new_coeffs, exponents, grid), objective_function(new_coeffs, exponents,
                                                                                              weights, electron_density,
                                                                                              grid), np.sum(new_coeffs),
                                                        integrate_error2(new_coeffs, exponents, electron_density, grid))
    return new_coeffs, exponents

def plot_density_sections(dens, prodens, points, ELEMENT_NAME, title='None'):
    '''
    '''
    # choose fonts
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Times New Roman']
    points *= 0.5291772082999999
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
    counter = 1
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
            pass
        # Hide the right and top spines
        ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        ax2.grid(True, zorder=0, color='0.60')

        plt.tight_layout()
        plt.savefig(ELEMENT_NAME + " Farnax_exps_plot" + str(counter))
        plt.show()
        plt.close()
        counter += 1

if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4
    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 300; NUMBER_OF_DIFFUSED_PTS = 400
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS)
    column_grid_points = np.reshape(row_grid_points[1:], (len(row_grid_points) - 1, 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)
    exps = be.UGBS_s_exponents
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    """
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
    """

    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    coeffs[coeffs == 0.] = 1e-6

    parameters = np.append(coeffs, exps)
    weights = 1 / (4 * np.pi * np.ones(len(row_grid_points)-1))#np.ravel(np.power(be.grid, 2.)))
    #weights[0] = 0.#1.7976931348623157e+300
    weights = np.ma.asarray(np.ones(len(row_grid_points)-1))#weights) #
    print("objective", objective_function(coeffs, exps, weights, be.electron_density, be.grid))

    from scipy.optimize import minimize
    def constraint(x,):
        leng = len(x) // 2
        return np.sum(x[0:leng]) - ATOMIC_NUMBER
    def objective_function2(para, weights, electron_density, grid):
        coefficients = para[0:len(para)//2]
        exponents = para[len(para)//2:]
        normalized_gaussian_density = normalized_gaussian_model(coefficients, exponents, grid)
        masked_gaussian_density = np.ma.asarray(normalized_gaussian_density)
        masked_electron_density = np.ma.asarray(np.ravel(electron_density))
        #plt.semilogy(np.ravel(grid), normalized_gaussian_density, 'ro')
        #plt.semilogy(np.ravel(grid), electron_density)
        #plt.show()
        masked_weights = np.ma.asarray(weights)
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(grid, 2.)))
        ratio = masked_electron_density / masked_gaussian_density
        return (4 * np.pi)**four_pi_exp * np.trapz(y=masked_weights * masked_electron_density * np.log(ratio) * masked_grid_squared , x=np.ravel(grid))

    bounds = np.array([(0.0, np.inf) for x in range(0, len(coeffs)*2)], dtype=np.float64)
    cons = ({'type': 'eq', 'fun':constraint})
    parameters = minimize(objective_function2, np.append(coeffs, exps), method="slsqp", constraints=cons, bounds=bounds,
                          args=(weights, be.electron_density, be.grid))
    parameters = parameters['x']
    coeffs = parameters[0:len(parameters)//2]
    coeffs[coeffs == 0.] = 1e-6

    exps = parameters[len(parameters)//2 :]
    print(np.sum(coeffs), integrate_error(coeffs, exps, be.grid), integrate_error2(coeffs, exps, be.electron_density, be.grid))
    MBIS_method(1e-6, coeffs, exps, weights, be.electron_density, be.grid, ATOMIC_NUMBER)