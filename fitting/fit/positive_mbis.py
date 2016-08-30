from __future__ import division
import numpy as np


def get_objective_function( electron_dens, grid_obj, model, weights):
    log_ratio_of_models = np.log(electron_dens / np.ma.asarray(model))**2.
    return grid_obj.integrate(electron_dens * weights * log_ratio_of_models)


def integrate_model_with_four_pi(grid_obj, model):
    return grid_obj.integrate(model)


def goodness_of_fit_grid_squared(electron_dens, grid_obj, model):
    return grid_obj.integrate(np.abs(model - np.ravel(electron_dens))) / (4 * np.pi)


def goodness_of_fit(electron_dens, grid_obj, model):
    return grid_obj.integrate(np.abs(model - np.ravel(electron_dens)) / grid_obj.radii**2.) / (
    4 * np.pi)


def lagrange_multiplier(electron_dens, model, radial_grid, atomic_number, weights=1.):
    ratio = np.ma.asarray(electron_dens) / np.ma.asarray(model)
    log_ratio = np.log(ratio)
    integrand = weights * electron_dens * log_ratio
    integrand = np.ma.filled(integrand, 0.)
    return 2. * radial_grid.integrate(integrand) / atomic_number


def get_normalized_gaussian_density(coeff_arr, exp_arr, radii):
    radii = np.reshape(radii, (len(radii), 1.))
    exponential = np.exp(-exp_arr * np.power(radii, 2.))
    assert exponential.shape == (len(radii), len(exp_arr))

    normalized_exps = (exp_arr / np.pi) ** (3. / 2.)
    normalized_coeffs = coeff_arr * normalized_exps

    assert normalized_coeffs.ndim == 1.
    normalized_gaussian_density = np.dot(exponential, normalized_coeffs)
    index_where_zero_occurs = np.where(normalized_gaussian_density == 0.)
    if len(index_where_zero_occurs[0]) != 0:
        normalized_gaussian_density[index_where_zero_occurs] = \
            normalized_gaussian_density[index_where_zero_occurs[0][0] - 1]
    return normalized_gaussian_density

def update_coeffs(coeff_arr, exp_arr,  electron_dens,radial_obj,atomic_number, weights=1):
    model = get_normalized_gaussian_density(coeff_arr, exp_arr, radial_obj.radii)
    lagrange = lagrange_multiplier(electron_dens, model, radial_obj, atomic_number, weights=weights)
    print(lagrange)
    masked_ed = np.ma.asarray(electron_dens)
    ratio = masked_ed / model
    log_ratio = np.log(ratio)
    pre_integrand = log_ratio * ratio * masked_ed
    pre_integrand = np.ma.filled(pre_integrand, 0.)
    new_coeff = coeff_arr.copy()
    for i, c in enumerate(coeff_arr):
        post_inte = pre_integrand * np.ma.asarray(np.exp(-exp_arr[i] * radial_obj.radii**2.))
        new_coeff[i] *= 2. * radial_obj.integrate(post_inte * weights / model)
        new_coeff[i] *= (exp_arr[i] / np.pi)**(3. / 2.) / lagrange

    print(get_objective_function(electron_dens, radial_obj, model, weights), np.sum(new_coeff))
    return new_coeff


def optimize_using_slsqp(parameters):
    from scipy.optimize import minimize
    def constraint(x, *args):
        leng = len(x) // 2
        return np.sum(x[0:leng]) - self.atomic_number

    cons = (  # {'type':'eq','fun':integration_constraint},
        {'type': 'eq', 'fun': constraint})
    bounds = np.array([(0.0, np.inf) for x in range(0, len(parameters))], dtype=np.float64)
    f_min_slsqp = minimize(get_objective_function(), x0=parameters, method="SLSQP",
                           bounds=bounds, constraints=cons,
                           jac=False)

    parameters = f_min_slsqp['x']
    print(f_min_slsqp)
    return parameters

if __name__ == "__main__":
    ATOMIC_NUMBER = 9
    ELEMENT_NAME = "f"
    USE_HORTON = False
    USE_FILLED_VALUES_TO_ZERO = True
    THRESHOLD_COEFF = 1e-2
    THRESHOLD_EXPS = 40
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data/examples//" + ELEMENT_NAME




    NUMB_OF_CORE_POINTS = 400;
    NUMB_OF_DIFFUSE_POINTS = 500
    from fitting.density.radial_grid import Radial_Grid
    from fitting.density.atomic_slater_density import Atomic_Density

    radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMB_OF_CORE_POINTS, NUMB_OF_DIFFUSE_POINTS, [50, 75, 100],
                              filled=USE_FILLED_VALUES_TO_ZERO)




    from fitting.density import Atomic_Density
    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet

    from fitting.fit.model import Fitting
    atomic_gaussian = GaussianTotalBasisSet(ELEMENT_NAME, np.reshape(radial_grid.radii,
                                                                    (len(radial_grid.radii), 1)), file_path)
    weights = 1 / (np.pi * radial_grid.radii**2.) #np.exp(-0.01 * radial_grid.radii**2.)

    fitting_obj = Fitting(atomic_gaussian)
    #mbis = TotalMBIS(ELEMENT_NAME, ATOMIC_NUMBER, radial_grid, atomic_density.electron_density, weights=weights)

    exps = atomic_gaussian.UGBS_s_exponents[:-3]
    coeffs = fitting_obj.optimize_using_nnls(atomic_gaussian.create_cofactor_matrix(exps))
    print(exps)
    coeffs[coeffs == 0.] = 1e-6

    model = get_normalized_gaussian_density(coeffs, exps, radial_grid.radii)
    print(lagrange_multiplier(atomic_density.electron_density, model, radial_grid, ATOMIC_NUMBER))

    for x in range(0, 50):
        coeffs = update_coeffs(coeffs, exps, atomic_density.atomic_density(), radial_grid, ATOMIC_NUMBER)
        model = get_normalized_gaussian_density(coeffs, exps, radial_grid.radii)