from __future__ import division
from mbis_abc import MBIS_ABC
import numpy as np

class TotalMBIS(MBIS_ABC):
    def __init__(self, element_name, atomic_number, grid_obj, electron_density, weights=None):
        super(TotalMBIS, self).__init__(element_name, atomic_number, grid_obj, electron_density, weights=weights)

    def get_normalized_coefficients(self, coeff_arr, exp_arr):
        normalized_constants = self.get_all_normalization_constants(exp_arr)
        return coeff_arr * normalized_constants

    def get_normalized_gaussian_density(self, coeff_arr, exp_arr):
        exponential = np.exp(-exp_arr * np.power(self.grid_points, 2.))
        assert exponential.shape == (len(self.grid_points), len(exp_arr))
        normalized_coeffs = self.get_normalized_coefficients(coeff_arr, exp_arr)
        assert normalized_coeffs.ndim == 1.
        return np.dot(exponential, normalized_coeffs)

    def get_integration_factor(self, exponent, masked_normed_gaussian, upt_exponent=False):
        ratio = self.masked_electron_density / masked_normed_gaussian
        assert ratio.ndim == 1.
        integrand = ratio * np.exp(-exponent * self.masked_grid_squared)
        assert integrand.ndim == 1.
        if upt_exponent:
            integrand *= self.masked_grid_squared
            assert integrand.ndim == 1.
        return self.get_normalization_constant(exponent) * self.grid_obj.integrate(self.weights * integrand)

    def update_coefficients(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        assert masked_normed_gaussian.ndim == 1.
        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_integration_factor(exp_arr[i], masked_normed_gaussian)
            new_coeff[i] /= self.lagrange_multiplier

        return new_coeff

    def update_exponents(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            new_exps[i] = 3 * self.lagrange_multiplier
            integration = self.get_integration_factor(exp_arr[i], masked_normed_gaussian, upt_exponent=True)
            assert integration != 0, "Integration of the integrand is zero."
            assert not np.isnan(integration), "Integration should not be nan"
            new_exps[i] /= ( 2 * integration)
        return new_exps

    def get_normalization_constant(self, exponent):
        return (exponent / np.pi) **(3./2.)

    def get_new_coeffs_and_old_coeffs(self, coeff_arr, exp_arr):
        new_coeff = self.update_coefficients(coeff_arr, exp_arr)
        return new_coeff, coeff_arr

    def get_new_exps_and_old_exps(self, coeff_arr, exp_arr):
        new_exps = self.update_exponents(coeff_arr, exp_arr)
        return new_exps, exp_arr

    def run(self, threshold_coeff, threshold_exp, coeff_arr, exp_arr, iprint=False, iplot=False):
        old_coeffs = coeff_arr.copy() + threshold_coeff * 2.
        new_coeffs = coeff_arr.copy()
        old_exps = exp_arr.copy() + threshold_exp * 2.
        new_exps = exp_arr.copy()
        storage_of_errors = [["""Integration Using Trapz"""],
                             [""" goodness of fit"""],
                             [""" goof of fit with r^2"""],
                             [""" KL Divergence Formula"""]]
        previous_objective_func = 1e10
        current_objective_func = 1e4
        counter = 0
        while np.any(np.abs(new_exps - old_exps) > threshold_exp ) and np.abs(previous_objective_func - current_objective_func) > 1e-10:
            new_coeffs, old_coeffs = self.get_new_coeffs_and_old_coeffs(new_coeffs, new_exps)

            while np.any(np.abs(old_coeffs - new_coeffs) > threshold_coeff):
                new_coeffs, old_coeffs = self.get_new_coeffs_and_old_coeffs(new_coeffs, new_exps)

                if 0. in new_coeffs:
                    return new_coeffs, new_exps

                model = self.get_normalized_gaussian_density(new_coeffs , new_exps)
                sum_of_coeffs = np.sum(new_coeffs)
                integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared, objective_function = \
                        self.get_descriptors_of_model(model)
                if iprint:
                    print(counter, integration_model_four_pi, np.round(sum_of_coeffs), \
                          goodness_of_fit, goodness_of_fit_r_squared, \
                          objective_function,  True, np.max(np.abs(old_coeffs - new_coeffs)), model[0], self.electron_density[0])
                if iplot:
                    storage_of_errors[0].append(integration_model_four_pi)
                    storage_of_errors[1].append(goodness_of_fit)
                    storage_of_errors[2].append(goodness_of_fit_r_squared)
                    storage_of_errors[3].append(objective_function)
                counter += 1

            new_exps, old_exps = self.get_new_exps_and_old_exps(new_coeffs, new_exps)

            model = self.get_normalized_gaussian_density(new_coeffs, new_exps)
            sum_of_coeffs = np.sum(new_coeffs)
            integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared, objective_function = \
                        self.get_descriptors_of_model(model)
            temp_obj = current_objective_func
            current_objective_func = objective_function
            previous_objective_func = temp_obj
            if iprint:
                if counter % 100 == 0.:
                    for x in range(0, len(new_coeffs)):
                        print(new_coeffs[x], new_exps[x])
                print(counter, integration_model_four_pi, np.round(sum_of_coeffs), \
                      goodness_of_fit, goodness_of_fit_r_squared, \
                      objective_function, False, np.max(np.abs(new_exps - old_exps)), model[0], self.electron_density[0])
            if iplot:
                storage_of_errors[0].append(integration_model_four_pi)
                storage_of_errors[1].append(goodness_of_fit)
                storage_of_errors[2].append(goodness_of_fit_r_squared)
                storage_of_errors[3].append(objective_function)
            counter += 1
        if iplot:
            self.create_plots(storage_of_errors[0], storage_of_errors[1], storage_of_errors[2], storage_of_errors[3])
        return new_coeffs, new_exps

    def check_redundancies(self, coeffs, exps):
        for i, alpha in enumerate(exps):
            indexes_where_they_are_same = []
            for j in range(0, len(exps)):
                if i != j:
                    if np.abs(alpha - exps[j]) < 1e-5:
                        indexes_where_they_are_same.append(j)

            for index in indexes_where_they_are_same:
                coeffs[i] += coeffs[j]
            if len(indexes_where_they_are_same) != 0:
                print("-------- Redundancies found ---------")
                print()
                exps = np.delete(exps, indexes_where_they_are_same)
                coeffs = np.delete(coeffs, indexes_where_they_are_same)
        assert len(exps) == len(coeffs)
        return coeffs, exps

    def run_greedy(self,factor, threshold_coeff, threshold_exps, iprint=False):
        def get_next_possible_coeffs_and_exps(factor, coeffs, exps):
            size = exps.shape[0]
            all_choices_of_exponents = []
            all_choices_of_coeffs = []
            coeff_value=100.
            for index, exp in np.ndenumerate(exps):
                if index[0] == 0:
                    exponent_array = np.insert(exps, index, exp / factor )
                    coefficient_array = np.insert(coeffs, index, coeff_value)
                elif index[0] <= size:
                    exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]])/2)
                    coefficient_array = np.insert(coeffs, index, coeff_value)
                all_choices_of_exponents.append(exponent_array)
                all_choices_of_coeffs.append(coefficient_array)
                if index[0] == size - 1:
                    exponent_array = np.append(exps, np.array([ exp * factor] ))
                    all_choices_of_exponents.append(exponent_array)
                    all_choices_of_coeffs.append(np.append(coeffs, np.array([coeff_value])))
            return all_choices_of_coeffs, all_choices_of_exponents

        #######################################
        ##### SOLVE FOR ONE GAUSSIAN FUNCTION##
        ######################################
        coeffs = np.array([float(self.atomic_number)])
        print(coeffs)
        exps = np.array([0.034])
        coeffs, exps = self.run(1e-4, 1e-3, coeffs, exps, iprint=iprint)
        print("Single Best Coeffs and Exps: ", coeffs, exps)
        #######################################
        ##### ITERATION: NEXT GAUSSIAN FUNCS##
        ######################################
        num_of_functions = 1
        for x in range(0, 30):
            next_coeffs, next_exps = get_next_possible_coeffs_and_exps(factor, coeffs, exps)

            num_of_functions += 1

            best_local_found_objective_func = 1e10
            best_local_coeffs = None
            best_local_exps = None
            for i, exponents in enumerate(next_exps):
                exponents[exponents==0] = 1e-6
                next_coeffs[i][next_coeffs[i] == 0.] = 1e-12
                next_coeffs[i], exps = self.run(10., 1000., next_coeffs[i], exponents, iprint=False)
                objective_func = self.get_objective_function(self.get_normalized_gaussian_density(next_coeffs[i],
                                                                                               exponents))
                if objective_func < best_local_found_objective_func:
                    best_local_found_objective_func = objective_func
                    best_local_coeffs = next_coeffs[i]
                    best_local_exps = exponents

            print(num_of_functions,
                    self.get_descriptors_of_model(self.get_normalized_gaussian_density(best_local_coeffs, best_local_exps)))
            coeffs, exps = best_local_coeffs, best_local_exps
            coeffs, exps = self.run(threshold_coeff, threshold_exps, coeffs, exps, iprint=iprint)

            print(num_of_functions,
                    self.get_descriptors_of_model(self.get_normalized_gaussian_density(coeffs, exps)))
            print()
            coeffs, exps = self.check_redundancies(coeffs, exps)
            num_of_functions = len(coeffs)




if __name__ == "__main__":
    #################
    ## SET UP#######
    ###########
    ATOMIC_NUMBER = 29
    ELEMENT_NAME = "cu"
    USE_HORTON = False
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data/examples//" + ELEMENT_NAME
    if USE_HORTON:
        import horton
        rtf = horton.ExpRTransform(1.0e-2, 50, 900)
        radial_grid = horton.RadialGrid(rtf)
    else:
        NUMB_OF_CORE_POINTS = 400; NUMB_OF_DIFFUSE_POINTS = 500
        from fitting.density.radial_grid import Radial_Grid
        from fitting.density.atomic_slater_density import Atomic_Density
        radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMB_OF_CORE_POINTS, NUMB_OF_DIFFUSE_POINTS, [50, 75, 100])

    from fitting.density import Atomic_Density
    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet

    from fitting.fit.model import Fitting
    atomic_gaussian = GaussianTotalBasisSet(ELEMENT_NAME, np.reshape(radial_grid.radii,
                                                                    (len(radial_grid.radii), 1)), file_path)

    fitting_obj = Fitting(atomic_gaussian)
    weights = None #1. / (4. * np.pi * np.power(radial_grid.radii, 2.))
    mbis = TotalMBIS(ELEMENT_NAME, ATOMIC_NUMBER, radial_grid, atomic_density.electron_density, weights=weights)

    exps = atomic_gaussian.UGBS_s_exponents
    coeffs = fitting_obj.optimize_using_nnls(atomic_gaussian.create_cofactor_matrix(exps))
    coeffs[coeffs == 0.] = 1e-12
    #coeffs = np.array([mbis.atomic_number/len(exps) for x in range(0, len(exps))])
    print(radial_grid.integrate(mbis.electron_density))
    print(radial_grid.integrate(mbis.get_normalized_gaussian_density(coeffs, exps)))
    coeffs, exps = mbis.run(1e-4, 1e-3, coeffs, exps, iprint=True)

    #coeffs, exps = mbis.run_greedy(2. , 1e-2, 1e-1, iprint=True)
    print("Final Coeffs, Exps: ", coeffs, exps )
