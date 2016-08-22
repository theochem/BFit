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
        normalized_coeffs = self.get_normalized_coefficients(coeff_arr, exp_arr)
        assert normalized_coeffs.ndim == 1.
        return np.dot(exponential, normalized_coeffs)

    def get_integration_factor(self, exponent, masked_normed_gaussian, upt_exponent=False):
        ratio = self.masked_electron_density / masked_normed_gaussian

        integrand = self.weights * ratio * np.exp(-exponent * self.masked_grid_squared)
        if upt_exponent:
            integrand *= self.masked_grid_squared
        return self.get_normalization_constant(exponent) * self.grid_obj.integrate(integrand)

    def update_coefficients(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))

        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_integration_factor(exp_arr[i], masked_normed_gaussian)
            new_coeff[i] /= self.lagrange_multiplier
        return new_coeff

    def update_exponents(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        ratio = self.masked_electron_density / masked_normed_gaussian

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            new_exps[i] = 3 * self.lagrange_multiplier
            integration = self.get_integration_factor(exp_arr[i], masked_normed_gaussian, upt_exponent=True)
            assert integration != 0, "Integration of the integrand is zero."
            assert not np.isnan(integration), "Integration should not be nan"
            new_exps[i] /= ( 2 * integration)
        return new_exps

    def get_normalization_constant(self, exponent):
        return (exponent / np.pi) **(3/2)

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
                    print(counter, integration_model_four_pi, sum_of_coeffs, \
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
                print(counter, integration_model_four_pi, sum_of_coeffs, \
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

    def new_method(self, numb_of_funcs,probability_distribution_of_a, prob_dis_b):
        coeffs = self.atomic_number * probability_distribution_of_a
        assert np.dot(probability_distribution_of_a[:-1], prob_dis_b) < 1
        assert len(prob_dis_b) == len(probability_distribution_of_a[:-1])
        b_n = (1 - np.dot(probability_distribution_of_a[:-1], prob_dis_b) ) / probability_distribution_of_a[-1]
        assert b_n > 0
        print("b_n", b_n)
        prob_dis_b = np.append(prob_dis_b, b_n)

        exponents = np.pi * (prob_dis_b * self.electron_density[0] / self.atomic_number)**(2./3.)
        print("should be one", np.dot(coeffs, exponents))
        return coeffs, exponents


    def get_new_density(self, coeff, exps):
        exponential = np.exp(-exps * np.power(self.grid_points, 2.))
        normalized_coeffs = self.get_normalized_coefficients(coeff, exps)
        normalized_coeffs *= exps
        assert normalized_coeffs.ndim == 1.
        return np.dot(exponential, normalized_coeffs)

    def C(self, coeffs, exps):
        factor = - 2. / (3. * self.electron_density[0])
        U = (3.* self.atomic_number / 2.)
        ratio = np.log(self.masked_electron_density / self.get_normalized_gaussian_density(coeffs, exps))
        U -= self.grid_obj.integrate(ratio * self.get_new_density(coeffs, exps) * self.masked_grid_squared)
        return factor * U

    def new_method2(self, coeffs, exps):
        old_coeffs = coeffs.copy()
        new_coeffs = coeffs.copy()
        for x in range(0, 10):
            old_coeffs = new_coeffs.copy()
            for i, coeff in enumerate(old_coeffs):
                ratio = np.log(self.masked_electron_density / self.get_normalized_gaussian_density(coeffs, exps))
                factor = self.grid_obj.integrate(ratio * self.get_normalization_constant(exps[i]) * np.exp(-exps[i] * self.masked_grid_squared))
                factor += ( self.C(coeffs, exps) * ((self.electron_density[0] / self.atomic_number) - self.get_normalization_constant(exps[i])))
                new_coeffs[i] = coeff * factor


            model = self.get_normalized_gaussian_density(new_coeffs, exps)
            print(self.get_descriptors_of_model(model), model[0], self.electron_density[0])

    def new_method3(self, numb_of_funcs, prob_dist):
        coeffs = prob_dist * self.atomic_number
        current_atomic_number = self.atomic_number
        current_electron_density = self.electron_density

        assert (np.sum(coeffs) - self.atomic_number) < 1e-3
        self.atomic_number = coeffs[0]
        print(coeffs[0])
        self.lagrange_multiplier = self.get_lagrange_multiplier()
        one_coeffs, one_exps = self.run(1e-3, 1e-2, np.array([coeffs[0]]), np.array([100.]), iprint=True)

        all_coeffs, all_exps = one_coeffs.copy(), one_exps.copy()

        for i in range(1, len(coeffs)):
            print("------------", i, "-----------")
            self.electron_density -= self.get_normalized_gaussian_density(one_coeffs, one_exps)
            self.electron_density = np.abs(self.electron_density)
            self.masked_electron_density = np.ma.asarray(np.abs(self.electron_density))
            print((one_coeffs, one_exps))
            self.atomic_number = coeffs[i]
            self.lagrange_multiplier = self.get_lagrange_multiplier()
            one_coeffs, one_exps = self.run(1e-3, 1e-2, np.array([coeffs[i]]), np.array([10000.]), iprint=True)

            all_coeffs, all_exps = np.append(all_coeffs, one_coeffs), np.append(all_exps, one_exps)


            print()
            print()
            model = self.get_normalized_gaussian_density(all_coeffs, all_exps)
            print(self.get_descriptors_of_model(model), all_coeffs, all_exps)
            print()
        self.electron_density = current_electron_density
        self.masked_electron_density = np.ma.asarray(self.electron_density)
        self.atomic_number = current_atomic_number
        model = self.get_normalized_gaussian_density(all_coeffs, all_exps)
        print(self.get_descriptors_of_model(model))
        return all_coeffs, all_exps

if __name__ == "__main__":
    #################
    ## SET UP#######
    ###########
    ATOMIC_NUMBER = 29
    ELEMENT_NAME = "cu"

    import os
    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data\examples\\" + ELEMENT_NAME
    NUMB_OF_CORE_POINTS = 400; NUMB_OF_DIFFUSE_POINTS = 500
    from fitting.density.radial_grid import Radial_Grid
    from fitting.density.atomic_slater_density import Atomic_Density
    radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMB_OF_CORE_POINTS, NUMB_OF_DIFFUSE_POINTS, [50, 75, 100])

    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
    if ELEMENT_NAME != "h":
        atomic_gaussian = GaussianTotalBasisSet(ELEMENT_NAME, np.reshape(radial_grid.radii,
                                                                    (len(radial_grid.radii), 1)), file_path)
        from fitting.fit.model import Fitting
        exps = atomic_gaussian.UGBS_s_exponents
        fit = Fitting(atomic_gaussian)
        coeffs = fit.optimize_using_nnls(atomic_gaussian.create_cofactor_matrix(exps))
        coeffs[coeffs == 0.] = 1e-6
    else:
        exps = np.array( [ 2.50000000e-02 ,  5.00000000e-02 ,  1.00000000e-01,   2.00000000e-01,
                       4.00000000e-01,   8.00000000e-01,   1.60000000e+00,   3.20000000e+00,
                       6.40000000e+00,   1.28000000e+01,   2.56000000e+01,   5.12000000e+01,
                       1.02400000e+02,   2.04800000e+02,   4.09600000e+02,   8.19200000e+02,
                       1.63840000e+03,   3.27680000e+03,   6.55360000e+03,   1.31072000e+04,
                       2.62144000e+04,   5.24288000e+04,   1.04857600e+05,   2.09715200e+05,
                       4.19430400e+05,   8.38860800e+05,   1.67772160e+06,   3.35544320e+06,
                       6.71088640e+06,   1.34217728e+07,   2.68435456e+07,   5.36870912e+07,
                       1.07374182e+08,   2.14748365e+08,   4.29496730e+08,   8.58993459e+08,
                       1.71798692e+09,   3.43597384e+09])
        exponential = np.exp(-exps * np.power(atomic_density.GRID, 2.))
        from scipy.optimize import nnls
        coeffs = nnls(exponential, np.ravel(atomic_density.electron_density))[0]
        coeffs[coeffs == 0.] = 1e-12
    mbis = TotalMBIS(ELEMENT_NAME, ATOMIC_NUMBER, radial_grid, atomic_density.electron_density)

    weights = 1. / (4. * np.pi * np.power(radial_grid.radii, 2.))


    coeffs, exps = mbis.run(1e-3, 1e-2, coeffs, exps, iprint=True)
    #coeffs, exps = mbis.run_greedy(2. , 1e-2, 1e-1, iprint=True)
    print("Final Coeffs, Exps: ", coeffs, exps)

    ################
    ## NEW METHOD ##
    #################
    NUMBER_OF_FUNCS = 12

    def binomial_distrubituin(numb_of_funcs, probability):
        from scipy.misc import comb
        return np.array([comb(numb_of_funcs, x) * probability**x * (1-probability)**(numb_of_funcs - x) for x in range(0, numb_of_funcs)])

    print(np.sum(binomial_distrubituin(NUMBER_OF_FUNCS, 0.2)), binomial_distrubituin(NUMBER_OF_FUNCS, 0.5))
    probability_distribution_of_a = binomial_distrubituin(NUMBER_OF_FUNCS, 0.2)#  np.array([1./NUMBER_OF_FUNCS for x in range(0,NUMBER_OF_FUNCS)])
    b_stuff = np.array([0.001 * (1.2525)**x for x in range(0, 24)])

    #coeffs, exps = mbis.new_method3(NUMBER_OF_FUNCS, probability_distribution_of_a)
