from normalized_mbis import TotalMBIS
import numpy as np
class ValenceMBIS(TotalMBIS):
    def __init__(self, electron_density, grid_obj, weights, atomic_number, element_name):
        super(TotalMBIS, self).__init__(electron_density, grid_obj, weights, atomic_number, element_name)
        self.masked_grid_quadrupled = np.power(self.masked_grid_squared, 2.)

    def get_normalization_constant_valence(self, exponent):
        return (2 * exponent**(5/2) / (3 * np.pi**(3/2)))

    def get_normalized_coefficients_valence(self, coeff_val_arr, exp_val_arr):
        return coeff_val_arr * self.get_normalization_constant_valence(exp_val_arr)

    def get_normalized_gaussian_density(self, coeff_arr, coeff2_arr, exp_arr, exp2_arr):
        s_exponential = np.exp(-exp_arr * np.power(self.grid_points, 2.))
        p_exponential = np.exp(-exp2_arr * np.power(self.grid_points, 2.))
        normalized_coeffs = self.get_normalized_coefficients(coeff_arr, exp_arr)
        normalized_coeffs2 = self.get_normalized_coefficients_valence(coeff2_arr, exp2_arr)

        s_gaussian_model = np.dot(s_exponential, normalized_coeffs)
        p_gaussian_model = np.dot(p_exponential, normalized_coeffs2)
        p_gaussian_model = np.ravel(p_gaussian_model) * self.masked_grid_squared

        return s_gaussian_model + p_gaussian_model

    def get_integration_factor_valence(self, exponent, masked_normed_gaussian, upt_exponents=False):
        ratio = self.masked_electron_density / masked_normed_gaussian
        normalized_ratio_weights = self.get_normalization_constant_valence(exponent) * ratio * self.weights

        if upt_exponents:
            return self.grid_obj.integrate(normalized_ratio_weights * self.masked_grid_quadrupled *\
                                            np.exp(-exponent * self.masked_grid_squared))
        return self.grid_obj.integrate(normalized_ratio_weights *
                                       self.masked_grid_squared * np.exp(-exponent * self.masked_grid_squared))

    def update_coefficients(self, coeff_arr, coeff2_arr, exp_arr, exp2_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        assert np.all(coeff2_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff2_arr
        assert np.all(exp2_arr > 0), "Exponents should be positive. Instead we got %r" % exp2_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr,
                                                                                    coeff2_arr,
                                                                                    exp_arr,
                                                                                    exp2_arr))
        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_integration_factor(exp_arr[i], masked_normed_gaussian)
            new_coeff[i] /= self.lagrange_multiplier
        return new_coeff

    def update_valence_coefficients(self,  coeff_arr, coeff2_arr, exp_arr, exp2_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        assert np.all(coeff2_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff2_arr
        assert np.all(exp2_arr > 0), "Exponents should be positive. Instead we got %r" % exp2_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr,
                                                                                    coeff2_arr,
                                                                                    exp_arr,
                                                                                   exp2_arr))
        new_coeff2 = coeff2_arr.copy()
        for i in range(0, len(coeff2_arr)):
            new_coeff2 *= self.get_integration_factor_valence(exp2_arr[i], masked_normed_gaussian)
            new_coeff2 /= self.lagrange_multiplier
        return new_coeff2

    def update_exponents(self, coeff_arr, coeff2_arr, exp_arr, exp2_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        assert np.all(coeff2_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff2_arr
        assert np.all(exp2_arr > 0), "Exponents should be positive. Instead we got %r" % exp2_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr,
                                                                                    coeff2_arr,
                                                                                    exp_arr,
                                                                                    exp2_arr))
        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            new_exps[i] = 3 * self.lagrange_multiplier
            integration = self.get_integration_factor(exp_arr[i], masked_normed_gaussian, upt_exponent=True)
            assert integration != 0, "Integration of the integrand is zero."
            assert not np.isnan(integration), "Integration should not be nan"
            new_exps[i] /= ( 2 * integration)
        return new_exps

    def update_valence_exponents(self,  coeff_arr, coeff2_arr, exp_arr, exp2_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        assert np.all(coeff2_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff2_arr
        assert np.all(exp2_arr > 0), "Exponents should be positive. Instead we got %r" % exp2_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr,
                                                                                    coeff2_arr,
                                                                                    exp_arr,
                                                                                    exp2_arr))
        new_exps2 = exp2_arr.copy()
        for i in range(0, len(exp2_arr)):
            new_exps2[i] = 5 * self.lagrange_multiplier
            integration = self.get_integration_factor_valence(exp2_arr[i], masked_normed_gaussian, upt_exponents=True)
            assert integration != 0
            assert not np.isnan(integration)
            new_exps2[i] /= (2 * integration)
        return new_exps2

    def get_new_coeffs_and_old_coeffs2(self, old_coeffs, old_coeffs2, old_exps, old_exp2, f=None):
        new_coeffs = f(old_coeffs, old_coeffs2, old_exps, old_exp2)
        return new_coeffs, old_coeffs, old_coeffs2

    def get_new_exps_and_old_exps2(self, old_coeffs, old_coeffs2, old_exps, old_exp2, f=None):
        new_exps = f(old_coeffs, old_coeffs2, old_exps, old_exp2)
        return new_exps, old_exps, old_exp2

    def run_valence(self, threshold_coeff, threshold_exp, coeff_arr, coeff2_arr, exp_arr, exp2_arr, iprint=False, iplot=False):
        old_coeffs = coeff_arr.copy() + threshold_coeff * 2.
        new_coeffs = coeff_arr.copy()
        old_coeffs2 = coeff2_arr.copy() + threshold_coeff * 2.
        new_coeffs2 = coeff2_arr.copy()
        old_exps = exp_arr.copy() + threshold_exp * 2.
        new_exps = exp_arr.copy()
        old_exps2 = exp2_arr.copy() + threshold_exp * 2.
        new_exps2 = exp2_arr.copy()
        storage_of_errors = [["""Integration Using Trapz"""],
                             [""" goodness of fit"""],
                             [""" goof of fit with r^2"""],
                             [""" KL Divergence Formula"""]]

        counter = 0
        while np.any(np.abs(new_exps - old_exps) > threshold_exp ) or np.any(np.abs(new_exps2 - old_exps2) > threshold_exps):
            new_coeffs, old_coeffs, old_coeffs2 = self.get_new_coeffs_and_old_coeffs2(new_coeffs, new_coeffs2,
                                                                                     new_exps, new_exps2,
                                                                                    f=self.update_coefficients)
            new_coeffs2, old_coeffs, old_coeffs2 = self.get_new_coeffs_and_old_coeffs2(old_coeffs, new_coeffs2,
                                                                                      new_exps, new_exps2,
                                                                          f=self.update_valence_coefficients)

            while np.any(np.abs(old_coeffs - new_coeffs) > threshold_coeff) or np.any(\
                          np.abs(old_coeffs2, new_coeffs2) > threshold_coeff):
                new_coeffs, old_coeffs, old_coeffs2 = self.get_new_coeffs_and_old_coeffs2(new_coeffs, new_coeffs2,
                                                                                         new_exps, new_exps2,
                                                                              f=self.update_coefficients)
                new_coeffs2, old_coeffs, old_coeffs2 = self.get_new_coeffs_and_old_coeffs2(old_coeffs, new_coeffs2,
                                                                                          new_exps, new_exps2,
                                                                              f=self.update_valence_coefficients)
                #print(new_coeffs2)
                model = self.get_normalized_gaussian_density(new_coeffs, new_coeffs2, new_exps, new_exps2)
                sum_of_coeffs = np.sum(new_coeffs) + np.sum(new_coeffs2)
                integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared, objective_function = \
                        self.get_descriptors_of_model(model)
                if iprint:
                    print(counter, integration_model_four_pi, sum_of_coeffs, \
                          goodness_of_fit, goodness_of_fit_r_squared, \
                          objective_function,  True, np.max(np.abs(old_coeffs - new_coeffs)),
                          np.max(np.abs(old_coeffs2 - new_coeffs2)))
                if iplot:
                    storage_of_errors[0].append(integration_model_four_pi)
                    storage_of_errors[1].append(goodness_of_fit)
                    storage_of_errors[2].append(goodness_of_fit_r_squared)
                    storage_of_errors[3].append(objective_function)
                counter += 1

            new_exps, old_exps, old_exps2 = self.get_new_exps_and_old_exps2(new_coeffs, new_coeffs2,
                                                                           new_exps, new_exps2,
                                                                    f=self.update_exponents)
            new_exps2, old_exps, old_exps2 = self.get_new_exps_and_old_exps2(new_coeffs, new_coeffs2,
                                                                            old_exps, new_exps2,
                                                                    f=self.update_valence_exponents)
            model = self.get_normalized_gaussian_density(new_coeffs, new_coeffs2, new_exps, new_exps2)
            sum_of_coeffs = np.sum(new_coeffs) + np.sum(new_coeffs2)
            integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared, objective_function = \
                    self.get_descriptors_of_model(model)
            if iprint:
                print(counter, integration_model_four_pi, sum_of_coeffs, \
                      goodness_of_fit, goodness_of_fit_r_squared, \
                      objective_function,  False, np.max(np.abs(old_exps - new_exps)),
                       np.max(np.abs(old_exps2 - new_exps2)))
            if iplot:
                storage_of_errors[0].append(integration_model_four_pi)
                storage_of_errors[1].append(goodness_of_fit)
                storage_of_errors[2].append(goodness_of_fit_r_squared)
                storage_of_errors[3].append(objective_function)
            counter += 1
        if iplot:
            self.create_plots(storage_of_errors[0], storage_of_errors[1], storage_of_errors[2], storage_of_errors[3])
        return new_coeffs, new_coeffs2, new_exps, new_exps2

    def run_greedy(self):
        pass