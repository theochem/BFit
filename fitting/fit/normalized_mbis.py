from __future__ import division

from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

import numpy as np

from abc import ABCMeta, abstractmethod

class MBIS_ABC():
    __metaclass__ = ABCMeta

    def __init__(self, electron_density, grid_obj, weights, atomic_number, element_name):
        assert isinstance(atomic_number, int), "atomic_number should be of type Integer instead it is %r" % type(atomic_number)
        self.electron_density = electron_density
        self.grid_obj = grid_obj
        self.grid_points = np.reshape(grid_obj.radii, (len(grid_obj.radii), 1))

        self.weights = np.ma.asarray(weights) #Weights are masked due the fact that they tend to be small
        self.atomic_number = atomic_number
        self.element_name = element_name
        self.is_valence_density = None

        # Various methods relay on masked values due to division of small numbers.
        self.masked_electron_density = np.ma.asarray(np.ravel(electron_density))
        self.masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.grid_points, 2.)))
        self.lagrange_multiplier = self.get_lagrange_multiplier()
        assert not np.isnan(self.lagrange_multiplier), "Lagrange multiplier is not a number, NAN."
        assert self.lagrange_multiplier != 0., "Lagange multiplier cannot be zero"


    @abstractmethod
    def get_normalized_gaussian_density(self):
        raise NotImplementedError()

    @abstractmethod
    def update_coefficients(self):
        raise NotImplementedError()

    def update_valence_coefficients(self):
        pass

    @abstractmethod
    def update_exponents(self):
        raise NotImplementedError()

    def update_valence_exponents(self):
        pass

    @abstractmethod
    def get_normalization_constant(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def get_lagrange_multiplier(self):
        return self.grid_obj.integrate(self.weights * self.masked_electron_density)\
                / (self.atomic_number)

    def get_all_normalization_constants(self, exp_arr):
        assert exp_arr.ndim == 1
        return np.array([self.get_normalization_constant(x) for x in exp_arr])

    def get_objective_function(self, model):
        log_ratio_of_models = np.log(self.masked_electron_density / model)
        return self.grid_obj.integrate(self.masked_electron_density * self.weights * log_ratio_of_models)

    def run_for_valence_density(self):
        pass

    def run_greedy(self):
        pass


    def integrate_model_with_four_pi(self, model):
        return self.grid_obj.integrate(model)

    def goodness_of_fit_grid_squared(self, model):
        return self.grid_obj.integrate(np.abs(model - np.ravel(self.electron_density))) / (4 * np.pi)

    def goodness_of_fit(self, model):
        return self.grid_obj.integrate(np.abs(model - np.ravel(self.electron_density)) / self.masked_grid_squared) / (4 * np.pi)

    def get_descriptors_of_model(self, model):
        return(self.integrate_model_with_four_pi(model),
               self.goodness_of_fit(model),
               self.goodness_of_fit_grid_squared(model),
               self.get_objective_function(model))

    def create_plots(self, integration_trapz, goodness_of_fit, goodness_of_fit_with_r_sq, objective_function):
        plt.plot(integration_trapz)
        plt.title(self.element_name + " - Integration of Model Using Trapz")
        plt.xlabel("Num of Iterations")
        plt.ylabel("Integration of Model Using Trapz")
        plt.savefig(self.element_name + "_Integration_Trapz.png")
        plt.close()

        plt.semilogy(goodness_of_fit)
        plt.xlabel("Num of Iterations")
        plt.title(self.element_name + " - Goodness of Fit")
        plt.ylabel("Int |Model - True| dr")
        plt.savefig(self.element_name + "_good_of_fit.png")
        plt.close()

        plt.semilogy(goodness_of_fit_with_r_sq)
        plt.xlabel("Num of Iterations")
        plt.title(self.element_name + " - Goodness of Fit with r^2")
        plt.ylabel("Int |Model - True| r^2 dr")
        plt.savefig(self.element_name + "_goodness_of_fit_r_squared.png")
        plt.close()

        plt.semilogy(objective_function)
        plt.xlabel("Num of Iterations")
        plt.title(self.element_name + " -  Objective Function")
        plt.ylabel("KL Divergence Formula")
        plt.savefig(self.element_name + "_objective_function.png")
        plt.close()

class TotalMBIS(MBIS_ABC):
    def __init__(self, electron_density, grid_obj, weights, atomic_number, element_name):
        super(TotalMBIS, self).__init__(electron_density, grid_obj, weights, atomic_number, element_name)

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

        counter = 0
        while np.any(np.abs(new_exps - old_exps) > threshold_exp ):
            new_coeffs, old_coeffs = self.get_new_coeffs_and_old_coeffs(new_coeffs, new_exps)

            while np.any(np.abs(old_coeffs - new_coeffs) > threshold_coeff):
                new_coeffs, old_coeffs = self.get_new_coeffs_and_old_coeffs(new_coeffs, new_exps)

                model = self.get_normalized_gaussian_density(new_coeffs , new_exps)
                sum_of_coeffs = np.sum(new_coeffs)
                integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared, objective_function = \
                        self.get_descriptors_of_model(model)
                if iprint:
                    print(counter, integration_model_four_pi, sum_of_coeffs, \
                          goodness_of_fit, goodness_of_fit_r_squared, \
                          objective_function,  True, np.max(np.abs(old_coeffs - new_coeffs)))
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
            if iprint:
                if counter % 100 == 0.:
                    for x in range(0, len(new_coeffs)):
                        print(new_coeffs[x], new_exps[x])
                print(counter, integration_model_four_pi, sum_of_coeffs, \
                      goodness_of_fit, goodness_of_fit_r_squared, \
                      objective_function, False, np.max(np.abs(new_exps - old_exps)))
            if iplot:
                storage_of_errors[0].append(integration_model_four_pi)
                storage_of_errors[1].append(goodness_of_fit)
                storage_of_errors[2].append(goodness_of_fit_r_squared)
                storage_of_errors[3].append(objective_function)
            counter += 1
        if iplot:
            self.create_plots(storage_of_errors[0], storage_of_errors[1], storage_of_errors[2], storage_of_errors[3])
        return new_coeffs, new_exps

    def run_greedy(self):
        pass
    
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
if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4
    import os
    print()
    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    print(current_directory + "data\examples\\")
    file_path = current_directory + "data\examples\\" + ELEMENT_NAME #+ ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *

    NUMBER_OF_CORE_POINTS = 400; NUMBER_OF_DIFFUSED_PTS = 500
    radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    radial_grid.radii = radial_grid.radii
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    #import horton
    #rtf = horton.ExpRTransform(1.0e-4, 25, 800)
    #radial_grid = horton.RadialGrid(rtf)
    #row_grid_points = radial_grid.radii
    #column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    be_val = GaussianValenceBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)
    fitting_obj_val = Fitting(be_val)
    be.electron_density /= (4 * np.pi)
    be_val.electron_density_valence /= (4 * np.pi)

    exps = be.UGBS_s_exponents
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    coeffs[coeffs == 0.] = 1e-6

    #exps_valence = be_val.UGBS_p_exponents
    #coeffs_valence = fitting_obj_val.optimize_using_nnls_valence(be_val.create_cofactor_matrix(exps, exps_valence))
    #coeffs_valence[coeffs_valence == 0.] = 1e-12
    #coeffs = coeffs_valence[0:len(be_val.UGBS_s_exponents)]
    #coeffs_valence = coeffs_valence[len(exps):]

    coeffs = np.array([4.])
    exps = np.array([100.])

    def f():
        pass

    print(np.trapz(y=4 * np.pi * np.ravel(be.electron_density) * np.ravel(np.power(be.grid, 2.)),
                   x=np.ravel(be.grid)))
    weights = np.ones(len(row_grid_points)) #  1 / (4 * np.pi * np.power(row_grid_points, .1))
    mbis_obj_val = ValenceMBIS(be.electron_density_valence, radial_grid, weights=weights, element_name=ELEMENT_NAME, atomic_number=2)
    mbis_obj = TotalMBIS(be.electron_density, radial_grid, weights=weights, atomic_number=ATOMIC_NUMBER, element_name=ELEMENT_NAME)

    coeffs, exps = mbis_obj.run(1e-4, 1e-2, coeffs, exps, iprint=True)
    #coeffs, coeffs2, exps, exps2 = mbis_obj_val.run_valence(1e-2, 1e-2, coeffs, coeffs_valence, exps, exps_valence, iprint=True)
    print("final", coeffs, exps)

    model = mbis_obj.get_normalized_gaussian_density(coeffs, exps)
    def plot_atomic_density(radial_grid, model, electron_density, title, figure_name):
        #Density List should be in the form
        # [(electron density, legend reference),(model1, legend reference), ..]
        import matplotlib.pyplot as plt
        colors = ["#FF00FF", "#FF0000", "#FFAA00", "#00AA00", "#00AAFF", "#0000FF", "#777777", "#00AA00", "#00AAFF"]
        ls_list = ['-', ':', ':', '-.', '-.', '--', '--', ':', ':']

        radial_grid *= 0.5291772082999999   #convert a.u. to angstrom
        plt.semilogy(radial_grid, model, lw=3, label="approx_model", color=colors[0], ls=ls_list[0])
        plt.semilogy(radial_grid, electron_density, lw=3, label="True Model", color=colors[2], ls=ls_list[3])
        #plt.xlim(0, 25.0*0.5291772082999999)
        plt.xlim(0, 9)
        plt.ylim(ymin=1e-9)
        plt.xlabel('Distance from the nucleus [A]')
        plt.ylabel('Log(density [Bohr**-3])')
        plt.title(title)
        plt.legend()
        plt.savefig(figure_name)
        plt.show()
        plt.close()
    #plot_atomic_density(row_grid_points, model, np.ravel(be.electron_density), "Hey I Just Met You", "This is Crazy")
    import numpy as np
    import matplotlib.pyplot as plt
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

    #plot_density_sections(np.ravel(be.electron_density), model, row_grid_points, title="hey i just met you")

    def try_one_gaussian_func(coeff):
        return -(np.trapz(y=4 * np.pi * mbis_obj.masked_grid_squared * mbis_obj.masked_electron_density *
                         np.log(mbis_obj.masked_electron_density / coeff),
                         x=np.ravel(mbis_obj.grid_points))) / \
               (np.trapz(y= 4 * np.pi * mbis_obj_val.masked_grid_quadrupled * mbis_obj.masked_electron_density,
                         x=np.ravel(mbis_obj.grid_points)))
    exp = (try_one_gaussian_func(4.))
    print(exp)
    print(mbis_obj.get_descriptors_of_model(mbis_obj.get_normalized_gaussian_density(coeffs, np.array([exp]))))
