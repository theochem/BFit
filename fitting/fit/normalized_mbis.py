from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz

import numpy as np

from abc import ABCMeta, abstractmethod

class MBIS_ABC(metaclass=ABCMeta):
    def __init__(self, electron_density, grid_points, weights, atomic_number, element_name):
        assert isinstance(atomic_number, int), "atomic_number should be of type Integer instead it is %r" % type(atomic_number)
        self.electron_density = electron_density
        self.grid_points = grid_points
        self.weights = np.ma.asarray(weights) #Weights are masked due the fact that they tend to be small
        self.atomic_number = atomic_number
        self.lagrange_multiplier = self.get_lagrange_multiplier()
        self.element_name = element_name
        self.is_valence_density = None
        assert not np.isnan(self.lagrange_multiplier), "Lagrange multiplier is not a number, NAN."
        assert self.lagrange_multiplier != 0., "Lagange multiplier cannot be zero"

        # Various methods relay on masked values due to division of small numbers.
        self.masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density())
        self.masked_electron_density = np.ma.asarray(np.ravel(self.electron_density))
        self.masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.grid_points, 2.)))

    @abstractmethod
    def get_normalized_gaussian_density(self):
        raise NotImplementedError()

    @abstractmethod
    def update_coefficients(self):
        raise NotImplementedError()

    @abstractmethod
    def update_valence_coefficients(self):
        pass

    @abstractmethod
    def update_exponents(self):
        raise NotImplementedError()

    @abstractmethod
    def update_valence_exponents(self):
        pass

    @abstractmethod
    @staticmethod
    def get_normalization_constant():
        raise NotImplementedError()

    def get_lagrange_multiplier(self):
        return 4 * np.pi * np.trapz(y=self.weights * self.masked_electron_density * self.masked_grid_squared \
                                    , x=np.ravel(self.grid_points)) / self.atomic_number

    def get_all_normalization_constants(self, exp_arr):
        assert exp_arr.ndim == 1
        return np.array([MBIS.get_normalization_constant(x) for x in exp_arr])

    def get_objective_function(self, model):
        log_ratio_of_models = np.log(self.masked_electron_density / model)
        four_pi_grid_squared = 4 * np.pi * self.masked_grid_squared
        return np.trapz(y=self.masked_electron_density * self.weights * four_pi_grid_squared * log_ratio_of_models,\
                        x=np.ravel(self.model_object.grid))

    def run(self, threshold_coeff, threshold_exp, *args, iprint=False, iplot=False):
        if self.is_valence_density:
            self.run_for_valence_density(threshold_coeff, threshold_exp, *args, iprint=iprint, iplot=iplot)
        else:
            assert len(args) == 2, "Length of args should be 2 or 4, instead it is %r" % len(args)
            coeff_arr = args[0]
            exp_arr = args[1]

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
                new_coeffs, old_coeffs = self.get_updated_coeffs_and_old_coeffs(new_coeffs, new_exps)

                while np.any(np.abs(old_coeffs - new_coeffs) > threshold_coeff):
                    temp_storage_coeffs = new_coeffs.copy()
                    new_coeffs = self.update_coefficients(new_coeffs, new_exps)
                    old_coeffs = temp_storage_coeffs.copy()

                    model = self.get_normalized_gaussian_density(new_coeffs , new_exps)

                    sum_of_coeffs = np.sum(new_coeffs)
                    integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared,\
                        objective_function = self.get_descriptors_of_model(model)

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

                temp_storage_exps = new_exps.copy()
                new_exps = self.update_exponents(new_coeffs, new_exps)
                old_exps = temp_storage_exps.copy()

                model = self.get_normalized_gaussian_density(new_coeffs, new_exps)
                sum_of_coeffs = np.sum(new_coeffs)
                integration_model_four_pi, goodness_of_fit, goodness_of_fit_r_squared,\
                        objective_function = self.get_descriptors_of_model(model)
                if iprint:
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
                self.create_plots(storage_of_errors[0], storage_of_errors[1],
                                  storage_of_errors[2], storage_of_errors[3])

    def run_for_valence_density(self):
        pass

    def run_greedy(self):
        pass

    def get_updated_exps_and_old_exps(self, coeff_arr, exp_arr):
        new_exp = self.update_exponents()

    def get_updated_coeffs_and_old_coeffs(self, coeff_arr, exp_arr):
        new_coeffs = self.update_coefficients(coeff_arr, exp_arr)
        return new_coeffs, coeff_arr

    def integrate_model_with_four_pi(self, model):
        return np.trapz(y= 4 * np.pi * self.masked_grid_squared * model , x=np.ravel(self.grid_points))

    def goodness_of_fit_grid_squared(self, model):
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_grid_squared *np.abs(model - np.ravel(self.electron_density)), x=np.ravel(self.grid_points))

    def goodness_of_fit(self, model):
        return np.trapz(y=np.abs(model - np.ravel(self.electron_density)), x=np.ravel(self.grid_points))

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

class MBIS():
    def __init__(self, model_object, weights, atomic_number):
        assert isinstance(model_object, DensityModel)
        assert type(atomic_number) is int, "Atomic Number should be of type int. Instead it is %r" % type(atomic_number)
        self.model_object = model_object
        self.weights = np.ma.asarray(weights)
        self.atomic_number = atomic_number #self.model_object.integrated_total_electron_density
        self.lamda_multplier = self.lagrange_multiplier()
        assert not np.isnan(self.lamda_multplier), "lagrange multiplier should not nan"
        assert self.lamda_multplier != 0., "lagrange multiplier cannot be zero"

    @staticmethod
    def get_normalization_constant(exponent):
        return (exponent / np.pi)**(3./2.)

    def get_all_normalization_constants(self, exp_arr):
        assert exp_arr.ndim == 1
        return np.array([MBIS.get_normalization_constant(x) for x in exp_arr])

    def get_normalized_coefficients(self, coeff_arr, exp_arr):
        normalized_constants = self.get_all_normalization_constants(exp_arr)
        assert len(normalized_constants) == len(coeff_arr)
        return coeff_arr * normalized_constants

    def get_normalized_gaussian_density(self, coeff_arr, exp_arr):
        exponential = np.exp(-exp_arr * np.power(self.model_object.grid, 2.))
        normalized_coeffs = self.get_normalized_coefficients(coeff_arr, exp_arr)
        assert normalized_coeffs.ndim == 1.
        return np.dot(exponential, normalized_coeffs)

    def lagrange_multiplier(self):
        grid_squared = np.ravel(np.power(self.model_object.grid, 2.))
        return 4 * np.pi * np.trapz(y=self.weights * np.ravel(self.model_object.electron_density) * grid_squared \
                                    , x=np.ravel(self.model_object.grid)) / self.atomic_number

    def get_integration_factor(self, exponent, masked_normed_gaussian):
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        ratio = masked_electron_density / masked_normed_gaussian
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))

        prefactor = 4 * np.pi * MBIS.get_normalization_constant(exponent)
        integrand = self.weights * ratio * np.exp(-exponent * masked_grid_squared)
        return prefactor * np.trapz(y=integrand * masked_grid_squared, x=np.ravel(self.model_object.grid))

    def update_coefficients(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))

        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_integration_factor(exp_arr[i], masked_normed_gaussian)
            new_coeff[i] /= self.lamda_multplier
        return new_coeff

    def update_exponents(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        masked_grid_quadrupled = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 4.)))
        ratio = masked_electron_density / masked_normed_gaussian

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            prefactor = 4 * np.pi * MBIS.get_normalization_constant(exp_arr[i])
            integrand = self.weights * ratio * np.exp(-exp_arr[i] * masked_grid_squared)
            new_exps[i] = 3 * self.lamda_multplier
            integration = np.trapz(y=integrand * masked_grid_quadrupled, x=np.ravel(self.model_object.grid))
            assert prefactor != 0, "prefactor should not be zero but the exponent for normalization constant is %r" % str(exp_arr[i])
            assert integration != 0, "Integration of the integrand is zero. The Integrand Times r^4 is %r" % str(integrand * masked_grid_quadrupled)
            assert not np.isnan(integration), "Integration should not be nan"
            new_exps[i] /= ( 2 * prefactor * integration)
        return new_exps

    def run(self, threshold_coeff, threshold_exp, coeff_arr, exp_arr, iprint=False, iplot=False):
        old_coeffs = coeff_arr.copy() + threshold_coeff * 2.
        new_coeffs = coeff_arr.copy()
        old_exps = exp_arr.copy() + threshold_exp * 2.
        new_exps = exp_arr.copy()
        storage_of_errors = [["""Integration Using Trapz"""],
                             ["""Sum of Coefficients"""],
                             [""" goodness of fit"""],
                             [""" goof of fit with r^2"""],
                             [""" KL Divergence Formula"""]]

        counter = 0
        while np.any(np.abs(new_exps - old_exps) > threshold_exp ):
            temp_storage_coeffs = new_coeffs.copy()
            new_coeffs = self.update_coefficients(new_coeffs, new_exps)
            old_coeffs = temp_storage_coeffs.copy()

            while np.any(np.abs(old_coeffs - new_coeffs) > threshold_coeff):
                temp_storage_coeffs = new_coeffs.copy()
                new_coeffs = self.update_coefficients(new_coeffs, new_exps)
                # print(new_coeffs)
                old_coeffs = temp_storage_coeffs.copy()
                model = self.get_normalized_gaussian_density(new_coeffs , new_exps)

                integration_model_four_pi = self.integrate_model_with_four_pi(model)
                sum_of_coeffs = np.sum(new_coeffs)
                goodness_of_fit = self.goodness_of_fit(model)
                goodness_of_fit_r_squared = self.goodness_of_fit_grid_squared(model)
                objective_function = self.get_objective_function(new_coeffs, new_exps)
                if iprint:
                    print(counter, integration_model_four_pi, sum_of_coeffs, \
                          goodness_of_fit, goodness_of_fit_r_squared, \
                          objective_function,  True, np.max(np.abs(old_coeffs - new_coeffs)))
                if iplot:
                    storage_of_errors[0].append(integration_model_four_pi)
                    storage_of_errors[1].append(sum_of_coeffs)
                    storage_of_errors[2].append(goodness_of_fit)
                    storage_of_errors[3].append(goodness_of_fit_r_squared)
                    storage_of_errors[4].append(objective_function)
                counter += 1

            temp_storage_exps = new_exps.copy()
            new_exps = self.update_exponents(new_coeffs, new_exps)
            #print(new_exps)
            old_exps = temp_storage_exps.copy()
            model = self.get_normalized_gaussian_density(new_coeffs, new_exps)

            integration_model_four_pi = self.integrate_model_with_four_pi(model)
            sum_of_coeffs = np.sum(new_coeffs)
            goodness_of_fit = self.goodness_of_fit(model)
            goodness_of_fit_r_squared = self.goodness_of_fit_grid_squared(model)
            objective_function = self.get_objective_function(new_coeffs, new_exps)
            if iprint:
                if counter % 100 == 0.:
                    for x in range(0, len(new_coeffs)):
                        print(new_coeffs[x], new_exps[x])
                print(counter, integration_model_four_pi, sum_of_coeffs, \
                      goodness_of_fit, goodness_of_fit_r_squared, \
                      objective_function, False, np.max(np.abs(new_exps - old_exps)))
            if iplot:
                storage_of_errors[0].append(integration_model_four_pi)
                storage_of_errors[1].append(sum_of_coeffs)
                storage_of_errors[2].append(goodness_of_fit)
                storage_of_errors[3].append(goodness_of_fit_r_squared)
                storage_of_errors[4].append(objective_function)
            counter += 1

        #########
        # PLotting
        #####
        if iplot:
            storage_of_errors = np.array(storage_of_errors)
            plt.plot(storage_of_errors[0])
            plt.title(self.model_object.element + " - Integration of Model Using Trapz")
            plt.xlabel("Num of Iterations")
            plt.ylabel("Integration of Model Using Trapz")
            plt.savefig(self.model_object.element + "_Integration_Trapz.png")
            plt.close()

            plt.plot(storage_of_errors[1])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " - Sum of Coefficients")
            plt.ylabel("Sum of Coefficients")
            plt.savefig(self.model_object.element + "_Sum_of_coefficients.png")
            plt.close()

            plt.semilogy(storage_of_errors[2])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " - Goodness of Fit")
            plt.ylabel("Int |Model - True| dr")
            plt.savefig(self.model_object.element + "_good_of_fit.png")
            plt.close()

            plt.semilogy(storage_of_errors[3])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " - Goodness of Fit with r^2")
            plt.ylabel("Int |Model - True| r^2 dr")
            plt.savefig(self.model_object.element + "_goodness_of_fit_r_squared.png")
            plt.close()

            plt.semilogy(storage_of_errors[4])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " -  Objective Function")
            plt.ylabel("KL Divergence Formula")
            plt.savefig(self.model_object.element + "_objective_function.png")
            plt.close()
        return new_coeffs, new_exps

    def run_greedy(self, threshold, iprint=False):
        def next_list_of_exponents(exponents_array, coefficient_array, factor, coeff_guess):
            assert exponents_array.ndim == 1
            size = exponents_array.shape[0]
            all_choices_of_exponents = []
            all_choices_of_coeffs = []

            for index, exp in np.ndenumerate(exponents_array):
                if index[0] == 0:
                    exponent_array = np.insert(exponents_array, index, exp / factor )
                    coeff_array = np.insert(coefficient_array, index, coeff_guess)
                elif index[0] <= size:
                    exponent_array = np.insert(exponents_array, index, (exponents_array[index[0] - 1] + exponents_array[index[0]])/2)
                    coeff_array = np.insert(coeff_arr, index, coeff_guess)

                all_choices_of_exponents.append(exponent_array)
                all_choices_of_coeffs.append(coeff_array)

                if index[0] == size - 1:
                    exponent_array = np.append(exponents_array, np.array([ exp * factor] ))
                    coeff_array = np.append(coeff_arr, np.array([coeff_guess]))
                    all_choices_of_exponents.append(exponent_array)
                    all_choices_of_coeffs.append(coeff_array)
            return(all_choices_of_exponents, all_choices_of_coeffs)

        coeff_arr = np.array([4.])
        exp_arr = np.array([333.])
        coeff_arr, exp_arr = self.run(1e-1, 20, coeff_arr, exp_arr)
        model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)

        for x in range(0, 50):
            next_list_of_exps, next_list_of_coeffs = next_list_of_exponents(exp_arr, coeff_arr, 100., coeff_guess=1.)

            best_next_exps = None
            best_next_coeffs = None
            best_objective_func = 1e10
            for i in range(0, len(next_list_of_exps)):
                objective_func = self.get_objective_function(next_list_of_coeffs[i], next_list_of_exps[i])

                #objective_func, coeffs, exps = self.optimize_using_slsqp(np.append(next_list_of_coeffs[i], next_list_of_exps[i]))
                if objective_func < best_objective_func:
                    best_next_exps = next_list_of_exps[i]
                    best_next_coeffs = next_list_of_coeffs[i]

            best_next_coeffs[best_next_coeffs == 0.] = 1e-12
            coeff_arr, exp_arr = self.run(1e-1, 1e-1, best_next_coeffs, best_next_exps,iprint=False)
            model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
            if iprint:
                print("\n", x, self.integrate_model(model), self.integrate_model_with_four_pi(model), np.sum(coeff_arr),\
                      self.goodness_of_fit(model), self.goodness_of_fit_grid_squared(model), \
                      self.get_objective_function(coeff_arr, exp_arr))


        return coeff_arr, exp_arr


    def fixed_iterations(self, coeff_arr, exp_arr):
        for y in range(0, 5):
            for x in range(0, 1000):
                coeff_arr = self.update_coefficients(coeff_arr, exp_arr)
                #print(coeff_arr)
                model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
                print(x, self.integrate_model(model), self.integrate_model_with_four_pi(model), np.sum(coeff_arr),\
                      self.goodness_of_fit(model), self.goodness_of_fit_grid_squared(model), \
                      self.get_objective_function(coeff_arr, exp_arr))
            for x in range(0, 1000):
                exp_arr = self.update_exponents(coeff_arr, exp_arr)
                print(exp_arr)
                model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
                print(x, self.integrate_model(model), self.integrate_model_with_four_pi(model), np.sum(coeff_arr),\
                      self.goodness_of_fit(model), self.goodness_of_fit_grid_squared(model), \
                      self.get_objective_function(coeff_arr, exp_arr))

    def get_objective_function(self, coeff_arr, exp_arr):
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_electron_density * self.weights * 4 * np.pi * masked_grid_squared * np.log(masked_electron_density / masked_normed_gaussian),\
                        x=np.ravel(self.model_object.grid))

    def integrate_model(self, model):
        return np.trapz(y=model * np.ravel(np.power(self.model_object.grid, 2.)), x=np.ravel(self.model_object.grid))

    def integrate_model_with_four_pi(self, model):
        return np.trapz(y= 4 * np.pi * model * np.ravel(np.power(self.model_object.grid, 2.)), x=np.ravel(self.model_object.grid))

    def goodness_of_fit_grid_squared(self, model):
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_grid_squared *np.abs(model - np.ravel(self.model_object.electron_density)), x=np.ravel(self.model_object.grid))

    def goodness_of_fit(self, model):
        return np.trapz(y=np.abs(model - np.ravel(self.model_object.electron_density)), x=np.ravel(self.model_object.grid))

    def maximum_difference(self):
        return np.max()

    def get_KL_divergence(self, parameters):
        coeff_arr = parameters[:len(parameters)//2]
        exp_arr = parameters[len(parameters)//2:]
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_electron_density * self.weights * 4 * np.pi * masked_grid_squared * np.log(masked_electron_density / masked_normed_gaussian),\
                        x=np.ravel(self.model_object.grid))

    def optimize_using_slsqp(self, parameters):
        def constraint(x):
            num_of_funcs = len(x)//2
            return self.atomic_number - np.sum(x[:num_of_funcs])
        def constraint2(x):
            num_of_funcs = len(x)//2
            model = self.get_normalized_gaussian_density(x[:num_of_funcs], x[num_of_funcs:])
            return np.sum(np.abs(np.ravel(self.model_object.electron_density) - model))

        #cons=({'type':'eq', 'fun':constraint},
        #      {'type':'eq', 'fun':constraint2})
        cons = ({'type':'eq', 'fun':constraint})
        bounds = np.array([(0.0, np.inf) for x in range(0, len(parameters))], dtype=np.float64)
        f_min_slsqp = scipy.optimize.minimize(self.get_KL_divergence, x0=parameters, method="SLSQP",
                                              bounds=bounds, constraints=cons, jac=False)
        parameters = f_min_slsqp['x']
        coeffs = parameters[:len(parameters)//2]
        exps = parameters[len(parameters)//2:]
        objective_func = f_min_slsqp['fun']
        return objective_func, coeffs, exps

if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4
    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME #+ ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 400; NUMBER_OF_DIFFUSED_PTS = 500
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])[1:]
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)

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
    exps = be.UGBS_s_exponents
    exps = np.array([np.random.random()*1000. for x in exps])
    coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    coeffs[coeffs == 0.] = 1e-12
    #coeffs = np.array([4.])
    #exps = np.array([1000000.])
    def f():
        pass

    be.electron_density /= (4 * np.pi)
    print(np.trapz(y=4 * np.pi * np.ravel(be.electron_density) * np.ravel(np.power(be.grid, 2.)),
                   x=np.ravel(be.grid)))
    weights =  np.ones(len(row_grid_points)) #   1 / (4 * np.pi * np.power(row_grid_points, 2.))
    mbis_obj = MBIS(be, weights, ATOMIC_NUMBER)
    #print("final", mbis_obj.fixed_iterations(coeffs, exps))
    #print("final", mbis_obj.run_greedy(1e-4, iprint=True))

    coeffs, exps = (mbis_obj.run(1e-3, 1e-1, coeffs, exps, iprint=True))
    print("final", coeffs, exps)
    #103615 0.318309348608 3.99999324462 4.0 0.00670581613856 0.00280627515515 0.00036309846081 False 9.99977783067e-06

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
    plot_atomic_density(row_grid_points, model, np.ravel(be.electron_density), "Hey I Just Met You", "This is Crazy")
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

    plot_density_sections(np.ravel(be.electron_density), model, row_grid_points, title="hey i just met you")

