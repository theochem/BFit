from __future__ import division
from fitting.fit.GaussianBasisSet import *
import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod


class MBIS_ABC():
    __metaclass__ = ABCMeta

    def __init__(self, element_name, atomic_number, grid_obj, electron_density, weights=None):
        assert isinstance(atomic_number, int), \
            "atomic_number should be of type Integer instead it is %r" % type(atomic_number)
        assert np.abs(grid_obj.integrate(electron_density) - atomic_number) < 1e-1, \
            "Atomic number doesn't match the integration value. Instead it is %r" \
            % grid_obj.integrate(electron_density)
        self.grid_obj = grid_obj
        self.grid_points = np.ma.asarray(np.reshape(grid_obj.radii, (len(grid_obj.radii), 1)))

        if weights is None:
            self.weights = np.ma.asarray(np.ones(len(self.grid_obj.radii)))
        else:
            # Weights are masked due the fact that they tend to be small
            self.weights = np.ma.asarray(weights)

        self.atomic_number = float(atomic_number)
        self.element_name = element_name
        if element_name == "h" and atomic_number == 1:
            self.electron_density = self.get_hydrogen_electron_density()
        else:
            self.electron_density = electron_density
        # Various methods relay on masked values due to division of small numbers.
        self.masked_electron_density = np.ma.asarray(np.ravel(self.electron_density))
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

    def run_for_valence_density(self):
        pass

    def run_greedy(self):
        pass

    def get_lagrange_multiplier(self):
        return self.grid_obj.integrate(self.weights * self.masked_electron_density) / self.atomic_number

    def get_all_normalization_constants(self, exp_arr):
        assert exp_arr.ndim == 1
        return np.array([self.get_normalization_constant(x) for x in exp_arr])

    def get_objective_function(self, model):
        log_ratio_of_models = np.log(self.masked_electron_density / np.ma.asarray(model))
        return self.grid_obj.integrate(self.masked_electron_density * self.weights * log_ratio_of_models
                                       / self.atomic_number)

    def integrate_model_with_four_pi(self, model):
        return self.grid_obj.integrate(model)

    def goodness_of_fit_grid_squared(self, model):
        return self.grid_obj.integrate(np.abs(model - np.ravel(self.electron_density))) / (4 * np.pi)

    def goodness_of_fit(self, model):
        return self.grid_obj.integrate(np.abs(model - np.ravel(self.electron_density)) / self.masked_grid_squared)\
               / (4 * np.pi)

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
