# -*- coding: utf-8 -*-
# FittingBasisSets is a basis-set curve-fitting optimization package.
#
# Copyright (C) 2018 The FittingBasisSets Development Team.
#
# This file is part of FittingBasisSets.
#
# FittingBasisSets is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# FittingBasisSets is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""
TODO

"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import nnls

from bfit.greedy_utils import (
    check_redundancies, get_next_choices, get_two_next_choices, pick_two_lose_one
)
from bfit.model import AtomicGaussianDensity
from bfit.fit import GaussianBasisFit, KLDivergenceSCF

__all__ = ["GreedyLeastSquares", "GreedyKL"]


class GreedyStrategy(metaclass=ABCMeta):
    r"""
    TODO

    """

    def __init__(self, grid, choice_function="pick-one", scale=2.0):
        r"""

        Parameters
        ----------
        choice_function : str
            Determines how the next set of basis-functions are chosen.
            Can be either:
                "pick-one" :  Add a new basis-funciton, by taking the average between every two exponents.
                "pick-two" : Add two new basis-functions.
                pick-two-lose-one" : Pick two basis-functions then remove one.

            Further information regarding these, please see .

        scale : float
            Positive number that scales the guesses

        """
        if not isinstance(scale, float):
            raise TypeError(f"Scale {scale} parameter should be a float.")
        if scale <= 0.0:
            raise ValueError(f"Scale {scale} should be positive.")

        if choice_function == "pick-one":
            self.next_param_func = get_next_choices
            self.numb_func_increase = 1  # How many basis functions were increased
        elif choice_function == "pick-two":
            self.next_param_func = get_two_next_choices
            self.numb_func_increase = 2
        elif choice_function == "pick-two-lose-one":
            self.next_param_func = pick_two_lose_one
            self.numb_func_increase = 1
        else:
            raise ValueError(f"Choice parameter {choice_function} was not recognized.")
        self.num_s = 1
        self.num_p = 0
        self.model = AtomicGaussianDensity(grid.points, num_s=self.num_s, num_p=self.num_p, normalize=True)
        self.scale = scale
        self.err_arr = []
        self.redudan_info = []  # Stores information about redundancies of parameters.

    @abstractmethod
    def get_cost_function(self):
        r"""Return evaluation of the function that is being optimized."""
        raise NotImplementedError()

    @abstractmethod
    def get_best_one_function_solution(self):
        r"""Return the solution of the model parameters for one basis function."""
        # Usually the model with single basis function can be solved analytically.
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_routine(self, local=False, *args):
        raise NotImplementedError()

    @abstractmethod
    def get_errors_from_model(self, params):
        pass

    def get_next_iter_params(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.next_param_func(self.scale, coeffs, exps)

    def store_errors(self, params):
        err = self.get_errors_from_model(params)
        if len(self.err_arr) == 0:
            self.err_arr = [[x] for x in err]
        else:
            for i, x in enumerate(err):
                self.err_arr[i].append(x)

    def _find_best_lparams(self, param_list, num_s_choices, num_p_choices):
        r"""
        Return the best initial guess from a list of potential model parameter choices.

        Parameters
        ----------
        param_list : List[TODO]

        num_s_choices : int

        num_p_choices : int

        Returns
        -------
        (float, List, bool) :
            Returns the value of the cost-function and the best parameters.
            True if adding S-type is the most optimal, False otherwise.

        """
        best_local_value = 1e10
        best_local_param = None
        is_s_optimal = False
        for i in range(0, num_s_choices + num_p_choices):
            param = param_list[i]

            # Update model for the new number of S-type and P-type functions.
            if i < num_s_choices:
                self.model.change_numb_s_and_numb_p(self.num_s + self.numb_func_increase, self.num_p)
            else:
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p + self.numb_func_increase)

            local_param = self.get_optimization_routine(param, local=True)
            cost_func = self.get_cost_function(local_param)
            if cost_func < best_local_value:
                best_local_value = self.get_cost_function(local_param)
                best_local_param = local_param
                if i < num_s_choices:
                    is_s_optimal = True
                else:
                    is_s_optimal = False
        return best_local_value, best_local_param, is_s_optimal

    def _split_parameters(self, params):
        r"""
        Splits parameters into the s-type, p-type coefficients and exponents.
        """
        s_coeffs = params[:self.num_s]
        p_coeffs = params[self.num_s:self.num_s + self.num_p]
        s_exps = params[self.num_s + self.num_p:2 * self.num_s + self.num_p]
        p_exps = params[2 * self.num_s + self.num_p:]
        return s_coeffs, s_exps, p_coeffs, p_exps

    def __call__(self, factor, d_threshold=1e-8, max_numb_funcs=30, add_extra_choices=None, ioutput=False):
        # Initialize all the variables
        gparams = self.get_best_one_function_solution()
        self.store_errors(gparams)
        exit_info = None
        numb_funcs = 1  # Number of current functions in model.
        prev_gval, best_gval = 0, 1e10
        params_iter = [gparams]  # Storing the parameters at each iteration.
        numb_redum = 0  # Number of redundancies, termination criteria.
        factor0 = factor  # Scaling parameter that changes.

        # Start the greedy algorithm
        success = True
        while numb_funcs < max_numb_funcs - 1  and numb_redum < 5 and np.abs(best_gval - prev_gval) >= d_threshold:
            s_coeffs, s_exps, p_coeffs, p_exps = self._split_parameters(gparams)
            print("S params", s_coeffs, s_exps)
            print("P params", p_coeffs, p_exps)

            # Get the next list of choices of parameters for S-type and P-type orbitals.
            # Add new S-type
            choices_parameters_s = self.next_param_func(factor, s_coeffs, s_exps)
            choices_parameters_s = [
                np.hstack((x[:self.num_s], p_coeffs, x[self.num_s:], p_exps)) for x in choices_parameters_s
            ]
            #  Add new P-type
            if self.num_p == 0:
                # Default choices for the first P-type guess. Coefficient of 2 with exponent 100.
                choices_parameters_p = [np.array([2, 100])]
            else:
                choices_parameters_p = self.next_param_func(factor, p_coeffs, p_exps)
            choices_parameters_p = [
                np.hstack((s_coeffs, x[:self.num_p], s_exps, x[self.num_p:])) for x in choices_parameters_p
            ]
            num_s_choices = len(choices_parameters_s)
            num_p_choices = len(choices_parameters_p)
            total_choices = choices_parameters_s + choices_parameters_p
            if callable(add_extra_choices):
                # When you want extra choices to be added.
                # ie fitting the left-over density.
                total_choices.append(add_extra_choices(gparams))

            # Run fast, quick optimization and find the best parameter out of the choices.
            best_lval, best_lparam, is_s_optimal = self._find_best_lparams(total_choices, num_s_choices, num_p_choices)
            print("Is S-Type optimal", is_s_optimal)

            # Update model for the new number of S-type and P-type functions.
            if is_s_optimal:
                self.num_s += self.numb_func_increase
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            else:
                self.num_p += self.numb_func_increase
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            numb_funcs += self.numb_func_increase

            # Check if redundancies were found in the coefficients and exponents, remove them,
            #   change the factor and try again.
            s_coeffs, s_exps, p_coeffs, p_exps = self._split_parameters(best_lparam)
            s_coeffs_new, s_exps_new = check_redundancies(s_coeffs, s_exps)
            p_coeffs_new, p_exps_new = check_redundancies(p_coeffs, p_exps)
            if is_s_optimal and self.num_s != len(s_coeffs_new):
                # Found S-type Redundancies, remove them and change factor for different choices.
                print("Found S-type Redudancies")
                self.redudan_info.append([numb_funcs, self.num_s, "Go Back a Iteration with new factor"])
                numb_funcs -= 1
                self.num_s -= 1
                numb_redum += 1  # Update the termination criteria
                factor += 5      # Increase the factor that changes the kinds of potential choices.
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            elif not is_s_optimal and self.num_p != len(p_coeffs_new):
                # Found P-type Redundancies, remove them and change factor for different choices.
                print("Found P-type Redudancies")
                self.redudan_info.append([numb_funcs, self.num_p, "Go Back a Iteration with new factor"])
                numb_funcs -= 1
                self.num_p -= 1
                numb_redum += 1  # Update the termination criteria
                factor += 5      # Increase the factor that changes the kinds of potential choices.
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            elif best_lval <= best_gval:
                # Take the best local choice and optimization even further.
                prev_gval, best_gval = best_gval, best_lval
                gparams = self.get_optimization_routine(best_lparam, local=False)
                self.store_errors(gparams)
                print("errors", [x[-1] for x in self.err_arr])
                params_iter.append(gparams)
                numb_redum = 0    # Reset the number of redundancies.
                factor = factor0  # Reset Original Factor.
            else:
                success = False
                exit_info = "Next Iteration Did Not Find The Best Choice"
                break
            print("")

        if numb_funcs == max_numb_funcs - 1 or numb_redum == 5:
            success = False

        # "Next Iteration Did Not Find The Best Choice":
        if exit_info is None:
            # TODO check if these are well-define, when the while statements don't go through, it will fail.
            exit_info = self._final_exit_info(numb_funcs, max_numb_funcs, best_gval, prev_gval, numb_redum)

        results = {"x": gparams,
                   "fun": np.array(self.err_arr).T[:, -1],
                   "success" : success,
                   "parameters_iteration": params_iter,
                   "performance": np.array(self.err_arr).T,
                   "exit_information": exit_info}
        return results

    def _final_exit_info(self, num_func, max_func, best_val, prev_gval, redum):
        if not num_func < max_func - 1:
            exit_info = "Max number of functions reached,  " + str(num_func)
        elif not np.abs(best_val - prev_gval) >= 1e-5:
            exit_info = "Cost function is less than some epsilon " + \
                             str(best_val - prev_gval) + " <= " + str(1e-5)
        elif not redum < 5:
            exit_info = " Number of redudancies " + str(redum) + " found in a row is more than 5, "
        return exit_info


class GreedyLeastSquares(GreedyStrategy):
    def __init__(self, grid, density, choice="pick-one", local_tol=1e-5, global_tol=1e-8, scale=2.0, method="SLSQP"):
        self.gauss_obj = AtomicGaussianDensity(grid.points, num_s=1)
        self.grid = grid
        self.local_tol = local_tol
        self.global_tol = global_tol
        self.successful = None
        super(GreedyLeastSquares, self).__init__(grid, choice, scale)
        self.gaussian_obj = GaussianBasisFit(grid, density, self.model, measure="LS", method=method)

    @property
    def density(self):
        return self.gaussian_obj.density

    def get_model(self, params):
        return self.model.evaluate(params[:len(params)//2], params[len(params)//2:])

    def get_cost_function(self, params):
        model = self.get_model(params)
        return self.grid.integrate(self.gaussian_obj.measure.evaluate(model))

    def optimize_using_nnls(self, true_dens, cofactor_matrix):
        r"""
        """
        b_vector = np.copy(true_dens)
        b_vector = np.ravel(b_vector)
        row_nnls_coefficients = nnls(cofactor_matrix, b_vector)
        return row_nnls_coefficients[0]

    def _solve_one_function_weight(self, weight):
        a = 2.0 * np.sum(weight)
        sum_of_grid_squared = np.sum(weight * np.power(self.grid.points, 2))
        b = 2.0 * sum_of_grid_squared
        sum_ln_electron_density = np.sum(weight * np.log(self.density))
        c = 2.0 * sum_ln_electron_density
        d = b
        e = 2.0 * np.sum(weight * np.power(self.grid.points, 4))
        f = 2.0 * np.sum(weight * np.power(self.grid.points, 2) * np.log(self.density))
        big_a = (b * f - c * e) / (b * d - a * e)
        big_b = (a * f - c * d) / (a * e - b * d)
        coefficient = np.exp(big_a)
        exponent = - big_b
        return np.array([coefficient, exponent])

    def get_best_one_function_solution(self):
        # Minimizing weighted least squares with three different weights
        weight1 = np.ones(len(self.grid.points))
        weight3 = np.power(self.density, 2.)
        p1 = self._solve_one_function_weight(weight1)
        cost_func1 = self.get_cost_function(p1)

        p2 = self._solve_one_function_weight(self.density)
        cost_func2 = self.get_cost_function(p2)

        p3 = self._solve_one_function_weight(weight3)
        cost_func3 = self.get_cost_function(p3)

        p_min = min(
            [(cost_func1, p1), (cost_func2, p2), (cost_func3, p3)], key=lambda t: t[0]
        )
        return p_min[1]

    def create_cofactor_matrix(self, exponents):
        exponents_s = exponents[:self.model.num_s]
        grid_squared_col = np.power(self.grid.points, 2.0).reshape((len(self.grid.points), 1))
        exponential = np.exp(-exponents_s * grid_squared_col)

        if self.model.num_p != 0:
            exponents_p = exponents[self.model.num_s:]
            exponential = np.hstack(
                (exponential, grid_squared_col * np.exp(-exponents_p * grid_squared_col))
            )
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid.points)))
        assert np.ndim(exponential) == 2
        return exponential

    def get_optimization_routine(self, params, local=False):
        exps = params[len(params)//2:]
        cofac_matrix = self.create_cofactor_matrix(exps)
        coeffs = self.optimize_using_nnls(self.density, cofac_matrix)
        if local:
            results = self.gaussian_obj.run(coeffs, exps, tol=self.local_tol, maxiter=1000)["x"]
        else:
            results = self.gaussian_obj.run(coeffs, exps, tol=self.global_tol, maxiter=1000, disp=True,
                                            with_constraint=True)["x"]
        return np.hstack((results[0], results[1]))

    def get_errors_from_model(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.gaussian_obj.goodness_of_fit(coeffs, exps)


class GreedyKL(GreedyStrategy):
    def __init__(self, grid, density, choice="pick-one", eps_coeff=1e-4, eps_exp=1e-5, scale=2.0,
                 mask_value=1e-12, integration_val=None, normalize=True):
        self.threshold_coeff = eps_coeff
        self.threshold_exp = eps_exp
        self.successful = None
        self.normalize = normalize
        super(GreedyKL, self).__init__(grid, choice, scale)
        self.mbis_obj = KLDivergenceSCF(grid, density, self.model, mask_value, integration_val)

    @property
    def density(self):
        return self.mbis_obj.density

    @property
    def norm(self):
        return self.mbis_obj.norm

    @property
    def grid(self):
        return self.mbis_obj.grid

    def get_model(self, params):
        return self.model.evaluate(params[:len(params)//2], params[len(params)//2:])

    def get_cost_function(self, params):
        model = self.get_model(params)
        return self.grid.integrate(self.mbis_obj.measure.evaluate(model))

    def get_best_one_function_solution(self):
        denom = self.grid.integrate(self.density * np.power(self.grid.points ** 2, 2.))
        exps = 3. * self.norm / (2. * 4. * np.pi * denom)
        return np.array([self.norm, exps])

    def get_optimization_routine(self, params, local=False):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        if local:
            result = self.mbis_obj.run(coeffs, exps, opt_coeffs=True, opt_expons=True,
                                     maxiter=500, c_threshold=1e-2, e_threshold=1e-3,
                                     disp=False)['x']
            return np.hstack((result[0], result[1]))
        result = self.mbis_obj.run(coeffs, exps, opt_coeffs=True, opt_expons=True, maxiter=1000,
                                 c_threshold=self.threshold_coeff, e_threshold=self.threshold_exp,
                                   d_threshold=1e-6, disp=False)['x']
        return np.hstack((result[0], result[1]))

    def get_errors_from_model(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.mbis_obj.goodness_of_fit(coeffs, exps)


if __name__ == "__main__":
    from bfit.grid import ClenshawRadialGrid
    from bfit.density import AtomicDensity
    grid = ClenshawRadialGrid(4, num_core_pts=10000, num_diffuse_pts=900, extra_pts=[50, 75, 100])
    dens_obj = AtomicDensity("be")
    dens = dens_obj.atomic_density(grid.points)

    greedy = GreedyKL(grid, dens, integration_val=4.0, eps_coeff=1e-7, eps_exp=1e-8)
    # greedy = GreedyLeastSquares(grid, dens, choice="pick-one", method="SLSQP")
    greedy(2.0)
