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

from bfit.greedy.greedy_utils import (
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

    def _find_best_lparams(self, param_list):
        r"""
        Return the best initial guess from a list of potential model parameter choices.

        Parameters
        ----------
        param_list : List[TODO]

        Returns
        -------
        (float, List) :
            Returns the value of the cost-function and the best parameters.

        """
        best_local_value = 1e10
        best_local_param = None
        for param in param_list:
            local_param = self.get_optimization_routine(param, local=True)
            cost_func = self.get_cost_function(local_param)
            if cost_func < best_local_value:
                best_local_value = self.get_cost_function(local_param)
                best_local_param = local_param
        return best_local_value, best_local_param

    def _split_parameters(self, params):
        r"""
        Splits parameters into the s-type, p-type coefficients and exponents.
        """
        s_coeffs = params[:self.num_s]
        p_coeffs = params[self.num_s:self.num_s + self.num_p]
        s_exps = params[self.num_s + self.num_p:2 * self.num_s + self.num_p]
        p_exps = params[2 * self.num_s + self.num_p:]
        return s_coeffs, s_exps, p_coeffs ,p_exps

    def __call__(self, factor, max_numb_funcs=30, backward_elim_funcs=None,
                 add_extra_choices=None, ioutput=False):
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
        add_p_type = True
        while numb_funcs < max_numb_funcs - 1 and np.abs(best_gval - prev_gval) >= 1e-8 and \
                numb_redum < 5:
            # Get the next list of choices of parameters for S-type and P-type orbitals.
            s_coeffs, s_exps, p_coeffs, p_exps = self._split_parameters(gparams)
            print("S params", s_coeffs, s_exps)
            print("P params", p_coeffs, p_exps)
            if add_p_type:
                #  Add new P-type
                if self.num_p == 0:
                    choices_parameters = [np.array([2, 100])]
                else:
                    choices_parameters = self.next_param_func(factor, p_coeffs, p_exps)
                add_p_type = False
                choices_parameters = [
                    np.hstack((s_coeffs, x[:self.num_p], s_exps, x[self.num_p:])) for x in choices_parameters
                ]
                self.num_p += self.numb_func_increase
            else:
                # Add new S-type
                choices_parameters = self.next_param_func(factor, s_coeffs, s_exps)
                add_p_type = True
                choices_parameters = [
                    np.hstack((x[:self.num_s], p_coeffs, x[self.num_s:], p_exps)) for x in choices_parameters
                ]
                self.num_s += self.numb_func_increase

            if callable(add_extra_choices):
                # When you want extra choices to be added.
                # ie fitting the left-over density.
                choices_parameters.append(add_extra_choices(gparams))
            # Update model for the new number of S-type and P-type functions.
            self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            numb_funcs += self.numb_func_increase

            # Run fast, quick optimization and find the best parameter out of the choices.
            best_lval, best_lparam = self._find_best_lparams(choices_parameters)
            print(best_lparam)

            # Check if redundancies were found in the coefficients and exponents, remove them,
            #   change the factor and try again.
            s_coeffs, s_exps, p_coeffs, p_exps = self._split_parameters(best_lparam)
            s_coeffs_new, s_exps_new = check_redundancies(s_coeffs, s_exps)
            p_coeffs_new, p_exps_new = check_redundancies(p_coeffs, p_exps)
            if not add_p_type and self.num_s != len(s_coeffs_new):
                print("Found S-type Redudancies")
                self.redudan_info.append([numb_funcs, self.num_s, "Go Back a Iteration with new factor"])
                numb_funcs -= 1
                self.num_s -= 1
                numb_redum += 1  # Update the termination criteria
                factor += 5  # Increase the factor that changes the kinds of potential choices.
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            elif add_p_type and self.num_p != len(p_coeffs_new):
                print("Found P-type Redudancies")
                self.redudan_info.append([numb_funcs, self.num_p, "Go Back a Iteration with new factor"])
                numb_funcs -= 1
                self.num_p -= 1
                numb_redum += 1  # Update the termination criteria
                factor += 5  # Increase the factor that changes the kinds of potential choices.
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            # Take the best local choice and optimization even further.
            elif best_lval <= best_gval:
                prev_gval, best_gval = best_gval, best_lval
                gparams = self.get_optimization_routine(best_lparam, local=False)
                self.store_errors(gparams)
                print("errors", self.err_arr)
                params_iter.append(gparams)
                numb_redum = 0  # Reset the number of redundancies.
                factor = factor0  # Reset Factor.

            else:
                exit_info = "Next Iteration Did Not Find The Best Choice"
                break
            if backward_elim_funcs is not None:
                gparams = backward_elim_funcs(best_lparam)

        # "Next Iteration Did Not Find The Best Choice":
        if exit_info is None:
            exit_info = self._final_exit_info(numb_funcs, max_numb_funcs, best_gval, prev_gval, numb_redum)

        # Get the error message.

        results = {"x": gparams,
                   "fun": np.array(fun),
                   "parameters_iteration": params_iter,
                   "performance": np.array(performance),
                   "exit_information": exit_info}
        return gparams, params_iter, exit_info

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
    def __init__(self, grid, density, choice="pick-one", scale=2):
        self.gauss_obj = AtomicGaussianDensity(grid.points, num_s=density)
        self.grid = grid
        super(GreedyStrategy, self).__init__(choice, scale)

    @property
    def density(self):
        return self.gauss_obj._density

    def get_cost_function(self, params):
        self.gauss_obj.cost_function(params)

    def _solve_one_function_weight(self, weight):
        a = 2.0 * np.sum(weight)
        sum_of_grid_squared = np.sum(weight * np.power(self.grid.points, 2))
        b = 2.0 * sum_of_grid_squared
        sum_ln_electron_density = np.sum(weight * np.log(self.density))
        c = 2.0 * sum_ln_electron_density
        d = b
        e = 2.0 * np.sum(weight * np.power(self.grid.points, 4))
        f = 2.0 * np.sum(weight * np.power(self.grid.points, 2) *
                         np.log(self.density))
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
        cost_func1 = self.gauss_obj.cost_function(p1)

        p2 = self._solve_one_function_weight(self.density)
        cost_func2 = self.gauss_obj.cost_function(p2)

        p3 = self._solve_one_function_weight(weight3)
        cost_func3 = self.gauss_obj.cost_function(p3)

        p_min = min([(cost_func1, p1), (cost_func2, p2), (cost_func3, p3)],
                    key=lambda t: t[0])

        # Minimize by analytically finding coefficient.
        val = 1e10
        if self.gauss_obj.element is not None:
            exp_choice1 = self.gauss_obj.generation_of_UGBS_exponents(1.25,
                                                                      self.ugbs)
            exp_choice2 = self.gauss_obj.generation_of_UGBS_exponents(1.5,
                                                                      self.ugbs)
            exp_choice3 = self.gauss_obj.generation_of_UGBS_exponents(1.75,
                                                                      self.ugbs)
            grid_squared = self.grid.points**2.
            best_found = None
            for exp in np.append((exp_choice1, exp_choice2, exp_choice3)):
                num = np.sum(self.density * np.exp(-exp * grid_squared))
                den = np.sum(np.exp(-2. * exp * grid_squared))
                c = num / den
                p = np.array([c, exp])
                p = self.get_optimization_routine(p)
                cost_func = self.gauss_obj.cost_function(p)
                if cost_func < val:
                    val = cost_func
                    best_found = p

        if p_min[0] < val:
            return p_min[1]
        return best_found

    def get_optimization_routine(self, params):
        exps = params[len(params)//2:]
        cofac_matrix = self.gauss_obj.create_cofactor_matrix(exps)
        coeffs = optimize_using_nnls(self.density, cofac_matrix)

        p = np.append(coeffs, exps)
        params = optimize_using_slsqp(self.gauss_obj, p)
        return params

    def get_errors_from_model(self, params):
        model = self.gauss_obj.create_model(params)
        err1 = self.gauss_obj.integrate_model_trapz(model)
        err2 = self.gauss_obj.get_integration_error(self.density, model)
        err3 = self.gauss_obj.get_error_diffuse(self.density, model)
        return [err1, err2, err3]


class GreedyKL(GreedyStrategy):
    r"""

    """
    def __init__(self, grid, density, choice="pick-one", eps_coeff=1e-4, eps_exp=1e-5, scale=2.0,
                 mask_value=1e-12, integration_val=None, normalize=True):
        r"""

        Parameters
        ----------

        Returns
        -------

        """
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
        # Update Model if it doesn't match shape. This is due to poor design of the classes.
        if True:
            self.mbis_obj.model.change_numb_s_and_numb_p(len(coeffs), 0)

        if local:
            result = self.mbis_obj.run(coeffs, exps, opt_coeffs=True, opt_expons=True,
                                     maxiter=500, c_threshold=1e-2, e_threshold=1e-3,
                                     disp=False)['x']
            return np.hstack((result[0], result[1]))
        result = self.mbis_obj.run(coeffs, exps, opt_coeffs=True, opt_expons=True, maxiter=500,
                                 c_threshold=self.threshold_coeff, e_threshold=self.threshold_exp,
                                 disp=False)['x']
        return np.hstack((result[0], result[1]))

    def get_errors_from_model(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.mbis_obj.goodness_of_fit(coeffs, exps)


if __name__ == "__main__":
    from bfit.grid import ClenshawRadialGrid
    from bfit.density import AtomicDensity
    grid_obj = ClenshawRadialGrid(4, 1000, 1000)
    dens_obj = AtomicDensity("be")
    dens = dens_obj.atomic_density(grid_obj.points)

    greedy = GreedyKL(grid_obj, dens, integration_val=4.0)
    greedy(2.0)
