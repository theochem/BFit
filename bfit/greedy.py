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
r"""Greedy Fitting Module"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.optimize import nnls

from bfit.model import AtomicGaussianDensity
from bfit.fit import ScipyFit, KLDivergenceSCF

__all__ = ["GreedyLeastSquares", "GreedyKL"]


def check_redundancies(coeffs, fparams, eps=1e-3):
    r"""
    Check if the fparams have similar values and groups them together.

    If any two function parameters have similar values, then one of the
    function parameters is removed, and the corresponding coefficient,
    are added together.
    Note: as of now this only works if each basis function depends on only one
    parameters e.g. e^(-x), not e^(-x + y).

    Parameters
    ----------
    coeffs : np.ndarray(M,)
        Coefficients of the basis function set of size :math:`M`.
    fparams : np.ndarray(M,)
        Function parameters of the basis function set of size :math:`M`.
    eps : float
        Value that indicates the threshold for how close two parameters are.

    Returns
    -------
    (np.ndarray, np.ndarray)
                            New coefficients and new exponents, where close
                            values of the function parameters are removed and
                            the coefficients are added together.

    """
    new_coeffs = coeffs.copy()
    new_exps = fparams.copy()

    # Indexes where they're the same.
    indexes_same = []
    for i, alpha in enumerate(fparams):
        similar_indexes = []
        for j in range(i + 1, len(fparams)):
            if j not in similar_indexes:
                if np.abs(alpha - fparams[j]) < eps:
                    if i not in similar_indexes:
                        similar_indexes.append(i)
                    similar_indexes.append(j)
        if len(similar_indexes) != 0:
            indexes_same.append(similar_indexes)

    # Add the coefficients together and add/group them, if need be
    for group_similar_items in indexes_same:
        for i in range(1, len(group_similar_items)):
            new_coeffs[group_similar_items[0]] += coeffs[group_similar_items[i]]

    if len(indexes_same) != 0:
        indices = [y for x in indexes_same for y in x[1:]]
        new_exps = np.delete(new_exps, indices)
        new_coeffs = np.delete(new_coeffs, indices)
    return new_coeffs, new_exps


def get_next_choices(factor, coeffs, fparams, coeff_val=100.):
    r"""
    Get the next set of (n+1) fparams, used by the greedy-fitting algorithm.

    Given a set of n fparams and n coeffs, this method gets the next (n+1)
    fparams by using a constant factor and coefficient value :math:`c^\prime`.
    A list of (n+1) choices are returned. They are determined as follows,
    if fparams = [a1, a2, ..., an] and coeffs = [c1, c2, ..., cn].
    Then each choice is either
                [a1 / factor, a2, a3, .., an] & coeffs = [c1, c^\prime, c2, ..., cn],
                [a1, a2, ..., (ai + a(i+1)/2, a(i+1), ..., an] & similar coeffs,
                [a1, a2, ..., factor * an] & coeffs = [c1 c2, ..., c^\prime, cn].

    Parameters
    ----------
    factor : float
        Number used to give two choices by multiplying each end point.
    coeffs : np.ndarray
        Coefficients of the basis functions.
    fparams : np.ndarray
        Function parameters.
    coeff_val : float
        Number used to fill in the coefficient value for each guess.

    Returns
    -------
    List[List[np.ndarray, np.ndarray]]
        List of lists where the next possible initial choices for greedy based on factor.

    """
    size = fparams.shape[0]
    all_choices = []
    for index, exp in np.ndenumerate(fparams):
        if index[0] == 0:
            exps_arr = np.insert(fparams, index, exp / factor)
            coeffs_arr = np.insert(coeffs, index, coeff_val)
        elif index[0] <= size:
            exps_arr = np.insert(fparams, index, (fparams[index[0] - 1] +
                                                  fparams[index[0]]) / 2)
            coeffs_arr = np.insert(coeffs, index, coeff_val)
        all_choices.append(np.append(coeffs_arr, exps_arr))
        if index[0] == size - 1:
            exps_arr = np.append(fparams, np.array([exp * factor]))
            endpt = np.append(coeffs, np.array([coeff_val]))
            all_choices.append(np.append(endpt, exps_arr))
    return all_choices


def get_two_next_choices(factor, coeffs, fparams, coeff_val=100.):
    r"""
    Return a list of (n+2) set of initial guess for fparams for greedy.

    Assuming coeffs=[c1, c2 ,... ,cn] and fparams =[a1, a2, ..., an]. The
    next (n+2) choice is either a combination of a endpoint guess and a
    midpoint, or two mid point guess or two endpoint guess. In other words:
        [a1 / factor, ..., (ai + a(i+1))/2, ..., an],
        [a1, ..., (aj + a(j+1))/2, a(j+1) ..., (ai + a(i+1))/2, a(i+1), .., an],
        [a1 / factor, a2, a3, ..., a(n-1), factor * an], respectively.

    Parameters
    ----------
    factor : float
        Number used to give two choices by multiplying each end point.
    coeffs : np.ndarray(M,)
        Coefficients of the basis function set of size :math:`M`.
    fparams : np.ndarray(M,)
        Function parameters of the basis function set of size :math:`M`.
    coeff_val : float
        Number used to fill in the coefficient value for each guess.

    Returns
    -------
    List[List[np.ndarray, np.ndarray]]
        List of lists where the next possible initial choices for greedy based on factor.

    """
    size = len(fparams)
    choices_coeff = []
    choices_fparams = []
    for i, e in enumerate(fparams):
        if i == 0:
            fparam_arr = np.insert(fparams, i, e / factor)
            coeff_arr = np.insert(coeffs, i, coeff_val)
        elif i <= size:
            fparam_arr = np.insert(fparams, i, (fparams[i - 1] + fparams[i]) / 2)
            coeff_arr = np.insert(coeffs, i, coeff_val)

        coeff2, exp2 = get_next_possible_coeffs_and_exps2(factor, coeff_arr, fparam_arr, coeff_val)
        choices_coeff.extend(coeff2[i:])
        choices_fparams.extend(exp2[i:])

        if i == size - 1:
            fparam_arr = np.append(fparams, np.array([e * factor]))
            endpt = np.append(coeffs, np.array([coeff_val]))
            coeff2, exp2 = get_next_possible_coeffs_and_exps2(factor, endpt, fparam_arr, coeff_val)
            choices_coeff.extend(coeff2[-2:])
            choices_fparams.extend(exp2[-2:])
    all_choices_params = []
    for i, c in enumerate(choices_coeff):
        all_choices_params.append(np.append(c, choices_fparams[i]))
    return all_choices_params


def get_next_possible_coeffs_and_exps2(factor, coeffs, exps, coeff_val=100.):
    # This is the same function as get_next_choices except that
    #   it returns the coefficients and exponents separately.
    size = exps.shape[0]
    all_choices_of_exponents = []
    all_choices_of_coeffs = []
    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_val)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_val)
        all_choices_of_exponents.append(exponent_array)
        all_choices_of_coeffs.append(coefficient_array)
        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            all_choices_of_exponents.append(exponent_array)
            all_choices_of_coeffs.append(np.append(coeffs, np.array([coeff_val])))
    return all_choices_of_coeffs, all_choices_of_exponents


def pick_two_lose_one(factor, coeffs, exps, coeff_val=100.):
    r"""
    Get (n+1) choices by choosing (n+2) choices and losing one value each time.

    Most accurate out of the other methods in this file but has returns
    a large number of possible (n+1) initial guesses for greedy-algorithm.

    Using the choices from 'get_two_next_choices', this
    method removes each possible exponent, to get another (n+1) choice.

    Parameters
    ----------
    factor : float
        Number used to give two choices by multiplying each end point.
    coeffs : np.ndarray(M,)
        Coefficients of the basis function set of size :math:`M`.
    fparams : np.ndarray(M,)
        Function parameters of the basis function set of size :math:`M`.
    coeff_val : float
        Number used to fill in the coefficient value for each guess.

    Returns
    -------
    List[List[np.ndarray, np.ndarray]]
        List of lists where the next possible initial choices for greedy based on factor.

    """
    all_choices = []
    two_choices = get_two_next_choices(factor, coeffs, exps, coeff_val)
    for i, p in enumerate(two_choices):
        coeff, exp = p[:len(p) // 2], p[len(p) // 2:]
        for j in range(0, len(p) // 2):
            new_coeff = np.delete(coeff, j)
            new_exp = np.delete(exp, j)
            all_choices.append(np.append(new_coeff, new_exp))
    return all_choices


class GreedyStrategy(metaclass=ABCMeta):
    r"""Base greedy strategy class for fitting s-type, p-type Gaussians"""

    def __init__(self, grid, choice_function="pick-one", scale=2.0):
        r"""
        Construct the base greedy class.

        Parameters
        ----------
        choice_function : str
            Determines how the next set of basis-functions are chosen.
            Can be either:
                "pick-one" : Add a new basis-function by taking the average between every two
                             s-type or p-type exponents.
                "pick-two" : Add two new basis-functions by recursion of "pick-one" over each
                             guess from "pick-one".
                "pick-two-lose-one" : Add new basis-function by iterating though each guess
                                      in "pick-two" and removing one basis-function, generating
                                      a new guess each time.
        scale : float
            Positive number that scales the function parameters of each guess.

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
    def eval_obj_function(self):
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

    def eval_model(self, params):
        r"""Evaluate the model at given set of points."""
        return self.model.evaluate(params[:len(params)//2], params[len(params)//2:])

    def eval_obj_function(self, params):
        r"""Evaluate the objective function."""
        model = self.eval_model(params)
        return self.grid.integrate(self.gaussian_obj.measure.evaluate(self.density, model))

    def get_next_iter_params(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.next_param_func(self.scale, coeffs, exps)

    def store_errors(self, params):
        r"""Store errors inside the attribute `err_arr`."""
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
        param_list : List[`num_s_choices` + `num_p_choices`]
            List of size `num_s_choices` + `num_p_choices` of each initial guess.
        num_s_choices : int
            Number of guesses for s-type Gaussians.
        num_p_choices : int
            Number of guesses for p-type Gaussians.

        Returns
        -------
        (float, List, bool) :
            Returns the value of the cost-function and the best parameters corresponding
            to that value of cost-function.
            True if adding S-type is the most optimal, False otherwise.

        """
        # Initialize the values being returned.
        best_local_value = 1e10
        best_local_param = None
        is_s_optimal = False
        for i in range(0, num_s_choices + num_p_choices):
            # Get the function parameter guess.
            param = param_list[i]

            # Update model for the new number of S-type and P-type functions.
            if i < num_s_choices:
                # Update the number of s-type basis functions for this guess.
                self.model.change_numb_s_and_numb_p(self.num_s + self.numb_func_increase, self.num_p)
            else:
                # Update the number of p-type basis functions for this guess.
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p + self.numb_func_increase)

            # Optimize using this initial guess and get cost-function value.
            local_param = self.get_optimization_routine(param, local=True)
            cost_func = self.eval_obj_function(local_param)
            # If it is the best found, then return it.
            if cost_func < best_local_value:
                best_local_value = self.eval_obj_function(local_param)
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

    def __call__(self, factor, d_threshold=1e-8, max_numb_funcs=30, add_extra_choices=None, disp=False):
        r"""
        Run the greedy algorithm for fitting s-type, p-type Gaussians to a density.

        Initially, the algorithm solves for the best one-function basis-function.
        Then it generates next initial guesses based on the previous choice, and
        optimizes each initial guess, with a small threshold for convergence.
        It takes the best initial guess from that set and optimizes it further.
        Then this process repeats until convergence or termination criteria is met.

        Parameters
        ----------
        factor : float
            The factor that is used to generate new initial guess.
        d_threshold : float
            The convergence threshold for the objective function being minimized.
        max_numb_funcs : int
            Maximum number of basis-functions to have.
        add_extra_choices : callable(List, List[List])
            Function that returns extra initial guesses to the current guess.
            Input must be the model parameters.
        disp : bool
            Whether to display the output.

        Returns
        -------
        result : dict
            The optimization results presented as a dictionary containing:
            "coeffs" : ndarray
                The optimized coefficients of Gaussian model.
            "exps" : ndarray
                The optimized exponents of Gaussian model.
            "success": bool
                Whether or not the optimization exited successfully.
            "fun" : ndarray
                Values of KL divergence (objective function) at each iteration.
            "performance" : ndarray
                Values of various performance measures of modeled density at each iteration,
                as computed by `goodness_of_fit()` method.
            "parameters_iteration": List
                List of the optimal parameters of each iteration.
            "exit_information": str
                Information about termination.

        """
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
        while numb_funcs < max_numb_funcs - 1 and numb_redum < 5 and np.abs(best_gval - prev_gval) >= d_threshold:
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

        coeffs, exps = self._split_parameters(gparams)
        results = {"coeffs": coeffs,
                   "exps" : exps,
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
        self.gaussian_obj = ScipyFit(grid, density, self.model, measure="LS", method=method)

    @property
    def density(self):
        return self.gaussian_obj.density

    def optimize_using_nnls(self, true_dens, cofactor_matrix):
        r"""Solve for the coefficients using non-linear least squares"""
        b_vector = np.copy(true_dens)
        b_vector = np.ravel(b_vector)
        row_nnls_coefficients = nnls(cofactor_matrix, b_vector)
        return row_nnls_coefficients[0]

    def _solve_one_function_weight(self, weight):
        r"""Helper function for solving best one-basis function solution."""
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
        r"""Obtain the best one s-type function solution to least-squares using different weights."""
        # Minimizing weighted least squares with three different weights
        weight1 = np.ones(len(self.grid.points))
        weight3 = np.power(self.density, 2.)
        p1 = self._solve_one_function_weight(weight1)
        cost_func1 = self.eval_obj_function(p1)

        p2 = self._solve_one_function_weight(self.density)
        cost_func2 = self.eval_obj_function(p2)

        p3 = self._solve_one_function_weight(weight3)
        cost_func3 = self.eval_obj_function(p3)

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
            results = self.gaussian_obj.run(coeffs, exps, tol=self.local_tol, maxiter=1000)
        else:
            results = self.gaussian_obj.run(coeffs, exps, tol=self.global_tol, maxiter=1000, disp=True,
                                            with_constraint=True)
        return np.hstack((results["coeffs"], results["exps"]))

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
        r"""Density that is fitted to."""
        return self.mbis_obj.density

    @property
    def integral_dens(self):
        r"""Integral of the density."""
        return self.mbis_obj.integral_dens

    @property
    def grid(self):
        r"""Grid class object."""
        return self.mbis_obj.grid

    def get_best_one_function_solution(self):
        r"""Obtain the best one s-type function to Kullback-Leibler."""
        denom = self.grid.integrate(self.density * np.power(self.grid.points ** 2, 2.))
        exps = 3. * self.integral_dens / (2. * 4. * np.pi * denom)
        return np.array([self.integral_dens, exps])

    def get_optimization_routine(self, params, local=False):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        if local:
            result = self.mbis_obj.run(coeffs, exps, opt_coeffs=True, opt_expons=True,
                                     maxiter=500, c_threshold=1e-2, e_threshold=1e-3,
                                     disp=False)
            return np.hstack((result["coeffs"], result["exps"]))
        result = self.mbis_obj.run(coeffs, exps, opt_coeffs=True, opt_expons=True, maxiter=1000,
                                 c_threshold=self.threshold_coeff, e_threshold=self.threshold_exp,
                                   d_threshold=1e-6, disp=False)
        return np.hstack((result["coeffs"], result["exps"]))

    def get_errors_from_model(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.mbis_obj.goodness_of_fit(coeffs, exps)


if __name__ == "__main__":
    from bfit.grid import ClenshawRadialGrid
    from bfit.density import SlaterAtoms
    grid = ClenshawRadialGrid(4, num_core_pts=10000, num_diffuse_pts=900, extra_pts=[50, 75, 100])
    dens_obj = SlaterAtoms("be")
    dens = dens_obj.atomic_density(grid.points)

    greedy = GreedyKL(grid, dens, integration_val=4.0, eps_coeff=1e-7, eps_exp=1e-8)
    # greedy = GreedyLeastSquares(grid, dens, choice="pick-one", method="SLSQP")
    greedy(2.0)
