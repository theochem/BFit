# -*- coding: utf-8 -*-
# BFit is a Python library for fitting a convex sum of Gaussian
# functions to any probability distribution
#
# Copyright (C) 2020- The QC-Devs Community
#
# This file is part of BFit.
#
# BFit is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# BFit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""Greedy Fitting Module."""

from abc import ABCMeta, abstractmethod

from bfit.fit import _BaseFit, KLDivergenceFPI, ScipyFit
from bfit.measure import SquaredDifference
from bfit.model import AtomicGaussianDensity
import numpy as np
from scipy.optimize import nnls

__all__ = ["GreedyLeastSquares", "GreedyKLFPI"]


def remove_redundancies(coeffs, fparams, eps=1e-3):
    r"""
    Check if the exponents have similar values and group them together.

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
    (np.ndarray, np.ndarray) :
        New coefficients and new function parameters, where close values of the function parameters
        are removed and the coefficients corresponding to that parameter are added together.

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

    Assuming coeffs=:math:`\[c1, c2 ,... ,cn\]` and fparams =:math:`\[a1, a2, ..., an\]`,
    this method gets the next (n+1) fparams by using a constant factor
    and coefficient value :math:`c^\prime`. A list of (n+1) choices are
    returned. They are determined as follows:

    .. math::
        \begin{align*}
            \[a_1 / factor, a_2,  .., a_n\] &,\quad coeffs = [c1, c^\prime, c2, ..., cn], \\
            \[[a_1, a_2,...,(a_i + a_{(i+1)/2}, a_{i+1}, ..., an\] &,\quad \text{similar coeffs},\\
            \[[a_1, a_2, ..., factor * a_n\] &,\quad \text{similar coeffs},
        \end{align*}

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
    List[np.ndarray]
        List of the next possible initial guesses for :math:`(n+1)` basis-functions,
        coefficients are listed first, then exponents.
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

    Assuming coeffs=:math:`\[c1, c2 ,... ,cn\]` and fparams=:math:`\[a1, a2, ..., an\]`.
    The next (n+2) choice is either a combination of an endpoint guess and a
    midpoint, or two mid point guess or two endpoint guess.
    This is two-times recursion of the function `get_next_choices`.

    .. math::
        \begin{align*}
            \[a_1 / factor, ..., (a_i + a_{(i+1))/2}, ..., a_n\],
            \[a_1, ..., (a_j + a_{(j+1))/2}, a_{j+1} ..., (a_i + a_{(i+1))/2}, a_{i+1}, .., a_n\],
            \[a_1 / factor, a_2, a_3, ..., a_{n-1}, factor * a_n\]
        \end{align*}

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
    List[np.ndarray]
        List of the next possible initial guesses for :math:`(n+2)` basis-functions,
        coefficients are listed first, then exponents.

    """

    def _get_next_possible_coeffs_and_exps2(factor, coeffs, exps, coeff_val=100.):
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

    size = len(fparams)
    choices_coeff = []
    choices_fparams = []
    for i, e in enumerate(fparams):
        if i == 0:
            fparam_arr = np.insert(fparams, 0, e / factor)
            coeff_arr = np.insert(coeffs, 0, coeff_val)
        elif i <= size:
            fparam_arr = np.insert(fparams, i, (fparams[i - 1] + fparams[i]) / 2)
            coeff_arr = np.insert(coeffs, i, coeff_val)

        coeff2, exp2 = _get_next_possible_coeffs_and_exps2(factor, coeff_arr, fparam_arr, coeff_val)
        choices_coeff.extend(coeff2[i:])
        choices_fparams.extend(exp2[i:])

        if i == size - 1:
            fparam_arr = np.append(fparams, np.array([e * factor]))
            endpt = np.append(coeffs, np.array([coeff_val]))
            coeff2, exp2 = _get_next_possible_coeffs_and_exps2(factor, endpt, fparam_arr, coeff_val)
            choices_coeff.extend(coeff2[-2:])
            choices_fparams.extend(exp2[-2:])

    # Append coefficients and exponents together.
    all_choices_params = []
    for i, c in enumerate(choices_coeff):
        all_choices_params.append(np.append(c, choices_fparams[i]))
    return all_choices_params


def pick_two_lose_one(factor, coeffs, exps, coeff_val=100.):
    r"""
    Get (n+1) initial guesses by choosing (n+2) guesses and losing one value each time.

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
    List[np.ndarray]
        List of the next possible initial guesses for `(n+1)` basis-functions,
        coefficients are listed first, then exponents.
    """
    all_choices = []
    two_choices = get_two_next_choices(factor, coeffs, exps, coeff_val)
    for p in two_choices:
        coeff, exp = p[:len(p) // 2], p[len(p) // 2:]
        for j in range(0, len(p) // 2):
            new_coeff = np.delete(coeff, j)
            new_exp = np.delete(exp, j)
            all_choices.append(np.append(new_coeff, new_exp))
    return all_choices


class GreedyStrategy(metaclass=ABCMeta):
    r"""Base greedy strategy class for fitting s-type, p-type Gaussians."""

    def __init__(self, fitting_obj, choice_function="pick-one"):
        r"""
        Construct the base greedy class.

        Parameters
        ----------
        choice_function : str
            Determines how the next set of basis-functions are chosen.
            Can be either:

            * `pick-one` : add a new basis-function by taking the average between every two
              s-type or p-type exponents.
            * `pick-two` : add two new basis-functions by recursion of "pick-one" over each
              guess from `pick-one`.
            * `pick-two-lose-one` : add new basis-function by iterating though each guess
              in `pick-two` and removing one basis-function, generating a new guess each time.
        fitting_obj : _BaseFit
            The fitting class that is a child of _BaseFit class.  This is used to
            optimize the objective function.

        """
        if not isinstance(fitting_obj, _BaseFit):
            raise TypeError(f"Fitting object {type(fitting_obj)} should be of type _BaseFit.")

        if choice_function == "pick-one":
            self.next_param_func = get_next_choices
            self._numb_func_increase = 1  # How many basis functions were increased
        elif choice_function == "pick-two":
            self.next_param_func = get_two_next_choices
            self._numb_func_increase = 2
        elif choice_function == "pick-two-lose-one":
            self.next_param_func = pick_two_lose_one
            self._numb_func_increase = 1
        else:
            raise ValueError(f"Choice parameter {choice_function} was not recognized.")
        self.num_s = 1
        self.num_p = 0
        self.err_arr = []
        self.fitting_obj = fitting_obj

    @property
    def numb_func_increase(self):
        r"""Return number of basis-functions to add in each iteration."""
        return self._numb_func_increase

    @property
    def density(self):
        r"""Return density that is fitted to."""
        return self.fitting_obj.density

    @property
    def grid(self):
        r"""Return grid class object."""
        return self.fitting_obj.grid

    @property
    def model(self):
        r"""Return model class."""
        return self.fitting_obj.model

    @property
    def measure(self):
        r"""Return measure class."""
        return self.fitting_obj.measure

    @property
    def integral_dens(self):
        r"""Return the integral of the density."""
        return self.fitting_obj.integral_dens

    @abstractmethod
    def get_best_one_function_solution(self):
        r"""Return the solution of the model parameters for one basis function."""
        # Usually the model with single basis function can be solved analytically.
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_routine(self, params, *args, local=False):
        r"""Run the optimization for fitting the Gaussian model to a density."""
        raise NotImplementedError()

    def eval_obj_function(self, params):
        r"""Return evaluation the objective function."""
        model = self.model.evaluate(params[:len(params)//2], params[len(params)//2:])
        integrand = self.measure.evaluate(self.density, model)
        return self.fitting_obj.integrate(integrand)

    def store_errors(self, params):
        r"""Store errors inside the attribute `err_arr`."""
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        err = self.fitting_obj.goodness_of_fit(coeffs, exps)
        if len(self.err_arr) == 0:
            self.err_arr = [err]
        else:
            self.err_arr.append(err)

    def _find_best_lparams(self, param_list, num_s_choices, num_p_choices):
        r"""
        Return the best initial guess from a list of potential model parameter choices.

        Parameters
        ----------
        param_list : List[`num_s_choices` + `num_p_choices`]
            List of size `num_s_choices` + `num_p_choices` of each initial guess.
        num_s_choices : int
            Number of guesses for adding extra s-type Gaussians.
        num_p_choices : int
            Number of guesses for adding extra p-type Gaussians.

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
                self.model.change_numb_s_and_numb_p(
                    self.num_s + self.numb_func_increase, self.num_p
                )
            else:
                # Update the number of p-type basis functions for this guess.
                self.model.change_numb_s_and_numb_p(
                    self.num_s, self.num_p + self.numb_func_increase
                )

            # Optimize using this initial guess and get cost-function value.
            local_param = self.get_optimization_routine(param, local=True)
            cost_func = self.eval_obj_function(local_param)
            # If it is the best found, then return it.
            if cost_func < best_local_value:
                best_local_value = self.eval_obj_function(local_param)
                best_local_param = local_param
                is_s_optimal = bool(i < num_s_choices)
        return best_local_value, best_local_param, is_s_optimal

    def _split_parameters(self, params):
        r"""Split parameters into the s-type, p-type coefficients and exponents."""
        s_coeffs = params[:self.num_s]
        p_coeffs = params[self.num_s:self.num_s + self.num_p]
        s_exps = params[self.num_s + self.num_p:2 * self.num_s + self.num_p]
        p_exps = params[2 * self.num_s + self.num_p:]
        return s_coeffs, s_exps, p_coeffs, p_exps

    def _print_header(self):
        r"""Print the initial output for the greedy algorithm."""
        # Template for the header.
        print("-" * (10 + 12 + 15 + 15 + 15 + 15 + 15 + 15 + 15 + 17))
        # Format is {identifier:width}, ^ means center it, | means put a bar in it
        template_header = (
            "|{0:^15}|{1:^13}|{2:^13}|{3:^15}|{4:^15}|{5:^16}|{6:^15}|{7:^16}|{8:^16}|"
        )
        # Print the headers
        print(template_header.format(
            "# of Functions", "# of S-type", "# of P-type",
            "Integration", "L1", "L Infinity", "Least-squares",
            "Kullback-Leibler", "Change Objective")
        )
        print("-" * (10 + 12 + 15 + 15 + 15 + 15 + 15 + 15 + 15 + 17))
        # Template for float iteration
        template_iters = (
            "|{0:^15d}|{1:^13d}|{2:^13d}|{3:^15f}|{4:^15e}|{5:^16e}|{6:^15e}|{7:^16e}|{8:^16e}|"
        )
        print(template_iters.format(
            1, self.num_s, self.num_p, *self.err_arr[-1], np.nan
        ))
        return template_iters

    def run(self, factor, d_threshold=1e-8, max_numb_funcs=30, add_extra_choices=None, disp=False):
        r"""
        Add new Gaussians to fit to a density until convergence is achieved.

        Initially, the algorithm solves for the best one-function basis-function.
        Then it generates a group of initial guesses of size `(1 + C)`
        based on the previous optimal fit. It then optimizes each initial guess
        from that group, with a small threshold for convergence.
        It takes the best found initial guess from that set and optimizes it further.
        Then this process repeats for `(1 + 2C)` basis-functions until convergence or
        termination criteria is met.

        Parameters
        ----------
        factor : float
            The factor that is used to generate new initial guess.
        d_threshold : float
            The convergence threshold for the objective function being minimized.
        max_numb_funcs : int
            Maximum number of basis-functions to have.
        add_extra_choices : callable(List, List[List])
            Function that returns extra initial guesses to add.
            Input must be the model parameters and the output should be a
            list of initial guesses that should match attribute `numb_func_increase`.
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
            "num_s" : int
                Number of s-type Gaussian functions.
            "num_p" : int
                Number of p-type Gaussian functions.
            "success": bool
                Whether or not the optimization exited successfully.
            "fun" : float
                Objective function at the last iteration.
            "performance" : ndarray
                Values of various performance measures of modeled density at each iteration,
                as computed by `goodness_of_fit()` method.
            "parameters_iteration": List
                List of the optimal parameters of each iteration.
            "exit_information": str
                Information about termination of the greedy algorithm.

        """
        if not isinstance(factor, float):
            raise TypeError(f"Factor {factor} parameter should be a float.")
        if factor <= 0.0:
            raise ValueError(f"Scale {factor} should be positive.")

        # Initialize all the variables
        gparams = self.get_best_one_function_solution()
        self.store_errors(gparams)
        exit_info = None  # String containing information about how it exits.
        numb_funcs = 1  # Number of current functions in model.
        prev_gval, best_gval = np.inf, self.eval_obj_function(gparams)
        params_iter = [gparams]  # Storing the parameters at each iteration.
        numb_redum = 0  # Number of redundancies, termination criteria.
        factor0 = factor  # Scaling parameter that changes.

        # Start the greedy algorithm
        if disp:
            template_iters = self._print_header()

        success = True
        while numb_funcs <= max_numb_funcs - 1 and numb_redum < 5 and \
                np.abs(best_gval - prev_gval) >= d_threshold:
            s_coeffs, s_exps, p_coeffs, p_exps = self._split_parameters(gparams)

            # Get the next list of choices of parameters for S-type and P-type orbitals.
            # Add new S-type and then P-types initial guess.
            choices_parameters_s = self.next_param_func(factor, s_coeffs, s_exps)
            choices_parameters_s = [
                np.hstack((x[:self.num_s + self.numb_func_increase], p_coeffs,
                           x[self.num_s + self.numb_func_increase:], p_exps))
                for x in choices_parameters_s
            ]
            if self.num_p == 0:
                # Get Three Guesses one from 0 to 1, 1 to 10 and 10 to 100.
                choices_parameters_p = [
                    np.random.random((2 * self.numb_func_increase,)),
                    np.random.random((2 * self.numb_func_increase,)) * 9 + 1,
                    np.random.random((2 * self.numb_func_increase,)) * 90 + 10,
                ]
            else:
                choices_parameters_p = self.next_param_func(factor, p_coeffs, p_exps)
            choices_parameters_p = [
                np.hstack((s_coeffs, x[:self.num_p + self.numb_func_increase],
                           s_exps, x[self.num_p + self.numb_func_increase:]))
                for x in choices_parameters_p
            ]
            num_s_choices = len(choices_parameters_s)
            num_p_choices = len(choices_parameters_p)
            total_choices = choices_parameters_s + choices_parameters_p
            if callable(add_extra_choices):
                # When you want extra choices to be added.
                total_choices.append(add_extra_choices(gparams))

            # Run fast, quick optimization and find the best parameter out of the choices.
            _, best_lparam, is_s_optimal = self._find_best_lparams(
                total_choices, num_s_choices, num_p_choices
            )

            # Update model for the new number of S-type and P-type functions.
            if is_s_optimal:
                # If adding s-type was optimal then increase number of s-type.
                self.num_s += self.numb_func_increase
            else:
                self.num_p += self.numb_func_increase
            self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
            numb_funcs += self.numb_func_increase

            # Optimize further the best local answer and get new objective func
            opt_lparam = self.get_optimization_routine(best_lparam, local=False)
            opt_lvalue = self.eval_obj_function(opt_lparam)

            # Check if redundancies were found in the coefficients and exponents, remove them,
            #   change the factor and try again.
            s_coeffs, s_exps, p_coeffs, p_exps = self._split_parameters(opt_lparam)
            s_coeffs_new, _ = remove_redundancies(s_coeffs, s_exps)
            p_coeffs_new, _ = remove_redundancies(p_coeffs, p_exps)
            found_s_redundances = self.num_s != len(s_coeffs_new)
            found_p_redundancies = self.num_p != len(p_coeffs_new)
            if found_s_redundances or found_p_redundancies or opt_lvalue > best_gval:
                # Move back one function and try a different factor to generate
                #    better initial guesses.
                numb_redum += 1  # Update the termination criteria
                factor *= 2      # Increase the factor that changes the kinds of potential choices.
                if is_s_optimal:
                    self.num_s -= self.numb_func_increase
                else:
                    self.num_p -= self.numb_func_increase
                self.model.change_numb_s_and_numb_p(self.num_s, self.num_p)
                numb_funcs -= self.numb_func_increase
            elif opt_lvalue <= best_gval:
                # Take the best local choice and optimization even further.
                gparams = opt_lparam  # Store for next iteration.
                self.store_errors(gparams)
                params_iter.append(gparams)  # Store parameters.
                prev_gval, best_gval = best_gval, opt_lvalue
                numb_redum = 0    # Reset the number of redundancies.
                factor = factor0  # Reset Original Factor.

            if disp:
                print(template_iters.format(
                    numb_funcs, self.num_s, self.num_p, *self.err_arr[-1],
                    np.abs(best_gval - prev_gval)
                ))

        if numb_funcs == max_numb_funcs - 1 or numb_redum == 5:
            success = False
            if numb_redum == 5:
                exit_info = f"Next Iteration Did Not Find The Best Choice at" \
                            f" number of s-type {self.num_s} and p-type {self.num_p}"

        if exit_info is None:
            exit_info = self._final_exit_info(
                numb_funcs, max_numb_funcs, best_gval, prev_gval, d_threshold
            )

        # Get the coefficients and exponents of the most recent parameters stored.
        coeffs_s, exps_s, coeffs_p, exps_p = self._split_parameters(params_iter[-1])
        obj_func = self.eval_obj_function(params_iter[-1])
        coeffs = np.hstack((coeffs_s, coeffs_p))
        expons = np.hstack((exps_s, exps_p))
        if disp:
            print("-" * (10 + 12 + 15 + 15 + 15 + 15 + 15 + 15 + 15 + 17))
            print(f"Successful?: {success}")
            print(f"Number of s-type: {self.num_s}")
            print(f"Number of p-type: {self.num_p}")
            print("Termination Information: " + exit_info)
            print(f"Objective Function: {obj_func}")
            print(f"Coefficients {coeffs}")
            print(f"Exponents {expons}")

        results = {"coeffs": coeffs,
                   "exps": expons,
                   "num_s": self.num_s,
                   "num_p": self.num_p,
                   "fun": obj_func,
                   "success": success,
                   "parameters_iteration": params_iter,
                   "performance": np.array(self.err_arr),
                   "exit_information": exit_info}
        return results

    @staticmethod
    def _final_exit_info(num_func, max_func, best_val, prev_gval, d_threshold):
        r"""Return string that holds how the greedy algorithm teminated."""
        exit_info = None
        if not num_func < max_func - 1:
            exit_info = "Max number of functions reached,  " + str(num_func)
        elif np.abs(best_val - prev_gval) < d_threshold:
            exit_info = "Cost function is less than some epsilon: " + \
                             str(best_val - prev_gval) + " <= " + str(d_threshold)
        if exit_info is None:
            raise RuntimeError(f"Exit information {exit_info} should not be None. "
                               f"There is an error in the termination of the algorithm.")
        return exit_info


class GreedyLeastSquares(GreedyStrategy):
    r"""Optimize Least-Squares using Greedy and ScipyFit methods."""

    def __init__(
        self, grid, density, choice="pick-one", local_tol=1e-5, global_tol=1e-8,
        method="SLSQP", normalize=False, integral_dens=None, with_constraint=False,
        l_maxiter=1000, g_maxiter=2500, spherical=False,
    ):
        r"""
        Construct the GreedyLestSquares object.

        Parameters
        ----------
        grid : (_BaseRadialGrid, CubicGrid)
            Grid class that contains the grid points and integration methods on them.
        density : ndarray
            The true density evaluated on the grid points.
        choice : str, optional
            Determines how the next set of basis-functions are chosen.
            Can be either:

            * `pick-one` : add a new basis-function by taking the average between every two
              s-type or p-type exponents.
            * `pick-two` : add two new basis-functions by recursion of "pick-one" over each
              guess from `pick-one`.
            * `pick-two-lose-one` : add new basis-function by iterating though each guess
              in `pick-two` and removing one basis-function, generating a new guess each time.
        local_tol : float, optional
            The tolerance for convergence of scipy.optimize method for optimizing each local guess.
            Should be larger than `global_tol`.
        global_tol : float, optional
            The tolerance for convergence of scipy.optimize method for further refining/optimizing
            the best local guess found out of all choices.  Should be smaller than
            `local_tol`.
        method : str, optional
            The method used for optimizing parameters. Default is "slsqp".
            See "scipy.optimize.minimize" for options.
        normalize : bool, optional
            Whether to fit with a normalized s-type and p-type Gaussian model.
        integral_dens : float, optional
            If this is provided, then the model is constrained to integrate to this value.
            If not, then the model is constrained to the numerical integration of the
            density. Useful when one knows the actual integration value of the density.
        with_constraint : bool
            If true, then adds the constraint that the integration of the model density must
            be equal to the constraint of true density. The default is True.
        l_maxiter : int, optional
            Maximum number of iterations when optimizing an initial guess in the `scipy.optimize`
            method for optimizing through all initial guesses (local choices).
        g_maxiter : int, optional
            Maximum number of iterations when optimizing an initial guess in the `scipy.optimize`
            method for further optimizing the best local choice (global choice).
        spherical : bool
            Whether to perform spherical integration by adding :math:`4 \pi r^2` term
            to the integrand when calculating the objective function.
            Only used when grid is one-dimensional and positive (radial grid).

        """
        self._local_tol = local_tol
        self._global_tol = global_tol
        self._l_maxiter = l_maxiter
        self._g_maxiter = g_maxiter
        self._with_constraint = with_constraint
        model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=normalize)
        gaussian_obj = ScipyFit(grid, density, model, measure=SquaredDifference(), method=method,
                                integral_dens=integral_dens, spherical=spherical)
        super().__init__(gaussian_obj, choice)

    @property
    def local_tol(self):
        r"""Return local tolerance for convergence in `scipy.optimize`."""
        return self._local_tol

    @property
    def global_tol(self):
        r"""Return global tolerance for convergence in `scipy.optimize`."""
        return self._global_tol

    @property
    def l_maxiter(self):
        r"""Return maximum number of iterations in optimization routine for all local choices."""
        return self._l_maxiter

    @property
    def g_maxiter(self):
        r"""Return maximum number of iterations in optimization routine for global choice."""
        return self._g_maxiter

    @property
    def with_constraint(self):
        r"""Return constraint that integral of model should equal integral of density."""
        return self._with_constraint

    def _solve_one_function_weight(self, weight):
        r"""Solve the best one-basis function solution."""
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
        r"""Obtain the best s-type function solution to least-squares using different weights."""
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

    def _create_cofactor_matrix(self, exponents):
        r"""Create cofactor matrix for solving nnls."""
        exponents_s = exponents[:self.model.num_s]
        grid_squared_col = np.power(self.grid.points, 2.0).reshape((len(self.grid.points), 1))
        exponential = np.exp(-exponents_s * grid_squared_col)

        if self.model.num_p != 0:
            exponents_p = exponents[self.model.num_s:]
            exponential = np.hstack(
                (exponential, grid_squared_col * np.exp(-exponents_p * grid_squared_col))
            )
        return exponential

    @staticmethod
    def optimize_using_nnls(true_dens, cofactor_matrix):
        r"""Solve for the coefficients using non-linear least squares."""
        b_vector = np.copy(true_dens)
        b_vector = np.ravel(b_vector)
        row_nnls_coefficients = nnls(cofactor_matrix, b_vector)
        return row_nnls_coefficients[0]

    def get_optimization_routine(self, params, local=False):
        r"""Optimize least-squares using nnls and scipy.optimize from ScipyFit."""
        # First solves the optimal coefficients (while exponents are fixed) using NNLS.
        # Then it optimizes both coefficients and exponents using scipy.optimize.
        exps = params[len(params)//2:]
        cofac_matrix = self._create_cofactor_matrix(exps)
        coeffs = self.optimize_using_nnls(self.density, cofac_matrix)
        if local:
            results = self.fitting_obj.run(
                coeffs, exps, tol=self.local_tol, maxiter=self.l_maxiter,
                with_constraint=self.with_constraint
            )
        else:
            results = self.fitting_obj.run(
                coeffs, exps, tol=self.global_tol, maxiter=self.g_maxiter,
                with_constraint=self.with_constraint
            )
        return np.hstack((results["coeffs"], results["exps"]))


class GreedyKLFPI(GreedyStrategy):
    r"""Optimize Kullback-Leibler using the Greedy method and fixed point iteration method."""

    def __init__(
        self, grid, density, choice="pick-one", g_eps_coeff=1e-4, g_eps_exp=1e-5,
            g_eps_obj=1e-10, l_eps_coeff=1e-2, l_eps_exp=1e-3, l_eps_obj=1e-8,
            l_maxiter=1000, g_maxiter=2500, mask_value=1e-12, integral_dens=None,
            spherical=False,
    ):
        r"""
        Construct the GreedyKLFPI object.

        Parameters
        ----------
        grid : (_BaseRadialGrid, CubicGrid)
            Grid class that contains the grid points and integration methods on them.
        density : ndarray
            The true density evaluated on the grid points.
        choice : str, optional
            Determines how the next set of basis-functions are chosen.
            Can be either:

            * `pick-one` : add a new basis-function by taking the average between every two
              s-type or p-type exponents.
            * `pick-two` : add two new basis-functions by recursion of "pick-one" over each
              guess from `pick-one`.
            * `pick-two-lose-one` : add new basis-function by iterating though each guess
              in `pick-two` and removing one basis-function, generating a new guess each time.
        g_eps_coeff : float, optional
            The tolerance for convergence of coefficients in KL-FPI method for further
            refining and optimizing the best found local guess.
        g_eps_exp : float, optional
            The tolerance for convergence of exponents in KL-FPI method for further
            refining/optimizing the best local guess found out of all choices.
        g_eps_obj : float, optional
            The tolerance for convergence of objective function in KL-FPI method for
            further refining/optimizing the best local guess found out of all choices.
        l_eps_coeff : float, optional
            The tolerance for convergence of coefficients in KL-FPI method for optimizing
            each local initial guess. Should be larger than `g_eps_coeff`.
        l_eps_exp : float, optional
            The tolerance for convergence of exponents in KL-FPI method for optimizing
            each local initial guess. Should be larger than `g_eps_exp`.
        l_eps_obj : float, optional
            The tolerance for convergence of objective function in KL-FPI method for optimizing
            each local initial guess. Should be larger than `g_eps_obj`.
        l_maxiter : int, optional
            Maximum number of iterations when optimizing an initial guess in the KL-FPI method.
            for optimizing through all initial guesses (local choices).
        g_maxiter : int, optional
            Maximum number of iterations when optimizing an initial guess in the KL-FPI method
            for further optimizing the best local choice (global choice).
        mask_value : str, optional
            The method used for optimizing parameters. Default is "slsqp".
            See "scipy.optimize.minimize" for options.
        integral_dens : float, optional
            If this is provided, then the model is constrained to integrate to this value.
            If not, then the model is constrained to the numerical integration of the
            density. Useful when one knows the actual integration value of the density.
        spherical : bool
            Whether to perform spherical integration by adding :math:`4 \pi r^2` term
            to the integrand when calculating objective function.
            Only used when grid is one-dimensional and positive (radial grid).

        """
        # Algorithm parameters for KL-FPI (KLDivergenceFPI) method.
        self._l_threshold_coeff = l_eps_coeff
        self._l_threshold_exp = l_eps_exp
        self._l_threshold_obj = l_eps_obj
        self._g_threshold_coeff = g_eps_coeff
        self._g_threshold_exp = g_eps_exp
        self._g_threshold_obj = g_eps_obj
        self._l_maxiter = l_maxiter
        self._g_maxiter = g_maxiter
        # Model that is fitted to.
        model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=True)
        fpi_obj = KLDivergenceFPI(grid, density, model, mask_value, integral_dens, spherical)
        super().__init__(fpi_obj, choice)

    @property
    def l_threshold_coeff(self):
        r"""Return local threshold for KL-FPI method for convergence of coefficients."""
        return self._l_threshold_coeff

    @property
    def l_threshold_exp(self):
        r"""Return local threshold for KL-FPI method for convergence of exponents."""
        return self._l_threshold_exp

    @property
    def l_threshold_obj(self):
        r"""Return local threshold for KL-FPI method for convergence of objective function."""
        return self._l_threshold_obj

    @property
    def g_threshold_coeff(self):
        r"""Return global threshold for KL-FPI method for convergence of coefficients."""
        return self._g_threshold_coeff

    @property
    def g_threshold_exp(self):
        r"""Return global threshold for KL-FPI method for convergence of exponents."""
        return self._g_threshold_exp

    @property
    def g_threshold_obj(self):
        r"""Return global threshold for KL-FPI method for convergence of objective function."""
        return self._g_threshold_obj

    @property
    def l_maxiter(self):
        r"""Return maximum number of iterations in optimization routine for all local choices."""
        return self._l_maxiter

    @property
    def g_maxiter(self):
        r"""Return maximum number of iterations in optimization routine for global choice."""
        return self._g_maxiter

    def get_best_one_function_solution(self):
        r"""Obtain the best one s-type function to Kullback-Leibler."""
        denom = self.grid.integrate(self.density * np.power(self.grid.points ** 2, 2.))
        exps = 3. * self.integral_dens / (2. * 4. * np.pi * denom)
        return np.array([self.integral_dens, exps])

    def get_optimization_routine(self, params, local=False):
        r"""Optimize KL using KL-FPI method."""
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        if local:
            result = self.fitting_obj.run(
                coeffs, exps, opt_coeffs=True, opt_expons=True,
                maxiter=self.l_maxiter, c_threshold=self.l_threshold_coeff,
                e_threshold=self.l_threshold_exp, d_threshold=self.l_threshold_obj,
                disp=False
            )
            return np.hstack((result["coeffs"], result["exps"]))
        result = self.fitting_obj.run(
            coeffs, exps, opt_coeffs=True, opt_expons=True, maxiter=self.g_maxiter,
            c_threshold=self.g_threshold_coeff, e_threshold=self.g_threshold_exp,
            d_threshold=self.g_threshold_obj, disp=False
        )
        return np.hstack((result["coeffs"], result["exps"]))
