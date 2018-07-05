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
Contains utility functions that are used by any greedy-fitting algorithms.

As of know, most of these functions rely on the basis function to depend on
only one parameters. E.g. e^(-x) or e^(-x^2) but not e^(-x - y).
"""
import numpy as np

__all__ = ["check_redundancies", "get_two_next_choices", "get_next_choices",
           "get_next_possible_coeffs_and_exps2"]


def check_redundancies(coeffs, fparams, simi=1e-3):
    r"""
    Check if the fparams have similar values and groups them together.

    If any two function parameters have similar values, then one of the
    function parameters is removed, and the corresponding coefficient,
    is doubled.
    Note: as of now this only works if each basis function depends on only one
    parameters e.g. e^(-x), not e^(-x + y).

    Parameters
    ----------
    coeffs : np.ndarray
             Coefficients of the basis function set.

    fparams : np.ndarray
              Function Parameters of the basis function set.

    simi : float
           Value that indicates the treshold for how close two parameters are.

    Returns
    -------
    (np.ndarray, np.ndarray)
                            New coefficients and new exponents, where close
                            values are grouped together.

    """
    new_coeffs = coeffs.copy()
    new_exps = fparams.copy()

    # Indexes where they're the same.
    indexes_same = []
    for i, alpha in enumerate(fparams):
        similar_indexes = []
        for j in range(i + 1, len(fparams)):
            if j not in similar_indexes:
                if np.abs(alpha - fparams[j]) < simi:
                    if i not in similar_indexes:
                        similar_indexes.append(i)
                    similar_indexes.append(j)
        if len(similar_indexes) != 0:
            indexes_same.append(similar_indexes)

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
    fparams by using the factor.
    A list of (n+1) choices are returned. They are determined as follows,
    if fparams = [a1, a2, ..., an] and coeffs = [c1, c2, ..., cn].
    Then each choice is either
                [a1 / factor, a2, a3, .., an] & coeffs = [c1, c1, c2, ..., cn],
                [a1, a2, ..., (ai + a(i+1)/2, a(i+1), ..., an] & similar coeffs,
                [a1, a2, ..., factor * an] & coeffs = [c1 c2, ..., cn, cn].
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
    list
        List of the next possible initial choices for greedy based on factor.

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
        [a1 / factor, a2, a3, ..., a(n-1), factor * an], respetively.

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
    list
        List of the next possible initial choices for greedy based on factor.

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
    
    coeffs : np.ndarray
             Coefficients of the basis functions.
    
    fparams : np.ndarray
              Function parameters.
    
    coeff_val : float
                Number used to fill in the coefficient value for each guess.
    
    Returns
    -------
    list
        List of the next possible initial choices for greedy based on factor.

    """
    all_choices = []
    two_choices = get_two_next_choices(factor, coeffs, exps, coeff_val)
    for i, p in enumerate(two_choices):
        coeff, exp = p[:len(p)//2], p[len(p)//2:]
        for j in range(0, len(p)//2):
            new_coeff = np.delete(coeff, j)
            new_exp = np.delete(exp, j)
            all_choices.append(np.append(new_coeff, new_exp))
    return all_choices
