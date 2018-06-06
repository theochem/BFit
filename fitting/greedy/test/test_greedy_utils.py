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
r"""Test file for 'fitting.greedy.greedy_utils'. """


import numpy.testing as npt
import numpy as np
from fitting.greedy.greedy_utils import check_redundancies, get_next_choices, \
    get_next_possible_coeffs_and_exps2, get_two_next_choices, pick_two_lose_one


__all__ = ["test_check_redundancies", "test_get_next_possible_coeffs_and_exps",
           "test_get_next_possible_coeffs_and_exps2",
           "test_get_two_next_possible_coeffs_and_exps", "test_pick_two_lose_one"]


def test_check_redundancies():
    r"""Testing for checking redundancies for greedy."""
    c = np.array([50., 30., 10., 2., 3.])
    exps = np.array([3.1, 3., 2, 1, 3.025])
    simi = 1.
    true_answer = check_redundancies(c, exps, simi)
    desired_answer_coeff = np.array([83., 10., 2.])
    desired_answer_exps = np.array([3.1, 2., 1.])
    npt.assert_allclose(true_answer[0], desired_answer_coeff)
    npt.assert_allclose(true_answer[1], desired_answer_exps)

    c = np.array([50., 30., 10., 2., 3.])
    exps = np.array([3.0012, 3.0005, 2, 1, 3.025])
    simi = 1e-2
    true_answer = check_redundancies(c, exps, simi)
    npt.assert_array_equal(true_answer[0], np.array([80, 10., 2., 3]))
    npt.assert_array_equal(true_answer[1], np.array([3.0012, 2., 1., 3.025]))

    simi = 1e-1
    true_answer = check_redundancies(c, exps, simi)
    npt.assert_array_equal(true_answer[0], np.array([83, 10., 2]))
    npt.assert_array_equal(true_answer[1], np.array([3.0012, 2, 1]))


def test_get_next_possible_coeffs_and_exps():
    r"""Testing 'fitting.greedy.greedy_utils.get_next_choices'."""
    c = np.array([1., 2.])
    e = np.array([3., 4.])
    factor = 2
    true_answer = get_next_choices(factor, c, e, coeff_val=0)
    desired_answer = [[0., 1, 2, 1.5, 3, 4],
                      [1, 0, 2, 3, 3.5, 4],
                      [1, 2, 0, 3, 4, 8]]
    npt.assert_array_equal(true_answer, desired_answer)


def test_get_next_possible_coeffs_and_exps2():
    c = np.array([1., 2.])
    e = np.array([3., 4.])
    factor = 2
    true_answer = get_next_possible_coeffs_and_exps2(factor, c, e, coeff_val=0)
    desired_answer_coeff = [[0., 1., 2], [1., 0., 2], [1., 2., 0]]
    desired_answer_exps = [[1.5, 3, 4], [3, 3.5, 4], [3., 4., 8]]
    npt.assert_array_equal(true_answer[0], desired_answer_coeff)
    npt.assert_array_equal(true_answer[1], desired_answer_exps)


def test_get_two_next_possible_coeffs_and_exps():
    r"""Testing 'fitting.greedy.greedy_utils.get_two_next_choices'."""
    # One Basis Function
    c = np.array([1.])
    e = np.array([3.])
    factor = 2
    true_answer = get_two_next_choices(factor, c, e, coeff_val=0)
    desired_answer = [[0, 0, 1, 0.75, 1.5, 3],
                      [0, 0, 1, 1.5, 2.25, 3],
                      [0, 1, 0, 1.5, 3, 6],
                      [1, 0, 0, 3, 4.5, 6],
                      [1, 0, 0, 3, 6, 12]]
    npt.assert_array_equal(true_answer, desired_answer)

    # Two Basis Function
    c = np.array([1., 2.])
    e = np.array([3., 4.])
    factor = 2
    true_answer = get_two_next_choices(factor, c, e, coeff_val=0)
    desired_answer = [[0, 0, 1, 2, 0.75, 1.5, 3., 4.],
                      [0, 0, 1, 2, 1.5, 2.25, 3, 4],
                      [0, 1., 0, 2, 1.5, 3, 3.5, 4.],
                      [0, 1, 2, 0, 1.5, 3, 4., 8.],
                      [1, 0, 0, 2, 3, 3.25, 3.5, 4],
                      [1, 0, 0, 2, 3, 3.5, 3.75, 4.],
                      [1, 0, 2, 0, 3, 3.5, 4., 8.],
                      [1, 2, 0, 0, 3, 4, 6, 8],
                      [1, 2, 0, 0, 3, 4, 8, 16]]
    npt.assert_array_equal(true_answer, desired_answer)


def test_pick_two_lose_one():
    r"""Testing 'fitting.greedy.greedy_utils.pick_two_lose_one'."""
    c = np.array([1])
    e = np.array([3.])
    factor = 2
    true_answer = pick_two_lose_one(factor, c, e, coeff_val=0)
    desired_answer = [[0, 1, 1.5, 3],
                      [0, 1, 0.75, 3],
                      [0, 0, 0.75, 1.5],
                      [0, 1, 2.25, 3],
                      [0, 1, 1.5, 3],
                      [0, 0, 1.5, 2.25],
                      [1, 0, 3, 6],
                      [0, 0, 1.5, 6],
                      [0, 1, 1.5, 3],
                      [0, 0, 4.5, 6],
                      [1, 0, 3, 6],
                      [1, 0, 3, 4.5],
                      [0, 0, 6, 12],
                      [1, 0, 3, 12],
                      [1, 0, 3, 6]]
    npt.assert_array_equal(true_answer, desired_answer)
