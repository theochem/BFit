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
r"""Test file for 'bfit.greedy'."""

from bfit.greedy import (
    get_next_choices, get_two_next_choices, GreedyKLFPI,
    GreedyLeastSquares, pick_two_lose_one, remove_redundancies
)
from bfit.grid import UniformRadialGrid
import numpy as np
import numpy.testing as npt


def test_check_redundancies():
    r"""Testing for checking redundancies for greedy."""
    c = np.array([50., 30., 10., 2., 3.])
    exps = np.array([3.1, 3., 2, 1, 3.025])
    simi = 1.
    true_answer = remove_redundancies(c, exps, simi)
    desired_answer_coeff = np.array([83., 10., 2.])
    desired_answer_exps = np.array([3.1, 2., 1.])
    npt.assert_allclose(true_answer[0], desired_answer_coeff)
    npt.assert_allclose(true_answer[1], desired_answer_exps)

    c = np.array([50., 30., 10., 2., 3.])
    exps = np.array([3.0012, 3.0005, 2, 1, 3.025])
    simi = 1e-2
    true_answer = remove_redundancies(c, exps, simi)
    npt.assert_array_equal(true_answer[0], np.array([80, 10., 2., 3]))
    npt.assert_array_equal(true_answer[1], np.array([3.0012, 2., 1., 3.025]))

    simi = 1e-1
    true_answer = remove_redundancies(c, exps, simi)
    npt.assert_array_equal(true_answer[0], np.array([83, 10., 2]))
    npt.assert_array_equal(true_answer[1], np.array([3.0012, 2, 1]))


def test_get_next_possible_coeffs_and_exps():
    r"""Testing 'bfit.greedy.greedy_utils.get_next_choices'."""
    c = np.array([1., 2.])
    e = np.array([3., 4.])
    factor = 2
    true_answer = get_next_choices(factor, c, e, coeff_val=0)
    desired_answer = [[0., 1, 2, 1.5, 3, 4],
                      [1, 0, 2, 3, 3.5, 4],
                      [1, 2, 0, 3, 4, 8]]
    npt.assert_array_equal(true_answer, desired_answer)


def test_get_two_next_possible_coeffs_and_exps():
    r"""Testing 'bfit.greedy.greedy_utils.get_two_next_choices'."""
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
    r"""Testing 'bfit.greedy.greedy_utils.pick_two_lose_one'."""
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


def test_greedy_kl_two_function():
    r"""Test Greedy Kullback-Leibler against two-function Gaussian combination."""
    def eval_density(points):
        return 0.25 * np.exp(-10. * points**2.0) * (10.0 / np.pi)**1.5 + \
                0.75 * np.exp(-5. * points**2.0) * (5.0 / np.pi)**1.5

    grid = UniformRadialGrid(1000, 0.0, 10.)
    density = eval_density(grid.points)
    greedy = GreedyKLFPI(grid, density, "pick-one",
                         g_eps_coeff=1e-10, g_eps_exp=1e-10,
                         l_eps_exp=1e-5, g_eps_obj=1e-15,
                         l_eps_obj=1e-10, l_eps_coeff=1e-8, mask_value=0.0,
                         l_maxiter=15000, g_maxiter=15000, integral_dens=1.0,
                         spherical=True)
    result = greedy.run(2.5, max_numb_funcs=2, disp=True)

    npt.assert_almost_equal(result["fun"], 0.0, decimal=6)
    npt.assert_almost_equal(np.sort(result["coeffs"]), [0.25, 0.75], decimal=3)
    npt.assert_almost_equal(np.sort(result["exps"]), [5.0, 10.0], decimal=3)
    assert result["success"]


def test_greedy_ls_two_function():
    r"""Test Greedy Least-Squares against two-function Gaussian combination."""
    def eval_density(points):
        return 0.25 * np.exp(-10. * points**2.0) * (10.0 / np.pi)**1.5 + \
                0.75 * np.exp(-5. * points**2.0) * (5.0 / np.pi)**1.5

    grid = UniformRadialGrid(1000, 0.0, 10.)
    density = eval_density(grid.points)
    greedy = GreedyLeastSquares(
        grid, density, "pick-one", local_tol=1e-10, global_tol=1e-15,
        integral_dens=1.0, normalize=True, spherical=True,
    )
    result = greedy.run(2.5, d_threshold=1e-10, max_numb_funcs=2, disp=True)

    npt.assert_almost_equal(result["fun"], 0.0, decimal=10)
    npt.assert_almost_equal(np.sort(result["coeffs"]), [0.25, 0.75], decimal=3)
    npt.assert_almost_equal(np.sort(result["exps"]), [5.0, 10.0], decimal=3)
    assert result["success"]
