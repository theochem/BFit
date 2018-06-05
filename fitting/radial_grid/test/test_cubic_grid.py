# -*- coding: utf-8 -*-
# An basis-set curve-fitting optimization package.
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
r"""Test file for 'fitting.radial_grid.cubic_grid'."""


import numpy as np
import numpy.testing as npt
from fitting.radial_grid.cubic_grid import CubicGrid

__all__ = ["test_input_checks",
           "test_making_cubic_grid",
           "test_integration_cubic_grid"]


def test_input_checks():
    r"""Test for input checks for cubic grid."""
    true_a = -1.
    true_b = 2.
    true_c = 3.
    stri = "not right"
    npt.assert_raises(TypeError, CubicGrid, stri, true_b, true_c)
    npt.assert_raises(TypeError, CubicGrid, true_a, stri, true_c)
    npt.assert_raises(TypeError, CubicGrid, true_a, true_b, stri)
    npt.assert_raises(ValueError, CubicGrid, true_a, true_b, -5.)
    npt.assert_raises(ValueError, CubicGrid, 5., -5., true_c)


def test_making_cubic_grid():
    r"""Test for making a uniform cubic grid."""
    smallest_pt = 0.
    largest_pt = 1.
    step_size = 0.5
    true_answer = CubicGrid.make_cubic_grid(smallest_pt, largest_pt, step_size)
    desired_answer = [[0., 0., 0.],
                      [0., 0., 0.5],
                      [0., 0., 1.],
                      [0., 0.5, 0.],
                      [0., 0.5, 0.5],
                      [0., 0.5, 1.],
                      [0., 1., 0.],
                      [0., 1., 0.5],
                      [0., 1., 1.],
                      [0.5, 0., 0.],
                      [0.5, 0., 0.5],
                      [0.5, 0., 1.],
                      [0.5, 0.5, 0.],
                      [0.5, 0.5, 0.5],
                      [0.5, 0.5, 1.],
                      [0.5, 1., 0.],
                      [0.5, 1., 0.5],
                      [0.5, 1., 1.],
                      [1., 0., 0.],
                      [1., 0., 0.5],
                      [1., 0., 1.],
                      [1., 0.5, 0.],
                      [1., 0.5, 0.5],
                      [1., 0.5, 1.],
                      [1., 1., 0.],
                      [1., 1., 0.5],
                      [1., 1., 1.]]
    npt.assert_array_equal(true_answer, desired_answer)


def test_integration_cubic_grid():
    r"""Test for integration on cubic grid."""
    smallest_pt = 0.
    largest_pt = 0.25
    step_size = 0.01
    cubic_obj = CubicGrid(smallest_pt, largest_pt, step_size)
    integrand = np.array([1.] * cubic_obj.grid.shape[0])
    true_answer = cubic_obj.integrate_spher(integrand)
    desired_answer = 0.25**3 * 1.
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-2, atol=1e-1)
