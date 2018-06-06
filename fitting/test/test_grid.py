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


import numpy as np
import numpy.testing as npt

from fitting.grid import BaseRadialGrid, UniformRadialGrid, ClenshawRadialGrid, CubicGrid


def test_input_for_radial_grid():
    r"""Input checks for general grid."""
    npt.assert_raises(TypeError, BaseRadialGrid, 5.)
    # Only one dimensional arrays are required.
    npt.assert_raises(TypeError, BaseRadialGrid, np.array([[5.]]))


def test_integration_general_grid():
    r"""Test normal integration on the radial grid."""
    grid = np.arange(0., 2., 0.0001)
    rad_obj = BaseRadialGrid(grid)

    # Assume no masked values.
    model = grid
    true_answer = rad_obj.integrate(model)
    desired_answer = 2. * 2. / 2.
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-3)


def test_input_checks_radial_grid():
    r"""Test input checks on radial _grid."""
    npt.assert_raises(TypeError, ClenshawRadialGrid, 10.1, 1, 1, [])
    npt.assert_raises(ValueError, ClenshawRadialGrid, -10, 1, 1, [])
    npt.assert_raises(TypeError, ClenshawRadialGrid, 1, 2.2, 1, [])
    npt.assert_raises(ValueError, ClenshawRadialGrid, 1, -2, 1, [])
    npt.assert_raises(TypeError, ClenshawRadialGrid, 1, 1, 1.1, [])
    npt.assert_raises(ValueError, ClenshawRadialGrid, 1, 1, -2, [])
    npt.assert_raises(TypeError, ClenshawRadialGrid, 1, 1, 1, "not list")
    cgrid = ClenshawRadialGrid(5, 10, 10)
    npt.assert_equal(cgrid.atomic_numb, 5)


def test_grid_is_clenshaw():
    r"""Test that radial _grid returns a clenshaw _grid."""
    core_pts = 10
    diff_pts = 20
    atomic_numb = 10
    fac = 1. / (2. * 10.)
    rad_obj = ClenshawRadialGrid(atomic_numb, core_pts, diff_pts, [1000])
    actual_pts = rad_obj.radii
    desired_pts = []
    for x in range(0, core_pts):
        desired_pts.append(fac * (1. - np.cos(np.pi * x / (2. * core_pts))))
    for x in range(1, diff_pts):
        desired_pts.append(25. * (1. - np.cos(np.pi * x / (2. * diff_pts))))
    desired_pts.append(1000)
    desired_pts = np.sort(desired_pts)
    npt.assert_allclose(actual_pts, desired_pts)


def test_grid_is_uniform():
    r"""Test that radial _grid returns a uniform _grid."""
    actual_pts = UniformRadialGrid(10, 100)
    npt.assert_allclose(actual_pts.radii, np.arange(0, 100 * 10.) / 10.)

    actual_pts = UniformRadialGrid(5, 33)
    npt.assert_allclose(actual_pts.radii, np.arange(0, 33 * 5.) / 5.)


def test_integration_on_grid():
    r"""Test that integrations works on radial _grid."""
    # Test exponential with wolfram
    numb_pts = 100
    rad_obj = ClenshawRadialGrid(10, numb_pts, numb_pts)
    arr = np.exp(-rad_obj.radii**2)
    actual_value = rad_obj.integrate_spher(False, arr)
    desired_val = 4. * np.pi * 0.443313
    assert np.abs(actual_value - desired_val) < 0.1

    # Test with singularities.
    numb_pts = 100
    rad_obj = ClenshawRadialGrid(10, numb_pts, numb_pts, filled=True)
    arr = np.exp(-rad_obj.radii ** 2)
    arr[np.random.randint(5)] = np.nan
    actual_value = rad_obj.integrate_spher(False, arr)
    desired_val = 4. * np.pi * 0.443313
    assert np.abs(actual_value - desired_val) < 0.1

    # Test with masked values
    arr = np.exp(-rad_obj.radii**2)
    arr[arr < 1e-10] = np.inf
    arr = np.ma.array(arr, mask=arr == np.inf)
    actual_value = rad_obj.integrate_spher(True, arr)
    desired_val = 4. * np.pi * 0.443313
    assert np.abs(actual_value - desired_val) < 0.1


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
    true_answer = cubic_obj.integrate_spher(False, integrand)
    desired_answer = 0.25**3 * 1.
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-2, atol=1e-1)
