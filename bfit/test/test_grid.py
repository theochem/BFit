# -*- coding: utf-8 -*-
# BFit - python program that fits a convex sum of
# positive basis functions to any probability distribution. .
#
# Copyright (C) 2020 The BFit Development Team.
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


import numpy as np

from numpy.testing import assert_raises, assert_almost_equal

from bfit.grid import _BaseRadialGrid, UniformRadialGrid, ClenshawRadialGrid, CubicGrid


def test_raises_base():
    # check array
    assert_raises(TypeError, _BaseRadialGrid, 5.)
    assert_raises(TypeError, _BaseRadialGrid, [1., 2., 3.])
    assert_raises(TypeError, _BaseRadialGrid, (1., 2., 3.))
    # check 1D array
    assert_raises(TypeError, _BaseRadialGrid, np.array([[5.]]))
    assert_raises(TypeError, _BaseRadialGrid, np.array([[5., 5.5], [1.3, 2.5]]))


def test_integration_base():
    # integrate a triangle
    grid = _BaseRadialGrid(np.arange(0., 2., 0.000001), spherical=False)
    value = grid.integrate(grid.points)
    assert_almost_equal(value, 2. * 2. / 2, decimal=5)
    # integrate a square
    value = grid.integrate(2. * np.ones(len(grid)))
    assert_almost_equal(value, 2. * 2., decimal=5)


def test_raises_integration():
    r"""Test that integration over BaseRadialGrid returns an assertion error if dimension aren't specified."""
    grid = _BaseRadialGrid(np.arange(0., 2., 0.000001), spherical=False)
    # Obtain an array with different dimension than the "grid.points".
    arr = np.arange(0., 2., 0.1)
    assert_raises(ValueError, grid.integrate, arr)


def test_raises_clenshaw():
    assert_raises(TypeError, ClenshawRadialGrid, 10.1, 1, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, -10, 1, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 2.2, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, -2, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 1, 1.1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 1, -2, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 1, 1, "not list")


def test_raises_uniform():
    assert_raises(TypeError, UniformRadialGrid, 10.1, 1, 1)
    assert_raises(TypeError, UniformRadialGrid, -10, 1, 1)
    assert_raises(TypeError, UniformRadialGrid, 10, "not numb", 1)
    assert_raises(TypeError, UniformRadialGrid, 10, 1, "not numb")
    assert_raises(ValueError, UniformRadialGrid, 1, 2.2, 1)


def test_points_clenshaw():
    grid = ClenshawRadialGrid(10, 10, 20, [1000])
    # check core points
    core = [(1. - np.cos(np.pi * x / 20)) / 20. for x in range(0, 10)]
    assert_almost_equal(core, grid._get_points(10, "core"), decimal=8)
    # check diffuse points
    diff = [25. * (1. - np.cos(np.pi * x / 40.)) for x in range(0, 20)]
    assert_almost_equal(diff, grid._get_points(20, "diffuse"), decimal=8)
    # check points
    assert_almost_equal(sorted(core + diff[1:] + [1000]), grid.points, decimal=8)
    # check assertion
    assert_raises(ValueError, grid._get_points, 10, "not core or diffuse")


def test_points_uniform():
    # check 1 point
    grid = UniformRadialGrid(1, 0, 1)
    assert_almost_equal(grid.points, np.array([0.]), decimal=8)
    # check 2 point
    grid = UniformRadialGrid(2, 0, 1)
    assert_almost_equal(grid.points, np.array([0, 1.]), decimal=8)
    # check 5 points
    grid = UniformRadialGrid(5, 0, 33)
    assert_almost_equal(grid.points, np.arange(0, 33.1, step=33/4.), decimal=8)
    # check 10 points
    grid = UniformRadialGrid(10, 0, 100)
    assert_almost_equal(grid.points, np.arange(0, 100.1, step=100/9.), decimal=8)


def test_integration_uniform_gaussian():
    grid = UniformRadialGrid(100, 0.0, 15.0)
    pnts = grid.points
    # integrate s-type gaussian functions
    value = grid.integrate(np.exp(-pnts**2))
    assert_almost_equal(value, np.pi**1.5, decimal=6)
    value = grid.integrate(1.23 * np.exp(-0.5 * pnts**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += grid.integrate(0.91 * np.exp(-0.1 * pnts**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5, decimal=6)
    # integrate p-type gaussian functions
    value = grid.integrate(grid.points**2 * np.exp(-2.5 * pnts**2))
    assert_almost_equal(value, 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = grid.integrate(1.2 * pnts**2 * np.exp(-pnts**2))
    assert_almost_equal(value, 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = grid.integrate(2.5 * pnts**2 * np.exp(-1.1 * pnts**2))
    assert_almost_equal(value, 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6)
    value = grid.integrate(2. * 2.75**2.5 * pnts**2 * np.exp(-2.75 * pnts**2) / (3. * np.pi**1.5))
    assert_almost_equal(value, 1.0, decimal=6)
    # integrate s-type + p-type gaussians
    value = grid.integrate(np.exp(-4.01 * pnts**2) + pnts**2 * np.exp(-1.25 * pnts**2))
    assert_almost_equal(value, (np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5), decimal=6)
    value = (0.5 / np.pi)**1.5 * np.exp(-0.5 * pnts**2) + 3.62 * pnts**2 * np.exp(-0.85 * pnts**2)
    value = grid.integrate(value)
    assert_almost_equal(value, 1.0 + 3.62 * 1.5 * (np.pi**1.5 / 0.85**2.5), decimal=6)


def test_integration_uniform_gaussian_on_x_axis():
    # integrating from -15 to 15, which is twice as much as integrating from 0 to 15
    grid = UniformRadialGrid(300, -15., 15., spherical=True)
    pnts = grid.points
    # integrate s-type gaussian functions
    value = grid.integrate(np.exp(-pnts**2))
    assert_almost_equal(value, 2 * np.pi**1.5, decimal=6)
    value = grid.integrate(1.23 * np.exp(-0.5 * pnts**2))
    assert_almost_equal(value, 2 * 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += grid.integrate(0.91 * np.exp(-0.1 * pnts**2))
    expected_value = 2 * (1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5)
    assert_almost_equal(value, expected_value, decimal=6)
    # integrate p-type gaussian functions
    value = grid.integrate(pnts**2 * np.exp(-2.5 * pnts**2))
    assert_almost_equal(value, 2 * 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = grid.integrate(1.2 * pnts**2 * np.exp(-pnts**2))
    assert_almost_equal(value, 2 * 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = grid.integrate(2.5 * pnts**2 * np.exp(-1.1 * pnts**2))
    assert_almost_equal(value, 2 * 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6)
    value = grid.integrate(2. * 2.75**2.5 * pnts**2 * np.exp(-2.75 * pnts**2) / (3. * np.pi**1.5))
    assert_almost_equal(value, 2 * 1.0, decimal=6)
    # integrate s-type + p-type gaussians
    value = grid.integrate(np.exp(-4.01 * pnts**2) + pnts**2 * np.exp(-1.25 * pnts**2))
    expected_value = 2 * ((np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5))
    assert_almost_equal(value, expected_value, decimal=6)
    value = (0.5 / np.pi)**1.5 * np.exp(-0.5 * pnts**2) + 3.62 * pnts**2 * np.exp(-0.85 * pnts**2)
    value = grid.integrate(value)
    assert_almost_equal(value, 2 * (1.0 + 3.62 * 1.5 * (np.pi**1.5 / 0.85**2.5)), decimal=6)


def test_integration_uniform_gaussian_shifted():
    # spherical=False is used but r^2 is added to the integrand
    grid = UniformRadialGrid(300, 0.0, 30.0, False)
    pnts = grid.points - 15.
    # integrate s-type gaussian functions
    value = 4 * np.pi * grid.integrate(pnts**2 * np.exp(-pnts**2))
    assert_almost_equal(value, 2 * np.pi**1.5, decimal=6)
    value = 4 * np.pi * grid.integrate(1.23 * pnts**2 * np.exp(-0.5 * pnts**2))
    assert_almost_equal(value, 2 * 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += 4 * np.pi * grid.integrate(0.91 * pnts**2 * np.exp(-0.1 * pnts**2))
    expected_value = 2 * (1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5)
    assert_almost_equal(value, expected_value, decimal=6)
    # integrate p-type gaussian functions
    value = 4 * np.pi * grid.integrate(pnts**4 * np.exp(-2.5 * pnts**2))
    assert_almost_equal(value, 2 * 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = 4 * np.pi * grid.integrate(1.2 * pnts**4 * np.exp(-pnts**2))
    assert_almost_equal(value, 2 * 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = 4 * np.pi * grid.integrate(2.5 * pnts**4 * np.exp(-1.1 * pnts**2))
    assert_almost_equal(value, 2 * 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6)
    value = grid.integrate(2. * 2.75**2.5 * pnts**4 * np.exp(-2.75 * pnts**2) / (3. * np.pi**1.5))
    assert_almost_equal(4 * np.pi * value, 2 * 1.0, decimal=6)
    # integrate s-type + p-type gaussians
    value = grid.integrate(pnts**2 * np.exp(-4.01 * pnts**2) + pnts**4 * np.exp(-1.25 * pnts**2))
    value *= 4 * np.pi
    expected_value = 2 * ((np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5))
    assert_almost_equal(value, expected_value, decimal=6)
    value = (0.5 / np.pi)**1.5 * pnts**2 * np.exp(-0.5 * pnts**2)
    value += 3.62 * pnts**4 * np.exp(-0.85 * pnts**2)
    value = 4 * np.pi * grid.integrate(value)
    assert_almost_equal(value, 2 * (1.0 + 3.62 * 1.5 * (np.pi**1.5 / 0.85**2.5)), decimal=6)


def test_integration_clenshaw_gaussian():
    # clenshaw grid & points
    grid = ClenshawRadialGrid(10, 10000, 10000, spherical=True)
    pnts = grid.points
    # integrate s-type gaussian functions
    value = grid.integrate(np.exp(-pnts**2))
    assert_almost_equal(value, np.pi**1.5, decimal=6)
    value = grid.integrate(1.23 * np.exp(-0.5 * pnts**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += grid.integrate(0.91 * np.exp(-0.1 * pnts**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5, decimal=6)
    # integrate p-type gaussian functions
    value = grid.integrate(grid.points**2 * np.exp(-2.5 * pnts**2))
    assert_almost_equal(value, 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = grid.integrate(1.2 * pnts**2 * np.exp(-pnts**2))
    assert_almost_equal(value, 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = grid.integrate(2.5 * pnts**2 * np.exp(-1.1 * pnts**2))
    assert_almost_equal(value, 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6)
    value = grid.integrate(2. * 2.75**2.5 * pnts**2 * np.exp(-2.75 * pnts**2) / (3. * np.pi**1.5))
    assert_almost_equal(value, 1.0, decimal=6)
    # integrate s-type + p-type gaussians
    value = grid.integrate(np.exp(-4.01 * pnts**2) + pnts**2 * np.exp(-1.25 * pnts**2))
    assert_almost_equal(value, (np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5), decimal=6)
    value = (0.5 / np.pi)**1.5 * np.exp(-0.5 * pnts**2) + 3.62 * pnts**2 * np.exp(-0.85 * pnts**2)
    value = grid.integrate(value)
    assert_almost_equal(value, 1.0 + 3.62 * 1.5 * (np.pi**1.5 / 0.85**2.5), decimal=6)


def test_raises_cubic():
    assert_raises(AttributeError, CubicGrid, "blah", 2., 3.)
    assert_raises(AttributeError, CubicGrid, -1., "blah", 3.)
    assert_raises(AttributeError, CubicGrid, -1., 2., "blah")
    assert_raises(AttributeError, CubicGrid, -1., 2., -5.)
    assert_raises(AttributeError, CubicGrid, 5., -5., 3.)


def test_integration_cubic():
    axes = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
    grid = CubicGrid(np.zeros(3), axes, (25, 25, 25))
    # integrate constant value of 1.
    value = grid.integrate(np.ones(len(grid)))
    assert_almost_equal(value, 0.25**3, decimal=3)
    # integrate constant value of 2.
    value = grid.integrate(2 * np.ones(len(grid)))
    assert_almost_equal(value, 2 * 0.25**3, decimal=3)
    # return error if arr is not the same length.
    assert_raises(ValueError, grid.integrate, np.arange(0., 0.25, 0.1))


def test_integration_cubic_gaussian():
    axes = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]])
    grid = CubicGrid(np.array([-15.0, -15.0, -15.0]), axes, (120, 120, 120))
    dist = np.linalg.norm(grid.points, axis=1)
    # integrate s-type gaussian functions
    value = grid.integrate(np.exp(-dist**2))
    assert_almost_equal(value, np.pi**1.5, decimal=6)
    value = grid.integrate(1.23 * np.exp(-0.5 * dist**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += grid.integrate(0.91 * np.exp(-0.1 * dist**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5, decimal=6)
    # integrate p-type gaussian functions
    value = grid.integrate(dist**2 * np.exp(-2.5 * dist**2))
    assert_almost_equal(value, 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = grid.integrate(1.2 * dist**2 * np.exp(-dist**2))
    assert_almost_equal(value, 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = grid.integrate(2.5 * dist**2 * np.exp(-1.1 * dist**2))
    assert_almost_equal(value, 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6)
    value = grid.integrate(2. * 2.75**2.5 * dist**2 * np.exp(-2.75 * dist**2) / (3. * np.pi**1.5))
    assert_almost_equal(value, 1.0, decimal=6)
    # integrate s-type + p-type gaussian functions
    value = grid.integrate(np.exp(-4.01 * dist**2) + dist**2 * np.exp(-1.25 * dist**2))
    assert_almost_equal(value, (np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5), decimal=6)
    value = (0.5 / np.pi)**1.5 * np.exp(-0.5 * dist**2) + 3.62 * dist**2 * np.exp(-0.85 * dist**2)
    value = grid.integrate(value)
    assert_almost_equal(value, 1.0 + 3.62 * 1.5 * (np.pi**1.5 / 0.85**2.5), decimal=6)


def test_integration_cubic_gaussian_shifted():
    axes = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]])
    grid = CubicGrid(np.zeros(3), axes, (120, 120, 120))
    # place the function at the center of cubic grid with coordinates of [15, 15, 15]
    dist = np.linalg.norm(grid.points - np.array([15., 15., 15.]), axis=1)
    # integrate s-type gaussian functions
    value = grid.integrate(np.exp(-dist**2))
    assert_almost_equal(value, np.pi**1.5, decimal=6)
    value = grid.integrate(1.23 * np.exp(-0.5 * dist**2))
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += grid.integrate(0.91 * np.exp(-0.1 * dist**2))
    expected_value = 1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5
    assert_almost_equal(value, expected_value, decimal=6)
    # integrate p-type gaussian functions
    value = grid.integrate(dist**2 * np.exp(-2.5 * dist**2))
    assert_almost_equal(value, 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = grid.integrate(1.2 * dist**2 * np.exp(-dist**2))
    assert_almost_equal(value, 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = grid.integrate(2.5 * dist**2 * np.exp(-1.1 * dist**2))
    assert_almost_equal(value, 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6)
    value = grid.integrate(2. * 2.75**2.5 * dist**2 * np.exp(-2.75 * dist**2) / (3. * np.pi**1.5))
    assert_almost_equal(value, 1.0, decimal=6)
    # integrate s-type + p-type gaussian functions
    value = grid.integrate(np.exp(-4.01 * dist**2) + dist**2 * np.exp(-1.25 * dist**2))
    expected_value = (np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5)
    assert_almost_equal(value, expected_value, decimal=6)
    value = (0.5 / np.pi)**1.5 * np.exp(-0.5 * dist**2) + 3.62 * dist**2 * np.exp(-0.85 * dist**2)
    value = grid.integrate(value)
    assert_almost_equal(value, 1.0 + 3.62 * 1.5 * (np.pi**1.5 / 0.85**2.5), decimal=6)
