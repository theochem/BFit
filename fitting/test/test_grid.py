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

from numpy.testing import assert_raises, assert_almost_equal

from fitting.grid import BaseRadialGrid, UniformRadialGrid, ClenshawRadialGrid, CubicGrid


def test_raises_base():
    # check array
    assert_raises(TypeError, BaseRadialGrid, 5.)
    assert_raises(TypeError, BaseRadialGrid, [1., 2., 3.])
    assert_raises(TypeError, BaseRadialGrid, (1., 2., 3.))
    # check 1D array
    assert_raises(TypeError, BaseRadialGrid, np.array([[5.]]))
    assert_raises(TypeError, BaseRadialGrid, np.array([[5., 5.5], [1.3, 2.5]]))


def test_integration_base():
    # integrate a triangle
    grid = BaseRadialGrid(np.arange(0., 2., 0.000001), spherical=False)
    value = grid.integrate(grid.points)
    assert_almost_equal(value, 2. * 2. / 2, decimal=5)
    # integrate a square
    value = grid.integrate(2. * np.ones(len(grid)))
    assert_almost_equal(value, 2. * 2., decimal=5)


def test_raises_clenshaw():
    assert_raises(TypeError, ClenshawRadialGrid, 10.1, 1, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, -10, 1, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 2.2, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, -2, 1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 1, 1.1, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 1, -2, [])
    assert_raises(TypeError, ClenshawRadialGrid, 1, 1, 1, "not list")


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


def test_integration_clenshaw():
    # test against wolfram
    grid = ClenshawRadialGrid(10, 1000, 1000, spherical=True)
    value = grid.integrate(np.exp(-grid.points ** 2))
    assert_almost_equal(value, 4. * np.pi * 0.443313, decimal=2)


def test_raises_cubic():
    assert_raises(TypeError, CubicGrid, "blah", 2., 3.)
    assert_raises(TypeError, CubicGrid, -1., "blah", 3.)
    assert_raises(TypeError, CubicGrid, -1., 2., "blah")
    assert_raises(TypeError, CubicGrid, -1., 2., -5.)
    assert_raises(ValueError, CubicGrid, 5., -5., 3.)


def test_points_cubic():
    grid = CubicGrid(0., 1., 0.5)
    desired_answer = [[0.0, 0.0, 0.], [0.0, 0.0, 0.5], [0.0, 0.0, 1.],
                      [0.0, 0.5, 0.], [0.0, 0.5, 0.5], [0.0, 0.5, 1.],
                      [0.0, 1.0, 0.], [0.0, 1.0, 0.5], [0.0, 1.0, 1.],
                      [0.5, 0.0, 0.], [0.5, 0.0, 0.5], [0.5, 0.0, 1.],
                      [0.5, 0.5, 0.], [0.5, 0.5, 0.5], [0.5, 0.5, 1.],
                      [0.5, 1.0, 0.], [0.5, 1.0, 0.5], [0.5, 1.0, 1.],
                      [1.0, 0.0, 0.], [1.0, 0.0, 0.5], [1.0, 0.0, 1.],
                      [1.0, 0.5, 0.], [1.0, 0.5, 0.5], [1.0, 0.5, 1.],
                      [1.0, 1.0, 0.], [1.0, 1.0, 0.5], [1.0, 1.0, 1.]]
    assert_almost_equal(grid.points, desired_answer, decimal=8)


def test_integration_cubic():
    grid = CubicGrid(0., 0.25, 0.001, spherical=False)
    # integrate constant value of 1.
    value = grid.integrate(np.ones(len(grid)))
    assert_almost_equal(value, 0.25**3, decimal=3)
    # integrate constant value of 2.
    value = grid.integrate(2 * np.ones(len(grid)))
    assert_almost_equal(value, 2 * 0.25**3, decimal=3)
