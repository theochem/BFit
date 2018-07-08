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

from fitting.model import GaussianModel


def test_raises_gaussian_model():
    # check points
    assert_raises(TypeError, GaussianModel, 10., False)
    assert_raises(TypeError, GaussianModel, [5.], True)
    assert_raises(TypeError, GaussianModel, (2, 1), True)
    assert_raises(TypeError, GaussianModel, np.array([[1., 2.]]), True)
    assert_raises(TypeError, GaussianModel, np.array([[1.], [2.]]), True)


def test_gaussian_s_one_basis_origin():
    # test one (un)normalized s-type gaussian on r=0. against sympy

    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[1., 0.]])
    model = GaussianModel(np.array([0.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([0.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    coeffs, expons = np.array([3.]), np.array([2.])
    # un-normalized at r=0.
    g, dg = np.array([3.]), np.array([[1., 0.]])
    model = GaussianModel(np.array([0.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([1.52384726242178]), np.array([[0.507949087473928, 1.14288544681634]])
    model = GaussianModel(np.array([0.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_s_one_basis_one_point():
    # test one (un)normalized s-type gaussian on one point against sympy

    coeffs, expons = np.array([3.]), np.array([0.])
    # un-normalized at r=1.
    g, dg = np.array([3.]), np.array([[1., -3.]])
    model = GaussianModel(np.array([1.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([1.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    coeffs, expons = np.array([3.15]), np.array([1.75])
    # un-normalized at r=1.5
    g, dg = np.array([0.0614152227]), np.array([[0.0194968961, -0.1381842511]])
    model = GaussianModel(np.array([1.5]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.5
    g, dg = np.array([0.0255333792]), np.array([[0.0081058346, -0.0355643496]])
    model = GaussianModel(np.array([1.5]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_s_one_basis_multiple_point():
    # test one (un)normalized s-type gaussian on multiple point against sympy

    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized r=[0., 0.5, 1.0]
    g, dg = np.zeros(3), np.array([[1., 0.], [1., 0.], [1., 0.]])
    model = GaussianModel(np.array([0., 0.5, 1.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 0.5, 1.0]
    g, dg = np.zeros(3), np.zeros((3, 2))
    model = GaussianModel(np.array([0., 0.5, 1.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    coeffs, expons = np.array([5.25]), np.array([0.8])
    # un-normalized r=[0., 1.5, 2.2, 3.1, 10.]
    g = np.array([5.25, 0.8678191632, 0.1092876455, 0.0024060427, 0.0])
    dg = np.array([[1.0, 0.0], [0.1652988882, -1.9525931171], [0.0208166944, -0.5289522041],
                   [0.0004582938, -0.0231220701], [0.0, 0.0]])
    model = GaussianModel(np.array([0., 1.5, 2.2, 3.1, 10.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 1.5, 2.2, 3.1, 10.]
    g = np.array([0.6746359418, 0.1115165711, 0.0140436902, 0.0003091815, 0.0])
    dg = np.array([[0.1285020841, 1.2649423908], [0.0212412516, -0.0418187142],
                   [0.0026749886, -0.0416395415], [5.88917e-05, -0.0023915189], [0.0, 0.0]])
    model = GaussianModel(np.array([0., 1.5, 2.2, 3.1, 10.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_s_multiple_basis_origin():
    # test multiple (un)normalized s-type gaussian on r=0. against sympy

    coeffs, expons = np.zeros(10), np.zeros(10)
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[1.] * 10 + [0.] * 10])
    model = GaussianModel(np.array([0.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.zeros(20)[None, :]
    model = GaussianModel(np.array([0.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=0.
    g, dg = np.array([4.5]), np.array([[1., 1., 1., 1., 0., 0., 0., 0.]])
    model = GaussianModel(np.array([0.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([0.2222277257])
    dg = np.array([[0., 0., 0.5079490874, 0.0634936359, 0., 0., 0., 0.6666831773]])
    model = GaussianModel(np.array([0.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_s_multiple_basis_one_point():
    # test multiple (un)normalized s-type gaussian on one point against sympy

    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=1.
    g = np.array([3.12285730899422])
    dg = np.array([[1, 1, 0.1353352832, 0.606530659712633, 0., -1., 0., -2.1228573089]])
    model = GaussianModel(np.array([1.]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    # normalized at r=1.
    g = np.array([0.1347879291])
    dg = np.array([[0., 0., 0.0687434336, 0.0385108368, 0., 0., 0., 0.2695758582]])
    model = GaussianModel(np.array([1.]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    # un-normalized at r=2.5
    g = np.array([1.1537792676])
    dg = np.array([[1., 1., 3.7266531720e-6, 0.0439369336, 0., -6.25, 0., -0.9611204230]])
    model = GaussianModel(np.array([2.5]), normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    # normalized at r=2.5
    g = np.array([0.0097640048])
    dg = np.array([[0., 0., 1.8929500780e-6, 0.0027897156, 0., 0., 0., -0.0317330157]])
    model = GaussianModel(np.array([2.5]), normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_s_multiple_basis_multiple_point():
    # test multiple (un)normalized s-type gaussian on multiple points against sympy
    points = np.array([0.0, 2.67, 0.43, 5.1])
    coeffs = np.array([2.0, 0.02, 1.51])
    expons = np.array([6.1, 0.19, 7.67])

    # un-normalized
    g = np.array([3.53, 0.0051615725, 1.0323926829, 0.0001428204])
    dg = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.258078623, 0.0, -0.0, -0.0367963339, -0.0],
          [0.3237155762, 0.9654789302, 0.2421536105, -0.1197100201, -0.0035703411, -0.0676090459],
          [0.0, 0.0071410175, 0.0, -0.0, -0.0037147573, -0.0]]
    dg = np.array(dg)
    model = GaussianModel(points, normalized=False, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)

    # normalized
    g = np.array([11.1718777104, 7.67693e-05, 3.1468802527, 2.1242e-06])
    dg = [[2.7056395801, 0.0148732402, 3.8147689308, 1.3306424164, 0.0023484064, 1.126525636],
          [0.0, 0.0038384654, 0.0, -0.0, 5.87928e-05, -0.0],
          [0.8758576756, 0.0143598001, 0.9237600699, 0.1068575081, 0.0022142343, 0.0148793624],
          [0.0, 0.0001062101, 0.0, -0.0, -3.84805e-05, -0.0]]
    dg = np.array(dg)
    model = GaussianModel(points, normalized=True, basis="s")
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
