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
    assert_raises(TypeError, GaussianModel, 10., 5, 5, False)
    assert_raises(TypeError, GaussianModel, [5.], 5, 5, True)
    assert_raises(TypeError, GaussianModel, (2, 1), 5, 5, True)
    assert_raises(TypeError, GaussianModel, np.array([[1., 2.]]), 5, 5, True)
    assert_raises(TypeError, GaussianModel, np.array([[1.], [2.]]), 5, 5, True)
    # check num_s & num_p
    assert_raises(TypeError, GaussianModel, np.array([0.]), 5, -1, False)
    assert_raises(TypeError, GaussianModel, np.array([0.]), -5, 1, False)
    assert_raises(TypeError, GaussianModel, np.array([0.]), 5, 1.0, False)
    assert_raises(TypeError, GaussianModel, np.array([0.]), 5.1, 1, False)
    assert_raises(ValueError, GaussianModel, np.array([0.]), 0, 0, False)
    assert_raises(ValueError, GaussianModel, np.array([0.]), 0, 0, True)


def test_gaussian_1s_origin():
    # test one (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[1., 0.]])
    model = GaussianModel(np.array([0.]), 1, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([0.]), 1, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.array([3.]), np.array([2.])
    # un-normalized at r=0.
    g, dg = np.array([3.]), np.array([[1., 0.]])
    model = GaussianModel(np.array([0.]), 1, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([1.52384726242178]), np.array([[0.507949087473928, 1.14288544681634]])
    model = GaussianModel(np.array([0.]), 1, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s_one_point():
    # test one (un)normalized s-type gaussian on one point against sympy
    coeffs, expons = np.array([3.]), np.array([0.])
    # un-normalized at r=1.
    g, dg = np.array([3.]), np.array([[1., -3.]])
    model = GaussianModel(np.array([1.]), 1, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([1.]), 1, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized s-type gaussian on one point against sympy
    coeffs, expons = np.array([3.15]), np.array([1.75])
    # un-normalized at r=-1.5
    g, dg = np.array([0.0614152227]), np.array([[0.0194968961, -0.1381842511]])
    model = GaussianModel(np.array([-1.5]), 1, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-1.5
    g, dg = np.array([0.0255333792]), np.array([[0.0081058346, -0.0355643496]])
    model = GaussianModel(np.array([-1.5]), 1, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s_multiple_point():
    # test one (un)normalized s-type gaussian on multiple point against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized r=[0., 0.5, 1.0]
    g, dg = np.zeros(3), np.array([[1., 0.], [1., 0.], [1., 0.]])
    model = GaussianModel(np.array([0., 0.5, 1.]), 1, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 0.5, 1.0]
    g, dg = np.zeros(3), np.zeros((3, 2))
    model = GaussianModel(np.array([0., 0.5, 1.]), 1, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized s-type gaussian on multiple point against sympy
    coeffs, expons = np.array([5.25]), np.array([0.8])
    # un-normalized r=[0., 1.5, -2.2, 3.1, 10.]
    g = np.array([5.25, 0.8678191632, 0.1092876455, 0.0024060427, 0.0])
    dg = np.array([[1.0, 0.0], [0.1652988882, -1.9525931171], [0.0208166944, -0.5289522041],
                   [0.0004582938, -0.0231220701], [0.0, 0.0]])
    model = GaussianModel(np.array([0., 1.5, -2.2, 3.1, 10.]), 1, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 1.5, 2.2, 3.1, 10.]
    g = np.array([0.6746359418, 0.1115165711, 0.0140436902, 0.0003091815, 0.0])
    dg = np.array([[0.1285020841, 1.2649423908], [0.0212412516, -0.0418187142],
                   [0.0026749886, -0.0416395415], [5.88917e-05, -0.0023915189], [0.0, 0.0]])
    model = GaussianModel(np.array([0., 1.5, -2.2, 3.1, 10.]), 1, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xs_origin():
    # test multiple (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.zeros(10), np.zeros(10)
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[1.] * 10 + [0.] * 10])
    model = GaussianModel(np.array([0.]), 10, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.zeros(20)[None, :]
    model = GaussianModel(np.array([0.]), 10, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test multiple (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=0.
    g, dg = np.array([4.5]), np.array([[1., 1., 1., 1., 0., 0., 0., 0.]])
    model = GaussianModel(np.array([0.]), 4, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([0.2222277257])
    dg = np.array([[0., 0., 0.5079490874, 0.0634936359, 0., 0., 0., 0.6666831773]])
    model = GaussianModel(np.array([0.]), 4, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xs_one_point():
    # test multiple (un)normalized s-type gaussian on one point against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=-1.
    g = np.array([3.12285730899422])
    dg = np.array([[1, 1, 0.1353352832, 0.606530659712633, 0., -1., 0., -2.1228573089]])
    model = GaussianModel(np.array([-1.]), 4, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-1.
    g = np.array([0.1347879291])
    dg = np.array([[0., 0., 0.0687434336, 0.0385108368, 0., 0., 0., 0.2695758582]])
    model = GaussianModel(np.array([-1.]), 4, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # un-normalized at r=2.5
    g = np.array([1.1537792676])
    dg = np.array([[1., 1., 3.7266531720e-6, 0.0439369336, 0., -6.25, 0., -0.9611204230]])
    model = GaussianModel(np.array([2.5]), 4, 0, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=2.5
    g = np.array([0.0097640048])
    dg = np.array([[0., 0., 1.8929500780e-6, 0.0027897156, 0., 0., 0., -0.0317330157]])
    model = GaussianModel(np.array([2.5]), 4, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xs_multiple_point():
    # test multiple (un)normalized s-type gaussian on multiple points against sympy
    points = np.array([0.0, -2.67, 0.43, 5.1])
    coeffs = np.array([2.0, 0.02, 1.51])
    expons = np.array([6.1, 0.19, 7.67])
    # un-normalized
    g = np.array([3.53, 0.0051615725, 1.0323926829, 0.0001428204])
    dg = [[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.258078623, 0.0, -0.0, -0.0367963339, -0.0],
          [0.3237155762, 0.9654789302, 0.2421536105, -0.1197100201, -0.0035703411, -0.0676090459],
          [0.0, 0.0071410175, 0.0, -0.0, -0.0037147573, -0.0]]
    dg = np.array(dg)
    model = GaussianModel(points, 3, 0, normalized=False)
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
    model = GaussianModel(points, 3, 0, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1p_origin():
    # test one (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([0.]), 0, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([0.]), 0, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.array([3.]), np.array([2.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([0.]), 0, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    model = GaussianModel(np.array([0.]), 0, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1p_one_point():
    # test one (un)normalized p-type gaussian on one point against sympy
    coeffs, expons = np.array([3.]), np.array([0.])
    # un-normalized at r=-1.
    g, dg = np.array([3.]), np.array([[1., -3.]])
    model = GaussianModel(np.array([-1.]), 0, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-1.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = GaussianModel(np.array([-1.]), 0, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized p-type gaussian on one point against sympy
    coeffs, expons = np.array([3.15]), np.array([1.75])
    # un-normalized at r=1.5
    g, dg = np.array([0.1381842511]), np.array([[0.0438680162, -0.3109145651]])
    model = GaussianModel(np.array([1.5]), 0, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.5
    g, dg = np.array([0.0670251204]), np.array([[0.0212778160, -0.0550563489]])
    model = GaussianModel(np.array([1.5]), 0, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1p_multiple_point():
    # test one (un)normalized p-type gaussian on multiple point against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized r=[0., -0.5, 1.0]
    g, dg = np.zeros(3), np.array([[0., 0.], [0.25, 0.], [1., 0.]])
    model = GaussianModel(np.array([0., -0.5, 1.]), 0, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., -0.5, 1.0]
    g, dg = np.zeros(3), np.zeros((3, 2))
    model = GaussianModel(np.array([0., -0.5, 1.]), 0, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized p-type gaussian on multiple point against sympy
    coeffs, expons = np.array([5.25]), np.array([0.8])
    # un-normalized r=[0., 1.5, 2.2, -3.1, 10.]
    g = np.array([0.0, 1.9525931171, 0.5289522041, 0.0231220701, 0.0])
    dg = np.array([[0.0, 0.0], [0.3719224985, -4.3933345135],
                   [0.1007528008, -2.5601286677], [0.0044042038, -0.222203094], [0.0, 0.0]])
    model = GaussianModel(np.array([0., 1.5, 2.2, -3.1, 10.]), 0, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 1.5, 2.2, -3.1, 10.]
    g = np.array([0.0, 0.1338198854, 0.0362514457, 0.0015846582, 0.0])
    dg = np.array([[0.0, 0.0], [0.025489502, 0.1170923997], [0.0069050373, -0.0621712293],
                   [0.0003018397, -0.0102765087], [0.0, 0.0]])
    model = GaussianModel(np.array([0., 1.5, 2.2, -3.1, 10.]), 0, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xp_origin():
    # test multiple (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.zeros(10), np.zeros(10)
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[0.] * 20])
    model = GaussianModel(np.array([0.]), 0, 10, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.zeros(20)[None, :]
    model = GaussianModel(np.array([0.]), 0, 10, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test multiple (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=0.
    g, dg = np.array([0]), np.array([[0.] * 8])
    model = GaussianModel(np.array([0.]), 0, 4, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    model = GaussianModel(np.array([0.]), 0, 4, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xp_one_point():
    # test multiple (un)normalized p-type gaussian on one point against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=1.
    g = np.array([3.12285730899422])
    dg = np.array([[1, 1, 0.1353352832, 0.606530659712633, 0., -1., 0., -2.1228573089]])
    model = GaussianModel(np.array([1.]), 0, 4, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.
    g = np.array([0.0449293097])
    dg = np.array([[0., 0., 0.0916579114, 0.0128369456, 0., 0., 0., 0.1797172388]])
    model = GaussianModel(np.array([1.]), 0, 4, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # un-normalized at r=-2.5
    g = np.array([7.2111204230])
    dg = np.array([[6.25, 6.25, 2.3291582325e-5, 0.2746058351, 0., -39.0625, 0., -6.0070026438]])
    model = GaussianModel(np.array([-2.5]), 0, 4, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-2.5
    g = np.array([0.0203416767])
    dg = np.array([[0., 0., 1.5774583984e-5, 0.0058119076, 0., 0., 0., -0.0254270959]])
    model = GaussianModel(np.array([-2.5]), 0, 4, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xp_multiple_point():
    # test multiple (un)normalized p-type gaussian on multiple points against sympy
    points = np.array([0.0, 2.67, -0.43, 5.1])
    coeffs = np.array([2.0, 0.02, 1.51])
    expons = np.array([6.1, 0.19, 7.67])
    # un-normalized
    g = np.array([0.0, 0.0367963339, 0.1908894071, 0.0037147573])
    dg = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.8398166958, 0.0, 0.0, -0.2623173849, 0.0],
          [0.05985501, 0.1785170542, 0.0447742026, -0.0221343827, -0.0006601561, -0.0125009126],
          [0.0, 0.1857378662, 0.0, 0.0, -0.096620838, 0.0]]
    dg = np.array(dg)
    model = GaussianModel(points, 0, 3, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized
    g = np.array([0.0, 6.93222e-05, 2.6359627773, 6.9984e-06])
    dg = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0034661112, 0.0, 0.0, 0.0004179433, 0.0],
          [0.6585807425, 0.0003363161, 0.8733738848, 0.2962771222, 8.72605e-05, 0.1860096977],
          [0.0, 0.0003499197, 0.0, 0.0, -8.99441e-05, 0.0]]
    dg = np.array(dg)
    model = GaussianModel(points, 0, 3, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s1p_basis_origin():
    # test (un)normalized s-type + p-type gaussian at origin against sympy
    coeffs = np.array([0.5, -2.0])
    expons = np.array([0.1, 0.4])
    # un-normalized at r=0.
    g, dg = np.array([0.5]), np.array([[1., 0., 0., 0.]])
    model = GaussianModel(np.array([0.]), 1, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.0028395217]), np.array([[0.0056790434, 0.04259282582, 0., 0.]])
    model = GaussianModel(np.array([0.]), 1, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s1p_basis_one_point():
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([0.5, 2.0])
    expons = np.array([0.1, 0.4])
    # un-normalized at r=-3.
    g, = np.array([0.6951118339])
    dg = np.array([[0.4065696597, -1.8295634688, 0.2459135020, -4.4264430364]])
    model = GaussianModel(np.array([-3.]), 1, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-3.
    g = np.array([0.0071130914])
    dg = np.array([[0.0023089267, 0.0069267802, 0.0029793140, -0.0163862272]])
    model = GaussianModel(np.array([-3.]), 1, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_2s1p_basis_origin():
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=0.
    g, dg = np.array([-3.3]), np.array([[1., 1., 0., 0., 0., 0.]])
    model = GaussianModel(np.array([0.]), 2, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([-0.9752103479])
    dg = np.array([[0.1285020841, 0.2509806330, 0.2891296893, -1.3552954187, 0., 0.]])
    model = GaussianModel(np.array([0.]), 2, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_2s1p_basis_multiple_point():
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=[0.89, -2.56]
    g = np.array([-1.1399500497, 0.0050932753])
    dg = np.array([[0.530635465, 0.3715302468, -0.5043796222,
                    1.3243009883, 0.1397686627, 0.0830330683],
                   [0.0052850141, 0.0002768596, -0.0415630417,
                    0.0081649222, 3.831100e-06, 1.8830400e-05]])
    model = GaussianModel(np.array([0.89, -2.56]), 2, 1, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=[0.89, -2.56]
    g = np.array([-0.4268626593, 0.000499832])
    dg = np.array([[0.0681877632, 0.0932468966, 0.0886086345,
                    -0.171159341, 0.1187692541, -0.0311281606],
                   [0.0006791353, 6.94864e-05, -0.003812883,
                    0.0016740108, 3.25550e-06, 1.321400e-05]])
    model = GaussianModel(np.array([0.89, -2.56]), 2, 1, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s2p_basis_origin():
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=0.
    g, dg = np.array([1.2]), np.array([[1., 0., 0., 0., 0., 0.]])
    model = GaussianModel(np.array([0.]), 1, 2, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([0.1542025009])
    dg = np.array([[0.1285020841, 0.2891296893, 0., 0., 0., 0.]])
    model = GaussianModel(np.array([0.]), 1, 2, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s2p_basis_multiple_point():
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=[-0.201, 1.701]
    g = np.array([0.9612490198, -0.2351426929])
    dg = np.array([[0.9681959350, -0.0469393008, 0.0384113615,
                    0.0369799675,  0.0069833584, 0.0011205208],
                   [0.0987937634, -0.3430199685, 0.0777451878,
                    0.0051224852,  1.0122660185, 0.0111160529]])
    model = GaussianModel(np.array([-0.201, 1.701]), 1, 2, normalized=False)
    print("m = ", model.evaluate(coeffs, expons, deriv=False))
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=[-0.201, 1.701]
    g = np.array([0.0895783695, -0.0612024131])
    dg = np.array([[0.1244151955,  0.27390239190,  0.0080337565,
                    0.0314239478, -0.07084323550, -0.0259518957],
                   [0.0126952045, -0.01551457070,  0.0162604470,
                    0.0043528623,  0.06537194840,  0.0057191665]])
    model = GaussianModel(np.array([-0.201, 1.701]), 1, 2, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_2s2p_basis_multiple_point():
    # test multiple (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([0.53, -6.1, -2.0, 4.3])
    expons = np.array([0.11, 3.4, 1.5, 0.78])
    points = np.array([0., -0.71, 1.68, -2.03, 3.12, 4.56])
    # un-normalized
    g = np.array([-5.57, 0.3920707885, 1.6490462639, 1.0318279643, 0.2027417225, 0.0538246312])
    dg = [[1., 1., 0., 0., 0., 0., 0., 0.],
          [0.9460583794, 0.1801545834, -0.2527612554,  0.5539771455,
           0.2366600353, 0.3402146225,  0.2386006476, -0.7374594222],
          [0.7331067158, 6.798930e-05, -1.0966338091,  0.0011705476,
           0.0409250713, 0.3122708330,  0.2310138426, -3.7898187562],
          [0.6355280823, 8.224000e-07, -1.3880422674,  2.06726e-05,
           0.0085205157, 0.1655916578,  0.0702243861, -2.9342626493],
          [0.3427397041, 0.0000000000, -1.7682736492,  0.0, 4.4352e-06,
           0.0049066395, 8.634840e-05, -0.2053817226],
          [0.1015406569, 0.0000000000, -1.1190397759, 0.0, 0.0, 1.8798e-06, 0.0, -0.0001680746]]
    dg = np.array(dg)
    model = GaussianModel(points, 2, 2, normalized=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized
    g = np.array([-6.8644186384, -1.2960445644, 0.0614559599,
                  0.0423855159, 0.0025445224, 0.0003531182])
    dg = [[0.0065518541, 1.1258837903, 0.0473520366, -3.0299519652, 0.0, 0.0, 0.0, 0.0],
          [0.0061984365, 0.2028331252, 0.0431417361, 0.0778541543,
           0.0780794966, 0.0218863537, -0.1815452401, 0.2541973327],
          [0.0048032083, 7.65481e-05, 0.0275291113, 0.0011118962,
           0.0135021064, 0.0200887013, 0.0312096688, 0.0330606041],
          [0.0041638873, 9.259e-07, 0.0209992985, 2.07832e-05,
           0.0028111108, 0.0106526803, 0.0137982435, -0.0419483238],
          [0.0022455805, 0.0, 0.004643952, 0.0,
           1.4633e-06, 0.0003156491, 2.36107e-05, -0.0088621233],
          [0.0006652796, 0.0, -0.0025236285, 0.0, 0.0, 1.209e-07, 0.0, -9.1458e-06]]
    dg = np.array(dg)
    model = GaussianModel(points, 2, 2, normalized=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
