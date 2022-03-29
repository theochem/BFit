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
r"""Test bfit.model module."""

from bfit.grid import CubicGrid, UniformRadialGrid
from bfit.model import AtomicGaussianDensity, MolecularGaussianDensity
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises


def test_raises_gaussian_model():
    r"""Test raises error of all Gaussian model classes."""
    # check points
    assert_raises(TypeError, AtomicGaussianDensity, 10., None, 5, 5, False)
    assert_raises(TypeError, AtomicGaussianDensity, [5.], None, 5, 5, True)
    assert_raises(TypeError, AtomicGaussianDensity, (2, 1), None, 5, 5, True)
    # check num_s & num_p
    assert_raises(TypeError, AtomicGaussianDensity, np.array([0.]), None, 5, -1, False)
    assert_raises(TypeError, AtomicGaussianDensity, np.array([0.]), None, -5, 1, False)
    assert_raises(TypeError, AtomicGaussianDensity, np.array([0.]), None, 5, 1.0, False)
    assert_raises(TypeError, AtomicGaussianDensity, np.array([0.]), None, 5.1, 1, False)
    assert_raises(ValueError, AtomicGaussianDensity, np.array([0.]), None, 0, 0, False)
    assert_raises(ValueError, AtomicGaussianDensity, np.array([0.]), None, 0, 0, True)
    assert_raises(ValueError, AtomicGaussianDensity, np.array([0.]), [1.], 1, 0, True)
    assert_raises(ValueError, AtomicGaussianDensity, np.array([0.]), np.array([[1.]]), 1, 0, True)
    assert_raises(ValueError, AtomicGaussianDensity, np.array([[0.]]), np.array([1., 2.]), 1)
    # test on molecular density.
    assert_raises(ValueError, MolecularGaussianDensity, np.array([0.]), np.array([1.]), [1, 1])
    assert_raises(ValueError, MolecularGaussianDensity,
                  np.array([0.]), np.array([[1.]]), np.array([1, 1]))
    assert_raises(ValueError, MolecularGaussianDensity,
                  np.array([0.]), np.array([[1.], [2.]]), np.array([[1, 1]]))
    assert_raises(ValueError, MolecularGaussianDensity,
                  np.array([[0., 1.]]), np.array([[1.]]), np.array([[1, 1]]))


def test_gaussian_1s_origin():
    r"""Test evaluation of single s-type Gaussian at the origin."""
    # test one (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[1., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.array([3.]), np.array([2.])
    # un-normalized at r=0.
    g, dg = np.array([3.]), np.array([[1., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([1.52384726242178]), np.array([[0.507949087473928, 1.14288544681634]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s_one_point():
    r"""Test evaluation of single s-type Gaussian at various single points."""
    # test one (un)normalized s-type gaussian on one point against sympy
    coeffs, expons = np.array([3.]), np.array([0.])
    # un-normalized at r=1.
    g, dg = np.array([3.]), np.array([[1., -3.]])
    model = AtomicGaussianDensity(np.array([1.]), num_s=1, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = AtomicGaussianDensity(np.array([1.]), num_s=1, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized s-type gaussian on one point against sympy
    coeffs, expons = np.array([3.15]), np.array([1.75])
    # un-normalized at r=-1.5
    g, dg = np.array([0.0614152227]), np.array([[0.0194968961, -0.1381842511]])
    model = AtomicGaussianDensity(np.array([-1.5]), num_s=1, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-1.5
    g, dg = np.array([0.0255333792]), np.array([[0.0081058346, -0.0355643496]])
    model = AtomicGaussianDensity(np.array([-1.5]), num_s=1, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s_multiple_point():
    r"""Test evaluation of single s-type Gaussian at multiple points."""
    # test one (un)normalized s-type gaussian on multiple point against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized r=[0., 0.5, 1.0]
    g, dg = np.zeros(3), np.array([[1., 0.], [1., 0.], [1., 0.]])
    model = AtomicGaussianDensity(np.array([0., 0.5, 1.]), num_s=1, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 0.5, 1.0]
    g, dg = np.zeros(3), np.zeros((3, 2))
    model = AtomicGaussianDensity(np.array([0., 0.5, 1.]), num_s=1, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized s-type gaussian on multiple point against sympy
    coeffs, expons = np.array([5.25]), np.array([0.8])
    # un-normalized r=[0., 1.5, -2.2, 3.1, 10.]
    g = np.array([5.25, 0.8678191632, 0.1092876455, 0.0024060427, 0.0])
    dg = np.array([[1.0, 0.0], [0.1652988882, -1.9525931171], [0.0208166944, -0.5289522041],
                   [0.0004582938, -0.0231220701], [0.0, 0.0]])
    model = AtomicGaussianDensity(np.array([0., 1.5, -2.2, 3.1, 10.]), None, 1, 0, False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 1.5, 2.2, 3.1, 10.]
    g = np.array([0.6746359418, 0.1115165711, 0.0140436902, 0.0003091815, 0.0])
    dg = np.array([[0.1285020841, 1.2649423908], [0.0212412516, -0.0418187142],
                   [0.0026749886, -0.0416395415], [5.88917e-05, -0.0023915189], [0.0, 0.0]])
    model = AtomicGaussianDensity(np.array([0., 1.5, -2.2, 3.1, 10.]), None, 1, 0, True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xs_origin():
    r"""Test evaluation of multiple s-type Gaussian at origin."""
    # test multiple (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.zeros(10), np.zeros(10)
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[1.] * 10 + [0.] * 10])
    model = AtomicGaussianDensity(np.array([0.]), num_s=10, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.zeros(20)[None, :]
    model = AtomicGaussianDensity(np.array([0.]), num_s=10, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test multiple (un)normalized s-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=0.
    g, dg = np.array([4.5]), np.array([[1., 1., 1., 1., 0., 0., 0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=4, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([0.2222277257])
    dg = np.array([[0., 0., 0.5079490874, 0.0634936359, 0., 0., 0., 0.6666831773]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=4, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xs_one_point():
    r"""Test evaluation of multiple s-type Gaussian at single point."""
    # test multiple (un)normalized s-type gaussian on one point against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=-1.
    g = np.array([3.12285730899422])
    dg = np.array([[1, 1, 0.1353352832, 0.606530659712633, 0., -1., 0., -2.1228573089]])
    model = AtomicGaussianDensity(np.array([-1.]), num_s=4, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-1.
    g = np.array([0.1347879291])
    dg = np.array([[0., 0., 0.0687434336, 0.0385108368, 0., 0., 0., 0.2695758582]])
    model = AtomicGaussianDensity(np.array([-1.]), num_s=4, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # un-normalized at r=2.5
    g = np.array([1.1537792676])
    dg = np.array([[1., 1., 3.7266531720e-6, 0.0439369336, 0., -6.25, 0., -0.9611204230]])
    model = AtomicGaussianDensity(np.array([2.5]), num_s=4, num_p=0, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=2.5
    g = np.array([0.0097640048])
    dg = np.array([[0., 0., 1.8929500780e-6, 0.0027897156, 0., 0., 0., -0.0317330157]])
    model = AtomicGaussianDensity(np.array([2.5]), num_s=4, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xs_multiple_point():
    r"""Test evaluation of multiple s-type Gaussian at multiple points."""
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
    model = AtomicGaussianDensity(points, num_s=3, num_p=0, normalize=False)
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
    model = AtomicGaussianDensity(points, num_s=3, num_p=0, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1p_origin():
    r"""Test evaluation of single p-type Gaussian at origin."""
    # test one (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.array([3.]), np.array([2.])
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1p_one_point():
    r"""Test evaluation of single p-type Gaussian at single point."""
    # test one (un)normalized p-type gaussian on one point against sympy
    coeffs, expons = np.array([3.]), np.array([0.])
    # un-normalized at r=-1.
    g, dg = np.array([3.]), np.array([[1., -3.]])
    model = AtomicGaussianDensity(np.array([-1.]), num_s=0, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-1.
    g, dg = np.array([0.]), np.array([[0., 0.]])
    model = AtomicGaussianDensity(np.array([-1.]), num_s=0, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized p-type gaussian on one point against sympy
    coeffs, expons = np.array([3.15]), np.array([1.75])
    # un-normalized at r=1.5
    g, dg = np.array([0.1381842511]), np.array([[0.0438680162, -0.3109145651]])
    model = AtomicGaussianDensity(np.array([1.5]), num_s=0, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.5
    g, dg = np.array([0.0670251204]), np.array([[0.0212778160, -0.0550563489]])
    model = AtomicGaussianDensity(np.array([1.5]), num_s=0, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1p_multiple_point():
    r"""Test evaluation of single p-type Gaussian at multiple points."""
    # test one (un)normalized p-type gaussian on multiple point against sympy
    coeffs, expons = np.array([0.]), np.array([0.])
    # un-normalized r=[0., -0.5, 1.0]
    g, dg = np.zeros(3), np.array([[0., 0.], [0.25, 0.], [1., 0.]])
    model = AtomicGaussianDensity(np.array([0., -0.5, 1.]), num_s=0, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., -0.5, 1.0]
    g, dg = np.zeros(3), np.zeros((3, 2))
    model = AtomicGaussianDensity(np.array([0., -0.5, 1.]), num_s=0, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test one (un)normalized p-type gaussian on multiple point against sympy
    coeffs, expons = np.array([5.25]), np.array([0.8])
    # un-normalized r=[0., 1.5, 2.2, -3.1, 10.]
    g = np.array([0.0, 1.9525931171, 0.5289522041, 0.0231220701, 0.0])
    dg = np.array([[0.0, 0.0], [0.3719224985, -4.3933345135],
                   [0.1007528008, -2.5601286677], [0.0044042038, -0.222203094], [0.0, 0.0]])
    model = AtomicGaussianDensity(np.array([0., 1.5, 2.2, -3.1, 10.]), None, 0, 1, False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized r=[0., 1.5, 2.2, -3.1, 10.]
    g = np.array([0.0, 0.1338198854, 0.0362514457, 0.0015846582, 0.0])
    dg = np.array([[0.0, 0.0], [0.025489502, 0.1170923997], [0.0069050373, -0.0621712293],
                   [0.0003018397, -0.0102765087], [0.0, 0.0]])
    model = AtomicGaussianDensity(np.array([0., 1.5, 2.2, -3.1, 10.]), None, 0, 1, True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xp_origin():
    r"""Test evaluation of multiple p-type Gaussian at origin."""
    # test multiple (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.zeros(10), np.zeros(10)
    # un-normalized at r=0.
    g, dg = np.array([0.]), np.array([[0.] * 20])
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=10, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.]), np.zeros(20)[None, :]
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=10, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # test multiple (un)normalized p-type gaussian on r=0. against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=0.
    g, dg = np.array([0]), np.array([[0.] * 8])
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=4, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    model = AtomicGaussianDensity(np.array([0.]), num_s=0, num_p=4, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xp_one_point():
    r"""Test evaluation of multiple p-type Gaussian at single point."""
    # test multiple (un)normalized p-type gaussian on one point against sympy
    coeffs, expons = np.array([0.0, 1.0, 0.0, 3.5]), np.array([0.0, 0.0, 2.0, 0.5])
    # un-normalized at r=1.
    g = np.array([3.12285730899422])
    dg = np.array([[1, 1, 0.1353352832, 0.606530659712633, 0., -1., 0., -2.1228573089]])
    model = AtomicGaussianDensity(np.array([1.]), num_s=0, num_p=4, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=1.
    g = np.array([0.0449293097])
    dg = np.array([[0., 0., 0.0916579114, 0.0128369456, 0., 0., 0., 0.1797172388]])
    model = AtomicGaussianDensity(np.array([1.]), num_s=0, num_p=4, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # un-normalized at r=-2.5
    g = np.array([7.2111204230])
    dg = np.array([[6.25, 6.25, 2.3291582325e-5, 0.2746058351, 0., -39.0625, 0., -6.0070026438]])
    model = AtomicGaussianDensity(np.array([-2.5]), num_s=0, num_p=4, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-2.5
    g = np.array([0.0203416767])
    dg = np.array([[0., 0., 1.5774583984e-5, 0.0058119076, 0., 0., 0., -0.0254270959]])
    model = AtomicGaussianDensity(np.array([-2.5]), num_s=0, num_p=4, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_xp_multiple_point():
    r"""Test evaluation of multiple p-type Gaussian at multiple points."""
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
    model = AtomicGaussianDensity(points, num_s=0, num_p=3, normalize=False)
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
    model = AtomicGaussianDensity(points, num_s=0, num_p=3, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s1p_basis_origin():
    r"""Test evaluation of single s-type and p-type Gaussian at origin."""
    # test (un)normalized s-type + p-type gaussian at origin against sympy
    coeffs = np.array([0.5, -2.0])
    expons = np.array([0.1, 0.4])
    # un-normalized at r=0.
    g, dg = np.array([0.5]), np.array([[1., 0., 0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g, dg = np.array([0.0028395217]), np.array([[0.0056790434, 0., 0.04259282582, 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s1p_basis_one_point():
    r"""Test evaluation of single s-type and p-type Gaussian at single point."""
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([0.5, 2.0])
    expons = np.array([0.1, 0.4])
    # un-normalized at r=-3.
    g, = np.array([0.6951118339])
    dg = np.array([[0.4065696597, 0.2459135020, -1.8295634688, -4.4264430364]])
    model = AtomicGaussianDensity(np.array([-3.]), num_s=1, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=-3.
    g = np.array([0.0071130914])
    dg = np.array([[0.0023089267, 0.0029793140, 0.0069267802, -0.0163862272]])
    model = AtomicGaussianDensity(np.array([-3.]), num_s=1, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_2s1p_basis_origin():
    r"""Test evaluation of two s-type and one p-type Gaussian at origin."""
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=0.
    g, dg = np.array([-3.3]), np.array([[1., 1., 0., 0., 0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=2, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([-0.9752103479])
    dg = np.array([[0.1285020841, 0.2509806330, 0., 0.2891296893, -1.3552954187, 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=2, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_2s1p_basis_multiple_point():
    r"""Test evaluation of two s-type and one p-type Gaussian at multiple points."""
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=[0.89, -2.56]
    g = np.array([-1.1399500497, 0.0050932753])
    dg = np.array([[5.30635465e-01,  3.71530247e-01,  1.39768663e-01,
                    -5.04379622e-01,  1.32430099e+00,  8.30330683e-02],
                   [5.28501406e-03,  2.76859611e-04,  3.83105827e-06,
                    -4.15630417e-02,  8.16492216e-03,  1.88304176e-05]])
    model = AtomicGaussianDensity(np.array([0.89, -2.56]), num_s=2, num_p=1, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=[0.89, -2.56]
    g = np.array([-0.4268626593, 0.000499832])
    dg = np.array([[6.81877632e-02,  9.32468966e-02,  1.18769254e-01,
                    8.86086345e-02, -1.71159341e-01, -3.11281606e-02],
                   [6.79135321e-04,  6.94864004e-05,  3.25546459e-06,
                    -3.81288302e-03,  1.67401077e-03,  1.32140467e-05]])
    model = AtomicGaussianDensity(np.array([0.89, -2.56]), num_s=2, num_p=1, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s2p_basis_origin():
    r"""Test evaluation of one s-type and two p-type Gaussian at origin."""
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=0.
    g, dg = np.array([1.2]), np.array([[1., 0., 0., 0., 0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=2, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=0.
    g = np.array([0.1542025009])
    dg = np.array([[0.1285020841, 0., 0., 0.2891296893, 0., 0.]])
    model = AtomicGaussianDensity(np.array([0.]), num_s=1, num_p=2, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_1s2p_basis_multiple_point():
    r"""Test evaluation of one s-type and two p-type Gaussian at multiple points."""
    # test (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([1.2, -4.5, -0.75])
    expons = np.array([0.8, 1.25, 2.19])
    # un-normalized at r=[-0.201, 1.701]
    g = np.array([0.9612490198, -0.2351426929])
    dg = np.array([[0.96819593, 0.03841136, 0.03697997, -0.04693930, 0.00698336, 0.00112052],
                   [0.09879376, 0.07774519, 0.00512249, -0.34301997, 1.01226602, 0.01111605]])
    model = AtomicGaussianDensity(np.array([-0.201, 1.701]), num_s=1, num_p=2, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized at r=[-0.201, 1.701]
    g = np.array([0.0895783695, -0.0612024131])
    dg = np.array([[0.1244152, 0.00803376, 0.03142395,  0.27390239, -0.07084324, -0.0259519],
                   [0.0126952, 0.01626045, 0.00435286, -0.01551457,  0.06537195,  0.00571917]])
    model = AtomicGaussianDensity(np.array([-0.201, 1.701]), num_s=1, num_p=2, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_2s2p_basis_multiple_point():
    r"""Test evaluation of two s-type and two p-type Gaussian at origin."""
    # test multiple (un)normalized s-type & p-type gaussian on one point against sympy
    coeffs = np.array([0.53, -6.1, -2.0, 4.3])
    expons = np.array([0.11, 3.4, 1.5, 0.78])
    points = np.array([0., -0.71, 1.68, -2.03, 3.12, 4.56])
    # un-normalized
    g = np.array([-5.57, 0.3920707885, 1.6490462639, 1.0318279643, 0.2027417225, 0.0538246312])
    dg = np.array([[1.00000000e+00, 1.00000000e+00, 0.00000000e+00,  0.00000000e+00,
                    -0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -0.00000000e+00],
                   [9.46058379e-01, 1.80154583e-01, 2.36660035e-01,  3.40214622e-01,
                    -2.52761255e-01, 5.53977145e-01, 2.38600648e-01, -7.37459422e-01],
                   [7.33106716e-01, 6.79893148e-05, 4.09250713e-02,  3.12270833e-01,
                    -1.09663381e+00, 1.17054756e-03, 2.31013843e-01, -3.78981876e+00],
                   [6.35528082e-01, 8.22382682e-07, 8.52051568e-03,  1.65591658e-01,
                    -1.38804227e+00, 2.06726365e-05, 7.02243861e-02, -2.93426265e+00],
                   [3.42739704e-01, 4.22836923e-15, 4.43521708e-06,  4.90663947e-03,
                    -1.76827365e+00, 2.51079889e-13, 8.63483543e-05, -2.05381723e-01],
                   [1.01540657e-01, 1.97762748e-31, 5.91712954e-13,  1.87976695e-06,
                    -1.11903978e+00, 2.50844168e-29, 2.46076850e-11, -1.68074625e-04]])
    dg = np.array(dg)
    model = AtomicGaussianDensity(points, num_s=2, num_p=2, normalize=False)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized
    g = np.array([-6.8644186384, -1.2960445644, 0.0614559599,
                  0.0423855159, 0.0025445224, 0.0003531182])
    dg = np.array([[6.55185411e-03,  1.12588379e+00,  0.00000000e+00,  0.00000000e+00,
                    4.73520366e-02, -3.02995197e+00,  0.00000000e+00,  0.00000000e+00],
                   [6.19843649e-03,  2.02833125e-01,  7.80794966e-02,  2.18863537e-02,
                    4.31417361e-02,  7.78541543e-02, -1.81545240e-01,  2.54197333e-01],
                   [4.80320825e-03,  7.65480675e-05,  1.35021064e-02,  2.00887013e-02,
                    2.75291113e-02,  1.11189616e-03,  3.12096688e-02,  3.30606041e-02],
                   [4.16388728e-03,  9.25907331e-07,  2.81111077e-03,  1.06526803e-02,
                    2.09992985e-02,  2.07832063e-05,  1.37982435e-02, -4.19483238e-02],
                   [2.24558054e-03,  4.76065238e-15,  1.46327840e-06,  3.15649123e-04,
                    4.64395201e-03,  2.69875021e-13,  2.36106798e-05, -8.86212334e-03],
                   [6.65279571e-04,  2.22657872e-31,  1.95219482e-13,  1.20927325e-07,
                    -2.52362846e-03,  2.76429266e-29,  7.46790004e-12, -9.14578547e-06]])
    dg = np.array(dg)
    model = AtomicGaussianDensity(points, num_s=2, num_p=2, normalize=True)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_molecular_gaussian_density_1d_1center_1s():
    r"""Test evaluation of one s-type Molecular Gaussian at origin."""
    # points in 1D space
    points = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # gaussian parameters
    coeffs = np.array([1.5])
    expons = np.array([0.5])
    coords = np.array([[2.5]])
    # un-normalized 1s basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[1, 0]]), normalize=False)
    # check basis & density
    assert_equal(model.nbasis, 1)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_raises(ValueError, model.assign_basis_to_center, 1)
    g = np.array([0.0659054004, 0.486978701, 1.3237453539, 1.3237453539, 0.486978701])
    dg = np.array([[0.0439369336, -0.4119087527], [0.3246524674, -1.0957020773],
                   [0.8824969026, -0.3309363385], [0.8824969026, -0.3309363385],
                   [0.3246524674, -1.0957020773]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized 1s basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[1, 0]]), normalize=True)
    # check basis & density
    assert_equal(model.nbasis, 1)
    assert_equal(model.assign_basis_to_center(0), 0)
    g = np.array([0.0041845735, 0.0309200484, 0.0840494056, 0.0840494056, 0.0309200484])
    dg = np.array([[0.0027897157, -0.0135998639], [0.0206133656, 0.0231900363],
                   [0.0560329370,  0.2311358653], [0.0560329370, 0.2311358653],
                   [0.0206133656,  0.0231900363]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_molecular_gaussian_density_1d_1center_2s():
    r"""Test evaluation of two s-type Molecular Gaussian at origin."""
    # points in 1D space
    points = np.array([-3.09, -1.53, 0.0, 0.79, 2.91, 4.09])
    # gaussian parameters
    coeffs = np.array([6.03, 2.56])
    expons = np.array([2.45, 0.36])
    coords = np.array([[1.45]])
    # un-normalized 1s basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[2, 0]]), normalize=False)
    # check basis & density
    assert_equal(model.nbasis, 2)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    g = np.array(
        [0.0015335519, 0.1046706295, 1.2358743034, 4.2625443763, 1.2209551847, 0.2082434127]
    )
    dg = np.array([[1.17169574e-22, 5.99043705e-04, -1.45627659e-20, -3.16089580e-02],
                   [3.55683438e-10, 4.08869638e-02, -1.90464255e-08, -9.29517039e-01],
                   [5.79288407e-03, 4.69118442e-01, -7.34426186e-02, -2.52498310e+00],
                   [3.43963408e-01, 8.54861339e-01, -9.03477676e-01, -9.53286654e-01],
                   [5.39425921e-03, 4.64229610e-01, -6.93353697e-02, -2.53325270e+00],
                   [3.83880356e-08, 8.13449927e-02, -1.61332199e-06, -1.45137168e+00]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized 1s basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[2, 0]]), normalize=True)
    # check basis & density
    assert_equal(model.nbasis, 2)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    g = np.array([5.94877e-05, 0.0040602608, 0.070642293, 1.5133048321, 0.0685013983, 0.0080780828])
    dg = np.array([[8.06936140e-23, 2.32373955e-05, -9.73133606e-21, -9.78271797e-04],
                   [2.44955932e-10, 1.58603879e-03, -1.22127618e-08, -1.91389796e-02],
                   [3.98950629e-03, 1.81974883e-02, -3.58506541e-02,  9.61603808e-02],
                   [2.36884454e-01, 3.31607709e-02,  2.52321914e-01,  3.16736121e-01],
                   [3.71497700e-03, 1.80078465e-02, -3.40355466e-02,  9.38167504e-02],
                   [2.64374891e-08, 3.15543884e-03, -1.01347721e-06, -2.26418808e-02]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_molecular_gaussian_density_1d_1center_1s2p():
    r"""Test evaluation of one s-type and two-p-type Molecular Gaussian at origin."""
    # points in 1D space
    points = np.array([-2.22, -0.76, 0.0, 1.98, 3.25, 5.50])
    # gaussian parameters
    coeffs = np.array([1.45, 0.0, 1.45])
    expons = np.array([0.75, 1.0, 0.75])
    coords = np.array([[1.00]])
    # un-normalized 1s + 2p basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[1, 2]]), normalize=False)
    # check basis & density
    assert_equal(model.nbasis, 3)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    assert_equal(model.assign_basis_to_center(2), 0)
    g = np.array([0.0069161303, 0.5820289984, 1.369863003, 1.3832172004, 0.1972685684, 7.8141e-06])
    dg = np.array([[0.0004195617, 0.0003256663, 0.0043501834, -0.0063077659, 0.0, -0.0654014398],
                   [0.0979596128, 0.1398797181, 0.3034396965, -0.4399875599, 0.0, -1.3629054655],
                   [0.4723665527, 0.3678794412, 0.4723665527, -0.6849315015, 0.0, -0.6849315015],
                   [0.4866062522, 0.3675832650, 0.4673366446, -0.6776381347, 0.0, -0.6508036645],
                   [0.0224407899, 0.0320441844, 0.1136064987, -0.1647294231, 0.0, -0.8339427044],
                   [2.536e-07, 3.25e-08, 5.1354e-06, -7.4464e-06, 0.0, -0.0001507893]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized 1s + 2p basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[1, 2]]), normalize=True)
    # check basis & density
    assert_equal(model.nbasis, 3)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    assert_equal(model.assign_basis_to_center(2), 0)
    g = np.array([0.0004388483, 0.0422296913, 0.119841017, 0.1218240891, 0.01340299, 4.772e-07])
    dg = np.array([[4.89399e-05, 3.89903e-05, 0.0002537141, -0.0005938453, 0.0, -0.0025880989],
                   [0.0114265243, 0.016747064, 0.0176974008, -0.0181855419, 0.0, 0.0060492075],
                   [0.0550993182, 0.0440442734, 0.0275496591, 0.0798940113, 0.0, 0.0932096799],
                   [0.0567603116, 0.0440088138, 0.0272563016, 0.0855616289, 0.0, 0.0937822106],
                   [0.0026176117, 0.0038364819, 0.0066258296, -0.011623832, 0.0, -0.0166128875],
                   [2.96e-08, 3.9e-09, 2.995e-07, -7.828e-07, 0.0, -7.3468e-06]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_molecular_gaussian_density_2d_1center_2s1p():
    r"""Test evaluation of two s-type and single p-type Molecular Gaussian."""
    # points in 2D space
    points = np.array([[0.0, 0.0], [0.25, -0.25], [-0.5, -0.5], [0.25, 0.5]])
    # gaussian parameters
    coeffs = np.array([1.45, 0.0, 1.45])
    expons = np.array([0.75, 1.0, 0.75])
    coords = np.array([[0.5, 0.5]])
    # un-normalized 2s + 1p basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[2, 1]]), normalize=False)
    g = np.array([1.4948541813, 1.4745035726, 0.9706161966, 1.4700746448])
    dg = np.array([[0.6872892788, 0.6065306597, 0.3436446394, -0.4982847271, 0.0, -0.2491423636],
                   [0.6257840096, 0.5352614285, 0.391115006, -0.5671167587, 0.0, -0.3544479742],
                   [0.2231301601, 0.1353352832, 0.4462603203, -0.6470774644, 0.0, -1.2941549289],
                   [0.954206666, 0.9394130628, 0.0596379166, -0.0864749791, 0.0, -0.0054046862]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized 2s + 1p basis function
    model = MolecularGaussianDensity(points, coords, basis=np.array([[2, 1]]), normalize=True)
    g = np.array([0.1453063757, 0.1389181087, 0.0754785174, 0.1664337873])
    dg = np.array([[0.0801690349, 0.1089250957, 0.0200422587, 0.1743676509, 0.0, 0.0823402796],
                   [0.0729947369, 0.0961260595, 0.0228108553, 0.1455332567, 0.0, 0.0895801296],
                   [0.026027075, 0.024304474, 0.026027075, 0.0, 0.0, 0.0503190116],
                   [0.1113036822, 0.1687064884, 0.0034782401, 0.3126937823, 0.0, 0.0164962782]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_molecular_gaussian_density_2d_2center_1s1p_2p():
    r"""Test evaluation of Molecular Gaussian of multiple centers."""
    # points in 2D space
    points = np.array([[0.0, 0.0], [0.25, -0.25], [-0.5, -0.5], [0.25, 0.5]])
    # gaussian parameters
    coeffs = np.array([1.45, 0.45, 2.91, 0.89])
    expons = np.array([0.75, 0.65, 1.23, 0.00])
    basis = np.array([[1, 1], [0, 2]])
    coords = np.array([[0.0, 0.0], [0.5, 0.5]])
    # un-normalized (1s + 1p) & (2p) basis functions
    model = MolecularGaussianDensity(points, coords, basis=basis, normalize=False)
    # check basis & density
    assert_equal(model.nbasis, 4)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    assert_equal(model.assign_basis_to_center(2), 1)
    assert_equal(model.assign_basis_to_center(3), 1)
    g = np.array([2.6816325027, 2.7715087598, 3.4363695234, 1.4858614587])
    dg = [[1.0, 0.0, 0.2703204, 0.5, 0.0, 0.0, -0.3933163, -0.2225],
          [0.9105104, 0.1152454, 0.2897451, 0.625, -0.16503, -0.0064826, -0.5269739, -0.3476563],
          [0.6872893, 0.3612637, 0.1708699, 2.0, -0.4982847, -0.0812843, -0.9944628, -3.56],
          [0.7910651, 0.2550551, 0.0578753, 0.0625, -0.3584514, -0.0358671, -0.0105261, -0.0034766]]
    dg = np.array(dg)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=7)
    assert_raises(ValueError, model.evaluate, np.array([[1.]]), np.array([1.]))
    assert_raises(ValueError, model.evaluate, np.array([1., 2.]), np.array([1.]))
    # normalized (1s + 1p) & (2p) basis functions
    model = MolecularGaussianDensity(points, coords, basis=basis, normalize=True)
    # check basis & density
    assert_equal(model.nbasis, 4)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    assert_equal(model.assign_basis_to_center(2), 1)
    assert_equal(model.assign_basis_to_center(3), 1)
    g = np.array([0.3271580029, 0.3254922487, 0.2227611062, 0.1723104634])
    dg = np.array([[0.1166453, 0.0, 0.0543032, 0.0, 0.3382712, 0.0, 0.2421725, 0.0],
                   [0.1062067, 0.0046999, 0.0582053, 0.0, 0.2887495, 0.0078701, 0.2384023, 0.0],
                   [0.080169, 0.014733, 0.0343251, 0.0, 0.1743677, 0.0221845, 0.0032483, 0.0],
                   [0.092274, 0.0104016, 0.0116263, 0.0, 0.2257829, 0.0165401, 0.0666506, 0.0]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=7)


def test_molecular_gaussian_density_3d_2center_1p_1p():
    r"""Test evaluation of Molecular Gaussian of multiple centers with p-type Gaussians."""
    # points in 3D space
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    # gaussian parameters
    coeffs = np.array([0.6, 1.6])
    expons = np.array([2.5, 0.7])
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, -0.5]])
    basis = np.array([[0, 1], [0, 1]])
    # un-normalized (1p) & (1p) basis functions
    model = MolecularGaussianDensity(points, coords, basis=basis, normalize=False)
    # check basis & density
    assert_equal(model.nbasis, 2)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 1)
    g = np.array([0.56375047, 0.84793613, 0.74434677])
    dg = np.array([[0.,  0.35234404,  0., -0.28187524],
                   [0.01347589,  0.52490662, -0.01617107, -1.2597759],
                   [0.082085,  0.43443486, -0.049251, -1.73773943]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # un-normalized (1p) & (1p) basis functions
    model = MolecularGaussianDensity(points, coords, basis=basis, normalize=True)
    # check basis & density
    assert_equal(model.nbasis, 2)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 1)
    g = np.array([0.02767043, 0.05078846, 0.09238776])
    dg = np.array([[0.00000000e+00,  1.72940204e-02,  0.00000000e+00, 8.49877575e-02],
                   [1.59437891e-02,  2.57638692e-02, -9.56627343e-03, 8.53888235e-02],
                   [9.71175569e-02,  2.13232647e-02, -1.38777878e-17, 3.65541680e-02]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_molecular_gaussian_density_3d_3center_2s1p_1p_2s():
    r"""Test evaluation of Molecular Gaussian with three centers."""
    # points in 3D space
    points = np.array([[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.0, 1.5, 0.5], [0.0, 0.0, -0.5]])
    # gaussian parameters
    coeffs = np.array([3.2, 1.7, 0.9, 4.1, 2.2, 7.1])
    expons = np.array([0.8, 2.5, 1.9, 1.1, 7.5, 3.1])
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, -0.5], [0.0, 1.0, 0.0]])
    basis = np.array([[2, 1], [0, 1], [2, 0]])
    # un-normalized (2s1p) & (1p) & (1s) basis functions
    model = MolecularGaussianDensity(points, coords, basis=basis, normalize=False)
    # check basis
    assert_equal(model.nbasis, 6)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    assert_equal(model.assign_basis_to_center(2), 0)
    assert_equal(model.assign_basis_to_center(3), 1)
    assert_equal(model.assign_basis_to_center(4), 2)
    assert_equal(model.assign_basis_to_center(5), 2)
    # compute density value & its derivatives
    g = np.array([4.177243712, 2.9091656819, 2.4681836067, 5.1131523036])
    dg = np.array([[5.48811636e-01,  1.53354967e-01,  1.80381347e-01,
                    3.16049495e-01,  3.60656314e-03,  9.77834441e-02,
                    -1.31714793e+00, -1.95527583e-01, -1.21757409e-01,
                    -1.61975366e+00, -5.95082917e-03, -5.20696840e-01],
                   [4.49328964e-01,  8.20849986e-02,  1.49568619e-01,
                    2.88474905e-01,  3.05902321e-07,  2.02943064e-03,
                    -1.43785269e+00, -1.39544498e-01, -1.34611757e-01,
                    -5.91373556e-01, -1.34597021e-06, -2.88179150e-02],
                   [1.35335283e-01,  1.93045414e-03,  2.16292380e-02,
                    1.10649502e-01,  2.35177459e-02,  2.12247974e-01,
                    -1.08268227e+00, -8.20443008e-03, -4.86657855e-02,
                    -1.36098888e+00, -2.58695204e-02, -7.53480307e-01],
                   [8.18730753e-01,  5.35261429e-01,  1.55471264e-01,
                    3.16049495e-01,  8.48182352e-05,  2.07543379e-02,
                    -6.54984602e-01, -2.27486107e-01, -3.49810344e-02,
                    -1.61975366e+00, -2.33250147e-04, -1.84194749e-01]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)
    # normalized (2s1p) & (1p) & (1s) basis functions
    model = MolecularGaussianDensity(points, coords, basis=basis, normalize=True)
    # check basis
    assert_equal(model.nbasis, 6)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 0)
    assert_equal(model.assign_basis_to_center(2), 0)
    assert_equal(model.assign_basis_to_center(3), 1)
    assert_equal(model.assign_basis_to_center(4), 2)
    assert_equal(model.assign_basis_to_center(5), 2)
    g = np.array([1.4141296252, 0.5578528034, 1.8064862846, 1.4079886645])
    dg = np.array([[7.05234390e-02,  1.08863690e-01,  1.07463149e-01,
                    4.80198440e-02,  1.33033380e-02,  9.58480014e-02,
                    2.53884381e-01, -2.77602409e-02,  5.47213665e-02,
                    2.01355937e-01, -1.60970390e-02, -1.81106344e-01],
                   [5.77397084e-02,  5.82705342e-02,  8.91063016e-02,
                    4.38302234e-02,  1.12836565e-06,  1.98926180e-03,
                    1.61671183e-01, -3.96239632e-02,  2.53249489e-02,
                    3.18566033e-01, -4.46832798e-06, -2.14134408e-02],
                   [1.73908660e-02,  1.37039161e-03,  1.28857338e-02,
                    1.68118346e-02,  8.67486608e-02,  2.08046917e-01,
                    -3.47817319e-02, -4.42636491e-03, -1.37334794e-02,
                    -5.01298341e-02, -5.72541161e-02, -2.38247276e-02],
                   [1.05208608e-01,  3.79971613e-01,  9.26228338e-02,
                    4.80198440e-02,  3.12864522e-04,  2.03435441e-02,
                    5.47084762e-01,  2.26083110e-01,  8.88447971e-02,
                    2.01355937e-01, -7.22717046e-04, -1.10659036e-01]])
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=False), decimal=8)
    assert_almost_equal(g, model.evaluate(coeffs, expons, deriv=True)[0], decimal=8)
    assert_almost_equal(dg, model.evaluate(coeffs, expons, deriv=True)[1], decimal=8)


def test_gaussian_model_s_integrate_uniform():
    r"""Test integration of one s-type Gaussian with uniform grid against analytic value."""
    grid = UniformRadialGrid(100, 0.0, 15.0)
    spherical = 4.0 * np.pi * grid.points**2.0
    # un-normalized s-type Gaussian at origin
    model = AtomicGaussianDensity(grid.points, num_s=1, num_p=0, normalize=False)
    # check center
    assert_almost_equal(model.coord, np.zeros(1), decimal=8)
    # check integration
    value = model.evaluate(np.ones(1), np.ones(1), deriv=False)
    assert_almost_equal(grid.integrate(value * spherical), np.pi**1.5, decimal=6)
    value = model.evaluate(np.array([1.23]), np.array([0.5]), deriv=False)
    assert_almost_equal(grid.integrate(value * spherical), 1.23 * (np.pi / 0.5)**1.5, decimal=6)
    value += model.evaluate(np.array([0.91]), np.array([0.1]), deriv=False)
    value = grid.integrate(value * spherical)
    assert_almost_equal(value, 1.23 * (np.pi / 0.5)**1.5 + 0.91 * (np.pi / 0.1)**1.5, decimal=6)


def test_gaussian_model_p_integrate_uniform():
    r"""Test integration of one p-type Gaussian with uniform grid against analytic value."""
    grid = UniformRadialGrid(100, 0.0, 15.0)
    spherical = 4.0 * np.pi * grid.points**2.0
    # un-normalized p-type Gaussian at origin
    model = AtomicGaussianDensity(grid.points, num_s=0, num_p=1, normalize=False)
    # check center
    assert_almost_equal(model.coord, np.zeros(1), decimal=8)
    # check integration
    value = model.evaluate(np.array([1.0]), np.array([2.5]), deriv=False)
    assert_almost_equal(grid.integrate(value * spherical), 1.5 * np.pi**1.5 / 2.5**2.5, decimal=6)
    value = model.evaluate(np.array([1.2]), np.array([1.0]), deriv=False)
    assert_almost_equal(grid.integrate(value * spherical), 1.2 * 1.5 * np.pi**1.5, decimal=6)
    value = model.evaluate(np.array([2.5]), np.array([1.1]), deriv=False)
    assert_almost_equal(
        grid.integrate(value * spherical), 2.5 * 1.5 * np.pi**1.5 / 1.1**2.5, decimal=6
    )
    # normalized p-type Gaussian at origin
    model = AtomicGaussianDensity(grid.points, num_s=0, num_p=1, normalize=True)
    # check center
    assert_almost_equal(model.coord, np.zeros(1), decimal=8)
    # check integration
    value = model.evaluate(np.array([2.]), np.array([2.75]), deriv=False)
    assert_almost_equal(grid.integrate(value * spherical), 2.0 * 1.0, decimal=6)


def test_gaussian_model_sp_integrate_uniform():
    r"""Test integration of one s/p-type Gaussian with uniform grid against analytic value."""
    grid = UniformRadialGrid(100, 0.0, 15.0)
    spherical = 4.0 * np.pi * grid.points**2.0
    # un-normalized s-type + p-type Gaussian at origin
    model = AtomicGaussianDensity(grid.points, num_s=1, num_p=1, normalize=False)
    # check center
    assert_almost_equal(model.coord, np.zeros(1), decimal=8)
    # check integration
    value = model.evaluate(np.array([1., 1.]), np.array([4.01, 1.25]), deriv=False)
    value = grid.integrate(value * spherical)
    assert_almost_equal(value, (np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5), decimal=6)
    # normalized s-type + p-type Gaussian at origin
    model = AtomicGaussianDensity(grid.points, num_s=2, num_p=3, normalize=True)
    # check center
    assert_almost_equal(model.coord, np.zeros(1), decimal=8)
    # check integration
    coeffs = np.array([1.05, 3.62, 0.56, 2.01, 3.56])
    expons = np.array([0.50, 1.85, 0.16, 1.36, 2.08])
    value = model.evaluate(coeffs, expons, deriv=False)
    assert_almost_equal(grid.integrate(value * spherical), np.sum(coeffs), decimal=6)


def test_gaussian_model_s_integrate_cubic():
    r"""Test integration of one s-type Molecular Gaussian with cubic grid."""
    axes = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    grid = CubicGrid(np.array([-5.0, -5.0, -5.0]), axes, (65, 65, 65))
    coord = None
    # un-normalized s-type + p-type Gaussian
    model = AtomicGaussianDensity(grid.points, center=coord, num_s=1)
    # check center
    assert_almost_equal(model.coord, np.array([0., 0., 0.]), decimal=8)
    # check integration
    value = model.evaluate(np.array([1.]), np.array([4.01]), deriv=False)
    value = grid.integrate(value)
    assert_almost_equal(value, (np.pi / 4.01) ** 1.5, decimal=6)
    # check assertion error on types of coefficients
    assert_raises(ValueError, model.evaluate, np.array([[1.]]), np.array([1.]))
    assert_raises(ValueError, model.evaluate, np.array([1., 2.]), np.array([1.]))
    assert_raises(ValueError, model.evaluate, np.array([1., 2.]), np.array([1., 2.]))


def test_gaussian_model_sp_integrate_cubic():
    r"""Test integration of one s-type and one p-type Molecular Gaussian with cubic grid."""
    axes = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    grid = CubicGrid(np.zeros(3), axes, (65, 65, 65))
    coord = np.array([10., 10., 10.])
    # un-normalized s-type + p-type Gaussian
    model = AtomicGaussianDensity(grid.points, center=coord, num_s=1, num_p=1)
    # check center
    assert_almost_equal(model.coord, coord, decimal=8)
    # check integration
    value = model.evaluate(np.array([1., 1.]), np.array([4.01, 1.25]), deriv=False)
    value = grid.integrate(value)
    assert_almost_equal(value, (np.pi / 4.01)**1.5 + 1.5 * (np.pi**1.5 / 1.25**2.5), decimal=6)
    # normalized s-type + p-type Gaussian
    model = AtomicGaussianDensity(grid.points, center=coord, num_s=2, num_p=3, normalize=True)
    # check center
    assert_almost_equal(model.coord, coord, decimal=8)
    # check integration
    coeffs = np.array([1.05, 3.62, 0.56, 2.01, 3.56])
    expons = np.array([0.50, 1.85, 0.16, 1.36, 2.08])
    value = model.evaluate(coeffs, expons, deriv=False)
    assert_almost_equal(grid.integrate(value), np.sum(coeffs), decimal=6)


def test_molecular_gaussian_model_sp_integrate_cubic():
    r"""Test integration of one s-type and one p-type Molecular Gaussian with cubic grid."""
    axes = np.array([[0.3, 0.0, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]])
    grid = CubicGrid(np.zeros(3), axes, (75, 75, 75))
    coord = np.array([[10., 10., 9.5], [10., 10., 10.5]])
    # un-normalized s-type & p-type Gaussian
    basis = np.array([[0, 1], [1, 0]])
    model = MolecularGaussianDensity(grid.points, coord, basis=basis, normalize=False)
    # check basis
    assert_equal(model.nbasis, 2)
    assert_equal(model.assign_basis_to_center(0), 0)
    assert_equal(model.assign_basis_to_center(1), 1)
    # check integration
    value = model.evaluate(np.array([1., 1.]), np.array([1.25, 4.01]), deriv=False)
    value = grid.integrate(value)
    assert_almost_equal(value, 1.5 * (np.pi**1.5 / 1.25**2.5) + (np.pi / 4.01)**1.5, decimal=6)
    # normalized s-type & p-type Gaussian
    basis = np.array([[2, 0], [0, 3]])
    model = MolecularGaussianDensity(grid.points, coords=coord, basis=basis, normalize=True)
    # check integration
    coeffs = np.array([1.05, 3.62, 0.56, 2.01, 3.56])
    expons = np.array([0.50, 1.85, 0.16, 1.36, 2.08])
    value = model.evaluate(coeffs, expons, deriv=False)
    assert_almost_equal(grid.integrate(value), np.sum(coeffs), decimal=6)
    # normalized (s-type + p-type) & p-type Gaussian
    basis = np.array([[2, 2], [0, 1]])
    model = MolecularGaussianDensity(grid.points, coords=coord, basis=basis, normalize=True)
    # check integration
    value = model.evaluate(coeffs, expons, deriv=False)
    assert_almost_equal(grid.integrate(value), np.sum(coeffs), decimal=6)
