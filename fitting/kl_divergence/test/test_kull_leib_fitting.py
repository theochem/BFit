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
r"""Test file for 'fitting.mbis.mbis_abc'"""


import numpy as np
import numpy.testing as npt
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting
from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler
from fitting.grid import BaseRadialGrid, ClenshawRadialGrid


def test_input_checks():
    r"""Test input checks for 'fitting.kl_divergence.KullbackLeiblerFitting'."""
    g = ClenshawRadialGrid(10, 2, 1)
    e = np.array(g.points * 5.)
    npt.assert_raises(TypeError, KullbackLeiblerFitting, 10., e, None)
    npt.assert_raises(TypeError, KullbackLeiblerFitting, g, 10., None)
    npt.assert_raises(TypeError, KullbackLeiblerFitting, g, e, None, 5j)
    npt.assert_raises(ValueError, KullbackLeiblerFitting, g, e, None, -5)
    npt.assert_raises(ValueError, KullbackLeiblerFitting, g, e, None, 0.)

    # Test that lagrange multiplier gives zero or nan.
    g = BaseRadialGrid(np.arange(0., 10.))
    e = np.exp(-g.points)
    npt.assert_raises(RuntimeError, KullbackLeiblerFitting, g, e, None, np.nan)
    e = np.zeros(10)
    npt.assert_raises(RuntimeError, KullbackLeiblerFitting, g, e, None, 1)

    # Test when Integration Value (norm) is None
    g = BaseRadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.points)
    kl = KullbackLeiblerFitting(g, e, None, None)
    npt.assert_allclose(kl.norm, 2. * 4. * np.pi)


def test_get_lagrange_multiplier():
    r"""Test the lagrange multiplier in KullbackLeiblerFitting."""
    g = BaseRadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.points)
    kl = KullbackLeiblerFitting(g, e, None, norm=1.)
    npt.assert_allclose(kl.lagrange_multiplier, 2. * 4 * np.pi)


def test_integration_spherical():
    r"""Test integration of model in KullbackLeiblerFitting."""
    g = BaseRadialGrid(np.arange(0., 26, 0.01))
    e = np.exp(-g.points)
    kl = KullbackLeiblerFitting(g, e, None, norm=1.)
    true_answer = kl.goodness_of_fit(e)[0]
    npt.assert_allclose(true_answer, 2. * 4 * np.pi)


def test_goodness_of_fit():
    r"""Test goodness of fit."""
    g = BaseRadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.points)
    kl = KullbackLeiblerFitting(g, e, None, norm=1.)
    model = np.exp(-g.points**2.)
    true_answer = kl.goodness_of_fit(model)[1]
    npt.assert_allclose(true_answer, 0.3431348, rtol=1e-3)


def test_goodness_of_fit_squared():
    r"""Test goodness of fit squared."""
    g = BaseRadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.points)
    kl = KullbackLeiblerFitting(g, e, None, norm=1.)
    model = np.exp(-g.points ** 2.)
    true_answer = kl.goodness_of_fit(model)[2]
    npt.assert_allclose(true_answer, 1.60909, rtol=1e-4)


def test_get_kullback_leibler():
    r"""Test kullback leibler formula."""
    # Test same probabiltiy distribution
    g = BaseRadialGrid(np.arange(0., 26, 0.01))
    e = np.exp(-g.points**2.)
    model = GaussianKullbackLeibler(g, e, weights=None)
    kl = KullbackLeiblerFitting(g, e, model, weights=None)
    true_answer = kl.get_kullback_leibler(e)
    npt.assert_allclose(true_answer, 0.)

    # Test Different Model with wolfram
    # Integrate e^(-x^2) * log(e^(-x^2) / x) 4 pi r^2 dr from 0 to 25
    fit_model = g.points
    true_answer = kl.get_kullback_leibler(fit_model)
    npt.assert_allclose(true_answer, -0.672755 * 4 * np.pi, rtol=1e-3)


def test_get_descriptors_of_model():
    r"""Test get descriptors of model."""
    g = BaseRadialGrid(np.arange(0., 10, 0.001))
    e = np.exp(-g.points)
    model = GaussianKullbackLeibler(g, e, weights=None)
    kl = KullbackLeiblerFitting(g, e, model, weights=None, norm=1.)
    aprox = np.exp(-g.points**2.)
    true_answer = kl.goodness_of_fit(aprox)
    desired_answer = [5.56833, 0.3431348, 1.60909, 4. * np.pi * 17.360]
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-4)


def test_update_errors():
    g = BaseRadialGrid(np.arange(0., 10, 0.001))
    e = np.exp(-g.points)
    lm = g.integrate(e, spherical=True) / 1.0
    m = GaussianKullbackLeibler(g, e)
    kl = KullbackLeiblerFitting(g, e, m, norm=1.0, weights=None)
    counter = 10
    c = np.array([5.])
    e = np.array([5.])
    c_new = kl._update_errors(c, e, counter, False, False)
    assert c_new == counter + 1


def test_run():
    g = BaseRadialGrid(np.arange(0., 10, 0.001))
    e = (1 / np.pi) ** 1.5 * np.exp(-g.points ** 2.)
    m = GaussianKullbackLeibler(g, e)
    kl = KullbackLeiblerFitting(g, e, m, norm=1.0, weights=None)

    # Test One Basis Function
    c = np.array([1.])

    denom = np.trapz(y=g.points ** 4. * e, x=g.points)
    exps = 3. / (2. * 4. * np.pi * denom)
    params = kl.run(1e-3, 1e-3, c, np.array([exps]), iprint=True)
    params_x = params["x"]
    npt.assert_allclose(1., params_x)
    assert np.abs(params["errors"][-1, 0] - 1.) < 1e-10
    assert np.all(np.abs(params["errors"][-1][1:]) < 1e-10)