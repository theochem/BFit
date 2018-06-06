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
from fitting.radial_grid.general_grid import RadialGrid
from fitting.radial_grid.clenshaw_curtis import ClenshawGrid

__all__ = ["test_get_descriptors_of_model",
           "test_get_kullback_leibler",
           "test_get_lagrange_multiplier",
           "test_goodness_of_fit",
           "test_goodness_of_fit_squared",
           "test_input_checks",
           "test_integration_spherical"]


def test_input_checks():
    r"""Test input checks for 'fitting.kl_divergence.KullbackLeiblerFitting'."""
    g = ClenshawGrid(10, 2, 1)
    e = np.array(g.radii * 5.)
    npt.assert_raises(TypeError, KullbackLeiblerFitting, 10., e)
    npt.assert_raises(TypeError, KullbackLeiblerFitting, g, 10.)
    npt.assert_raises(TypeError, KullbackLeiblerFitting, g, e, 5j)
    npt.assert_raises(ValueError, KullbackLeiblerFitting, g, e, -5)
    npt.assert_raises(ValueError, KullbackLeiblerFitting, g, e, 0.)

    # Test that lagrange multiplier gives zero or nan.
    g = RadialGrid(np.arange(0., 10.))
    e = np.exp(-g.radii)
    npt.assert_raises(RuntimeError, KullbackLeiblerFitting, g, e, np.nan)
    e = np.zeros(10)
    npt.assert_raises(RuntimeError, KullbackLeiblerFitting, g, e, 1)

    # Test when Integration Value (inte_val) is None
    g = RadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.radii)
    kl = KullbackLeiblerFitting(g, e, None)
    npt.assert_allclose(kl.inte_val, 2. * 4. * np.pi)


def test_raise_not_implemented():
    r"""Test for raising not implemented for KullbackLeiblerFitting class."""
    g = np.arange(10.)
    kl = KullbackLeiblerFitting(RadialGrid(g), g)
    npt.assert_raises(NotImplementedError, kl.get_model)
    npt.assert_raises(NotImplementedError, kl._update_func_params)
    npt.assert_raises(NotImplementedError, kl._update_coeffs)
    npt.assert_raises(NotImplementedError, kl._get_norm_constant)
    npt.assert_raises(NotImplementedError, kl._get_deriv_coeffs(g, g))
    npt.assert_raises(NotImplementedError, kl._get_deriv_fparams(g, g))


def test_get_lagrange_multiplier():
    r"""Test the lagrange multiplier in KullbackLeiblerFitting."""
    g = RadialGrid(np.arange(0., 26, 0.05))
    e = np.exp(-g.radii)
    kl = KullbackLeiblerFitting(g, e, inte_val=1.)
    npt.assert_allclose(kl.lagrange_multiplier, 2. * 4 * np.pi)


def test_integration_spherical():
    r"""Test integration of model in KullbackLeiblerFitting."""
    g = RadialGrid(np.arange(0., 26, 0.01))
    e = np.exp(-g.radii)
    kl = KullbackLeiblerFitting(g, e, inte_val=1.)
    true_answer = kl.integrate_model_spherically(e)
    npt.assert_allclose(true_answer, 2. * 4 * np.pi)


def test_goodness_of_fit():
    r"""Test goodness of fit."""
    g = RadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.radii)
    kl = KullbackLeiblerFitting(g, e, inte_val=1.)
    model = np.exp(-g.radii**2.)
    true_answer = kl.goodness_of_fit(model)
    npt.assert_allclose(true_answer, 0.3431348, rtol=1e-3)


def test_goodness_of_fit_squared():
    r"""Test goodness of fit squared."""
    g = RadialGrid(np.arange(0., 10, 0.01))
    e = np.exp(-g.radii)
    kl = KullbackLeiblerFitting(g, e, inte_val=1.)
    model = np.exp(-g.radii ** 2.)
    true_answer = kl.goodness_of_fit_grid_squared(model)
    npt.assert_allclose(true_answer, 1.60909, rtol=1e-4)


def test_get_kullback_leibler():
    r"""Test kullback leibler formula."""
    # Test same probabiltiy distribution
    g = RadialGrid(np.arange(0., 26, 0.01))
    e = np.exp(-g.radii**2.)
    kl = KullbackLeiblerFitting(g, e)
    true_answer = kl.get_kullback_leibler(e)
    npt.assert_allclose(true_answer, 0.)

    # Test Different Model with wolfram
    # Integrate e^(-x^2) * log(e^(-x^2) / x) 4 pi r^2 dr from 0 to 25
    fit_model = g.radii
    true_answer = kl.get_kullback_leibler(fit_model)
    npt.assert_allclose(true_answer, -0.672755 * 4 * np.pi, rtol=1e-3)


def test_get_descriptors_of_model():
    r"""Test get descriptors of model."""
    g = RadialGrid(np.arange(0., 10, 0.001))
    e = np.exp(-g.radii)
    kl = KullbackLeiblerFitting(g, e, inte_val=1.)
    model = np.exp(-g.radii**2.)
    true_answer = kl.get_descriptors_of_model(model)
    desired_answer = [5.56833, 0.3431348, 1.60909, 4. * np.pi * 17.360]
    npt.assert_allclose(true_answer, desired_answer, rtol=1e-4)
