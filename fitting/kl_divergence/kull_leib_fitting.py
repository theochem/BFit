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
r"""
Contains the abstract base class for optimizing the Kullback-Leibler Divergence.

The point of this class is to define the necessary equations in case
one wants to implement different kinds of linear convex sums of function
using fixed-point iteration or minimizing with traditional methods provided by
'scipy.minimize'.In Addition it can work with the greedy method provided.

If one wants to implement their own linear convex sums of function. They would
have to inherit from KullbackLeibler class.
"""

from __future__ import division
import numpy as np
import numpy.ma as ma
from numbers import Real
from fitting.grid import BaseRadialGrid, CubicGrid


__all__ = ["KullbackLeiblerFitting"]


class KullbackLeiblerFitting(object):
    r"""

    """

    def __init__(self, grid, density, norm=None, weights=None):
        r"""

        Parameters
        ----------


        """
        if not isinstance(norm, (type(None), Real)):
            raise TypeError("Integration Value should be an integer.")
        if not isinstance(weights, (type(None), np.ndarray)):
            raise TypeError("Weights should be none or a numpy array.")
        if norm is not None and norm <= 0.:
            raise ValueError("Integration value should be positive.")
        if not isinstance(grid, (BaseRadialGrid, CubicGrid)):
            raise TypeError("Grid Object should be 'fitting.radial_grid.radial_grid'.")
        if not isinstance(density, np.ndarray):
            raise TypeError("Electron Density should be a numpy array.")
        self.grid = grid
        self.density = ma.array(density)
        self.norm = norm
        if norm is None:
            self.norm = grid.integrate(density, spherical=True)
        if weights is None:
            weights = np.ones(len(density))
        self.weights = weights
        # Various methods relay on masked values due to division of small numbers.
        self._lagrange_multiplier = self.get_lagrange_multiplier()
        if self._lagrange_multiplier == 0.:
            raise RuntimeError("Lagrange multiplier cannot be zero.")
        if np.isnan(self._lagrange_multiplier):
            raise RuntimeError("Lagrange multiplier cannot be nan.")
        self.errors_arr = []

    @property
    def lagrange_multiplier(self):
        return self._lagrange_multiplier

    def get_model(self, coeffs, fparams):
        raise NotImplementedError()

    def _update_coeffs(self):
        raise NotImplementedError()

    def _update_fparams(self):
        raise NotImplementedError()

    def _update_errors(self, coeffs, exps, c, iprint, update_p=False):
        model = self.get_model(coeffs, exps)
        errors = self.goodness_of_fit(model)
        if iprint:
            if update_p:
                print(c + 1, "Update F-param", np.sum(coeffs), errors)
            else:
                print(c + 1, "Update Coeff ", np.sum(coeffs), errors)
        self.errors_arr.append(errors)
        return c + 1

    def _replace_coeffs(self, coeff_arr, exp_arr):
        new_coeff = self._update_coeffs(coeff_arr, exp_arr)
        return new_coeff, coeff_arr

    def _replace_fparams(self, coeff_arr, exp_arr):
        new_exps = self._update_fparams(coeff_arr, exp_arr)
        return new_exps, exp_arr

    def run(self, eps_coeff, eps_fparam, coeffs, fparams, iprint=False, iplot=False):
        r"""

        Parameters
        ----------


        Returns
        -------

        """
        # Old Coeffs/Exps are initialized to allow while loop to hold initially.
        coeffs_i1, coeffs_i = coeffs.copy(), 10. * coeffs.copy()
        fparams_i1, fparams_i = fparams.copy(), 10. * fparams.copy()
        self.errors_arr = []
        prev_func_val, curr_func_val = 1e6, 1e4

        counter = 0
        while np.any(np.abs(fparams_i1 - fparams_i) > eps_fparam) and \
                np.abs(prev_func_val - curr_func_val) > 1e-8:

            # One iteration to update coefficients
            coeffs_i1, coeffs_i = self._replace_coeffs(coeffs_i1, fparams_i1)
            counter = self._update_errors(coeffs_i1, fparams_i1, counter, iprint, iplot)

            while np.any(np.abs(coeffs_i - coeffs_i1) > eps_coeff):
                coeffs_i1, coeffs_i = self._replace_coeffs(coeffs_i1, fparams_i1)
                counter = self._update_errors(coeffs_i1, fparams_i1, counter, iprint, iplot)

            fparams_i1, fparams_i = self._replace_fparams(coeffs_i1, fparams_i1)
            counter = self._update_errors(coeffs_i1, fparams_i1, counter, iprint, update_p=True)
            prev_func_val, curr_func_val = curr_func_val, self.errors_arr[counter - 1][3]

        return {"x": np.append(coeffs_i1, fparams_i1), "iter": counter,
                "errors": np.array(self.errors_arr)}

    def get_lagrange_multiplier(self):
        r"""

        :return:
        """
        return self.grid.integrate(self.density * self.weights, spherical=True) / self.norm

    def get_norm_consts(self, exp_arr):
        r"""
        These are normalization constants for gaussian basis set.

        In order words, this is the inverse of the number you get
        from integrating a gaussian function over the positive reals.

        Parameters
        ----------
        exp_arr : np.ndarray
                  Exponents of the gaussian function.

        Returns
        -------
        np.ndarray
                  Normalization constants.
        """
        return np.array([self._get_norm_constant(x) for x in exp_arr])

    def get_kullback_leibler(self, model):
        r"""
        Compute the Kullback-Leibler formula between the two models over the grid.


        Parameters
        ----------
        model : np.ndarray
                Approximate / fitted model.

        Returns
        -------
        np.ndarray
                  Kullback Leibler fomula
        """
        div_model = np.divide(self.density, ma.array(model))
        log_ratio_models = self.weights * np.log(div_model)
        return self.grid.integrate(self.density * log_ratio_models, spherical=True)

    def goodness_of_fit(self, model):
        r"""Compute various measures to see how good is the fitted model.

        Parameters
        ----------
        model : ndarray, (N,)
            Value of the fitted model on the grid points.

        Returns
        -------
        model_norm : float
            Integrate(4 * pi * r**2 * model)
        l1_error : float
            Integrate(|density - model|)
        l1_error_modified : float
            Integrate(|density - model| * r**2)
        kl : float
            KL deviation between density and model
        """
        return [self.grid.integrate(model, spherical=True),
                self.grid.integrate(np.abs(self.density - model)),
                self.grid.integrate(np.abs(self.density - model), spherical=True) / (4 * np.pi),
                self.get_kullback_leibler(model)]

    def cost_function(self, params):
        r"""
        Get the kullback-leibler formula which is ought to be minimized.

        Used for optimization via SLSQP in the 'fitting.utils.optimize.py' File.

        Parameters
        ----------
        params : np.ndarray
            Coefficients and Function parameters appended together.
        """
        model = self.get_model(params[:len(params)//2], params[len(params)//2:])
        return self.get_kullback_leibler(model)
