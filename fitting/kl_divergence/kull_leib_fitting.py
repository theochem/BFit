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
from scipy.optimize import minimize
from numbers import Real
from fitting.grid import BaseRadialGrid, CubicGrid


__all__ = ["KullbackLeiblerFitting"]


class KullbackLeiblerFitting(object):
    r"""

    """

    def __init__(self, grid_obj, true_model, inte_val=None):
        r"""

        Parameters
        ----------


        """
        if not isinstance(inte_val, (type(None), Real)):
            raise TypeError("Integration Value should be an integer.")
        if inte_val is not None and inte_val <= 0.:
            raise ValueError("Integration value should be positive.")
        if not isinstance(grid_obj, (BaseRadialGrid, CubicGrid)):
            raise TypeError("Grid Object should be 'fitting.radial_grid.radial_grid'.")
        if not isinstance(true_model, np.ndarray):
            raise TypeError("Electron Density should be a numpy array.")
        self.grid_obj = grid_obj
        self.true_model = true_model
        self.ma_true_mod = ma.array(true_model)
        self.inte_val = inte_val
        if inte_val is None:
            self.inte_val = grid_obj.integrate_spher(False, true_model)
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

    def get_model(self):
        raise NotImplementedError()

    def _update_coeffs(self):
        raise NotImplementedError()

    def _update_fparams(self):
        raise NotImplementedError()

    def _get_norm_constant(self):
        raise NotImplementedError()

    def _update_errors(self, coeffs, exps, c, iprint, update_p=False):
        model = self.get_model(coeffs, exps)
        errors = self.get_descriptors_of_model(model)
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
                np.abs(prev_func_val - curr_func_val) > 1e-10:

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
        return self.grid_obj.integrate_spher(False, self.true_model) / self.inte_val

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
        div_model = np.divide(self.ma_true_mod, ma.array(model))
        log_ratio_models = np.log(div_model)
        return self.grid_obj.integrate_spher(False, self.ma_true_mod * log_ratio_models)

    def integrate_model_spherically(self, model):
        r"""
        Integrate the model with additional weights.

        Integrates with a four pi r^2 added to it.

        Parameters
        ----------
        model : np.ndarray
                Approximate / fitted model.

        Returns
        -------
        np.ndarray
                  Integration of the weighted model over the reals
        """
        return self.grid_obj.integrate_spher(False, model)

    def goodness_of_fit_grid_squared(self, model):
        r"""
        The L2 error measure with emphasis on the tail of the grid.

        The emphasis on the tail end of the grid is done by adding a r^2
        in the integrand.

        Parameters
        ----------
        model : np.ndarray
                Approximate / fitted model.

        Returns
        -------
        err : float
              An error measure on how good the fit is.
        """
        absolute_diff = np.abs(model - self.true_model)
        return self.grid_obj.integrate_spher(False, absolute_diff) / (4 * np.pi)

    def goodness_of_fit(self, model):
        r"""
        An error measure based on the L2 norm.

        Parameters
        ----------
        model : np.ndarray
                Approximate / fitted model.

        Returns
        -------
        """

        absolute_diff = np.abs(self.true_model - model)
        return self.grid_obj.integrate(absolute_diff)

    def get_descriptors_of_model(self, model):
        r"""
        Obtains different error measures on the fitted model.

        Integrates the model to see if it converges to the right value.
        Has two different goodness of fit error measures and
        the objective function value to see if it is getting minimized.

        Parameters
        ----------
        model : np.ndarray
                Approximate / fitted model.

        Returns
        -------
        list
            Get all possible forms of error measures on the model.
        """
        return [self.integrate_model_spherically(model),
                self.goodness_of_fit(model),
                self.goodness_of_fit_grid_squared(model),
                self.get_kullback_leibler(model)]

    def _get_deriv_coeffs(self, coeffs, fparams):
        pass

    def _get_deriv_fparams(self, coeffs, fparams):
        pass

    def optimize_slsqp(self, coeffs, fparams):
        def const(_):
            np.sum(coeffs - self.inte_val)
        params = np.append(coeffs, fparams)
        bounds = np.array([(0.0, np.inf)] * len(params))
        opts = {"maxiter": 100000000, "disp": True, "factr": 10.0, "eps": 1e-10}
        f_min_slsqp = minimize(self.get_kullback_leibler, x0=params,
                               method="SLSQP", bounds=bounds, constraints=const,
                               jac=True, options=opts)
        return f_min_slsqp
