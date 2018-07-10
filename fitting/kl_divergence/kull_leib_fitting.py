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


import numpy as np

from fitting.measure import KLDivergence


__all__ = ["KullbackLeiblerFitting"]


class KullbackLeiblerFitting(object):
    r"""

    """

    def __init__(self, grid, density, model, weights=None, mask_value=0.):
        r"""

        Parameters
        ----------


        """
        self.grid = grid
        self.density = density
        self.model = model
        self.norm = grid.integrate(density, spherical=True)
        if weights is None:
            weights = np.ones(len(density))
        self.weights = weights
        # Various methods relay on masked values due to division of small numbers.
        self._lm = self.grid.integrate(self.density * self.weights, spherical=True) / self.norm
        if self._lm == 0. or np.isnan(self._lm):
            raise RuntimeError("Lagrange multiplier cannot be {0}.".format(self._lm))
        self.errors_arr = []
        self.measure = KLDivergence(density, mask_value=mask_value)

    @property
    def lagrange_multiplier(self):
        return self._lm

    def get_inte_factor(self, exponent, masked_normed_gaussian, upt_exponent=False):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        if self.model.num_p == 0:
            ratio = self.weights * self.density / masked_normed_gaussian
            grid_squared = self.grid.points**2.
            integrand = ratio * np.ma.asarray(np.exp(-exponent * grid_squared))
            if upt_exponent:
                integrand = integrand * self.grid.points**2
            return (exponent / np.pi)**(3./2.) * self.grid.integrate(integrand, spherical=True)

    def _update_coeffs(self, coeff_arr, exp_arr, lm):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        if self.model.num_p == 0:
            gaussian = np.ma.asarray(self.model.evaluate(coeff_arr, exp_arr))
            new_coeff = coeff_arr.copy()
            for i in range(0, len(coeff_arr)):
                new_coeff[i] *= self.get_inte_factor(exp_arr[i], gaussian)
            return new_coeff / lm

    def _update_fparams(self, coeff_arr, exp_arr, lm, with_convergence=True):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        if self.model.num_p == 0:
            masked_normed_gaussian = np.ma.asarray(self.model.evaluate(coeff_arr, exp_arr)).copy()
            new_exps = exp_arr.copy()
            for i in range(0, len(exp_arr)):
                if with_convergence:
                    new_exps[i] = 3. * lm
                else:
                    new_exps[i] = 3. * self.get_inte_factor(exp_arr[i], masked_normed_gaussian)
                integration = self.get_inte_factor(exp_arr[i], masked_normed_gaussian, True)
                new_exps[i] /= (2. * integration)
            return new_exps

    def run(self, eps_coeff, eps_fparam, coeffs, fparams):
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
            coeffs_i1, coeffs_i = self._update_coeffs(coeffs_i1, fparams_i1, self._lm), coeffs_i1

            self.errors_arr.append(self.goodness_of_fit(coeffs_i1, fparams_i1))
            counter += 1

            while np.any(np.abs(coeffs_i - coeffs_i1) > eps_coeff):
                coeffs_i = coeffs_i1
                coeffs_i1 = self._update_coeffs(coeffs_i1, fparams_i1, self._lm)

                self.errors_arr.append(self.goodness_of_fit(coeffs_i1, fparams_i1))
                counter += 1

            fparams_i = fparams_i1
            fparams_i1 = self._update_fparams(coeffs_i1, fparams_i1, self.lagrange_multiplier)

            self.errors_arr.append(self.goodness_of_fit(coeffs_i1, fparams_i1))
            counter += 1
            prev_func_val, curr_func_val = curr_func_val, self.errors_arr[counter - 1][3]

        return {"x": np.append(coeffs_i1, fparams_i1), "iter": counter,
                "errors": np.array(self.errors_arr)}

    def goodness_of_fit(self, coeffs, expons):
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
        # evaluate model density
        dens = self.model.evaluate(coeffs, expons)
        # compute KL deviation measure on the grid
        value = self.measure.evaluate(dens, deriv=False)
        return [self.grid.integrate(dens, spherical=True),
                self.grid.integrate(np.abs(self.density - dens)),
                self.grid.integrate(np.abs(self.density - dens), spherical=True) / (4 * np.pi),
                self.grid.integrate(self.weights * value, spherical=True)]
