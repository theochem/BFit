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
This Contains the mbis class responsible for fitting to a gaussian basis set.

Contains the Minimal-Basis-Set-Inter - Stockholder algorithm.
This algorithm minizes the Kullback-Leibler between a probability distribution
composed of gaussian basis set and any other probability distribution.

Note that being a probability distribution means it is integrable.
"""


from __future__ import division
import numpy as np
import numpy.ma as ma
from numbers import Integral
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting

__all__ = ["GaussianKullbackLeibler", "GaussianValKL"]


class GaussianKullbackLeibler(KullbackLeiblerFitting):
    r"""

    """
    def __init__(self, grid_obj, true_model, inte_val=None):
        r"""

        Parameters
        ----------


        """
        super(GaussianKullbackLeibler, self).__init__(grid_obj, true_model, inte_val)
        self.grid_points = ma.asarray(np.reshape(grid_obj.radii, (len(grid_obj.radii), 1)))
        # TODO: Seems like I don't need this attribute
        self.masked_grid_squared = ma.asarray(np.power(self.grid_obj.radii, 2.))

    def get_norm_coeffs(self, coeff_arr, exp_arr):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        return coeff_arr * self.get_norm_consts(exp_arr)

    def get_model(self, coeff_arr, exp_arr, norm=True):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        exponential = np.exp(-exp_arr * np.power(self.grid_points, 2.))
        if norm:
            coeff_arr = self.get_norm_coeffs(coeff_arr, exp_arr)
        normalized_gaussian_density = np.dot(exponential, coeff_arr)
        return normalized_gaussian_density

    def get_inte_factor(self, exponent, masked_normed_gaussian, upt_exponent=False):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        ratio = self.ma_true_mod / masked_normed_gaussian
        grid_squared = self.grid_obj.radii**2.
        integrand = ratio * np.ma.asarray(np.exp(-exponent * grid_squared))
        if upt_exponent:
            integrand = integrand * self.masked_grid_squared
        return self._get_norm_constant(exponent) * self.grid_obj.integrate_spher(False, integrand)

    def _update_coeffs(self, coeff_arr, exp_arr):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        gaussian = ma.asarray(self.get_model(coeff_arr, exp_arr))
        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_inte_factor(exp_arr[i], gaussian)
        return new_coeff / self.lagrange_multiplier

    def _update_fparams(self, coeff_arr, exp_arr, with_convergence=True):
        r"""

        Parameters
        ----------

        Returns
        -------
        """
        masked_normed_gaussian = np.ma.asarray(self.get_model(coeff_arr, exp_arr)).copy()

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            if with_convergence:
                new_exps[i] = 3. * self._lagrange_multiplier
            else:
                new_exps[i] = 3. * self.get_inte_factor(exp_arr[i], masked_normed_gaussian)
            integration = self.get_inte_factor(exp_arr[i], masked_normed_gaussian, True)
            new_exps[i] /= (2. * integration)
        return new_exps

    def _get_norm_constant(self, exponent):
        return (exponent / np.pi) ** (3./2.)


class GaussianValKL(KullbackLeiblerFitting):
    r"""

    """

    # TODO: Discuss how to have numb_val or to include it in __call__
    def __init__(self, grid_obj, true_model, inte_val, numb_val):
        if not isinstance(numb_val, Integral):
            raise TypeError("Number of valence functions should be an integer.")
        if numb_val <= 0.:
            raise ValueError('Number of valence functions should be positive.')
        super(GaussianValKL, self).__init__(grid_obj, true_model, inte_val)
        self.masked_grid_squared = ma.asarray(np.power(self.grid_obj.radii, 2.))
        self.numb_val = numb_val

    def _get_norm_constant(self, fparam, val=False):
        if val:
            return 2. * (fparam ** (5. / 2)) / (3 * (np.pi ** 1.5))
        return (fparam / np.pi) ** (3. / 2.)

    def get_norm_coeffs(self, coeffs, fparams):
        n = len(fparams)
        norm_coeff = self._get_norm_constant(fparams[:n - self.numb_val])
        norm_coeff_val = self._get_norm_constant(fparams[self.numb_val:], val=True)
        return coeffs * np.append(norm_coeff, norm_coeff_val)

    def get_model(self, coeffs, fparams):
        n = len(fparams)
        grid = np.reshape(self.masked_grid_squared, (len(self.masked_grid_squared), 1))
        s_expo = np.exp(-fparams[:n - self.numb_val] * grid)
        p_expo = np.exp(-fparams[self.numb_val:] * grid)
        norm_coeffs = self.get_norm_coeffs(coeffs, fparams)

        s_gaussian_model = np.dot(s_expo, norm_coeffs[:n - self.numb_val])
        p_gaussian_model = np.dot(p_expo, norm_coeffs[self.numb_val:])
        p_gaussian_model = p_gaussian_model * self.masked_grid_squared
        return s_gaussian_model + p_gaussian_model

    def get_inte_factor(self, exponent, masked_normed_gaussian, upt_exponents=False, val=False):
        ratio = self.ma_true_mod / masked_normed_gaussian
        integrand = ratio * np.exp(-exponent * self.masked_grid_squared)
        const = self._get_norm_constant(exponent, val=val)
        if val:
            integrand *= self.masked_grid_squared
        if upt_exponents:
            integrand *= self.masked_grid_squared
        return const * self.grid_obj.integrate_spher(False, integrand)

    def _update_coeffs(self, coeffs, fparams):
        gaussian_model = self.get_model(coeffs, fparams)
        new_coeff = coeffs.copy()
        val = False
        for i in range(0, len(coeffs)):
            if i >= self.numb_val:
                val = True
            new_coeff[i] *= self.get_inte_factor(fparams[i], gaussian_model, val=val)
            new_coeff[i] /= self.lagrange_multiplier
        return new_coeff

    def _update_fparams(self, coeffs, fparams):
        gaussian_model = self.get_model(coeffs, fparams)
        new_exps = fparams.copy()
        val = False
        for i in range(0, len(fparams)):
            if i < len(fparams) - self.numb_val:
                fac = 3. / 2.
            else:
                fac = 5. / 2.
                val = True
            new_exps[i] = self.get_inte_factor(fparams[i], gaussian_model, val=val)
            integration = self.get_inte_factor(fparams[i], gaussian_model, True, val)
            new_exps[i] *= (fac / integration)
        return new_exps