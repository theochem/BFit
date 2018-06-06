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
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting
import numpy as np
import numpy.ma as ma

__all__ = ["GaussianKullbackLeibler"]


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

    def _get_deriv_coeffs(self, coeffs, fparams):
        pass

    def _get_deriv_fparams(self, coeffs, fparams):
        pass
