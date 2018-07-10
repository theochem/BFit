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
from numbers import Integral


__all__ = ["GaussianValKL"]


class GaussianValKL(object):
    r"""

    """

    # TODO: Discuss how to have numb_val or to include it in __call__
    def __init__(self, grid, density, numb_val, weights=None):
        if not isinstance(numb_val, Integral):
            raise ValueError("Number of valence functions should be an integer.")
        if numb_val <= 0.:
            raise ValueError('Number of valence functions should be positive.')

        self.grid = grid
        self.density = density
        if weights is None:
            weights = np.ones_like(self.density)
        self.weights = weights
        self.numb_val = numb_val

    def _get_norm_constant(self, fparam, val=False):
        if val:
            return 2. * (fparam ** (5. / 2)) / (3. * (np.pi ** 1.5))
        return (fparam / np.pi) ** (3. / 2.)

    def get_norm_coeffs(self, coeffs, fparams):
        n = len(fparams)
        norm_coeff = self._get_norm_constant(fparams[:n - self.numb_val])
        norm_coeff_val = self._get_norm_constant(fparams[self.numb_val:], val=True)
        return coeffs * np.append(norm_coeff, norm_coeff_val)

    def get_model(self, coeffs, fparams):
        n = len(fparams)
        grid = self.grid.points[:, None]**2
        s_expo = np.exp(-fparams[:n - self.numb_val] * grid)
        p_expo = np.exp(-fparams[self.numb_val:] * grid)
        norm_coeffs = self.get_norm_coeffs(coeffs, fparams)

        s_gaussian_model = np.dot(s_expo, norm_coeffs[:n - self.numb_val])
        p_gaussian_model = np.dot(p_expo, norm_coeffs[self.numb_val:])
        p_gaussian_model = p_gaussian_model * self.grid.points**2
        return s_gaussian_model + p_gaussian_model

    def get_inte_factor(self, exponent, masked_normed_gaussian, upt_exponents=False, val=False):
        ratio = np.ma.array(self.weights * self.density / masked_normed_gaussian, fill_value=0.0)
        integrand = ratio * np.exp(-exponent * self.grid.points**2)
        const = self._get_norm_constant(exponent, val=val)
        if val:
            integrand *= self.grid.points**2
        if upt_exponents:
            integrand *= self.grid.points**2
        return const * self.grid.integrate(integrand, spherical=True)

    def _update_coeffs(self, coeffs, fparams, lm):
        gaussian_model = self.get_model(coeffs, fparams)
        new_coeff = coeffs.copy()
        val = False
        for i in range(0, len(coeffs)):
            if i >= self.numb_val:
                val = True
            new_coeff[i] *= self.get_inte_factor(fparams[i], gaussian_model, val=val)
            new_coeff[i] /= lm
        return new_coeff

    def _update_fparams(self, coeffs, fparams, lm):
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
