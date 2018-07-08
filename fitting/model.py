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
r"""Model Density Module."""


import numpy as np


__all__ = ["GaussianModel"]


class GaussianModel(object):
    r"""Gaussian Density Model."""

    def __init__(self, points, normalized=False, basis="s"):
        r"""
        Parameters
        ----------
        points : ndarray, (N,)
            The grid points to evaluate model density & its derivatives.
        """
        if not isinstance(points, np.ndarray) or points.ndim != 1:
            raise TypeError("Arguments points should be a 1D numpy array.")
        if basis.lower() != "s":
            raise ImportError("Only s-type Gaussian basis is implemented!")
        self._points = points
        self._normalized = normalized

    @property
    def points(self):
        """The grid points."""
        return self._points

    def evaluate(self, coeffs, expons, deriv=False):
        """Compute linear combination of Gaussian basis and its derivatives on grid points.

        .. math::

        Parameters
        ----------
        coeffs : ndarray, (M,)
            The coefficients of Gaussian basis functions.
        expons : ndarray, (M,)
            The exponents of Gaussian basis functions.
        deriv : bool, optional
            Whether to normalize Gaussian basis functions.
        Returns
        -------
        g : ndarray, (N,)
            The linear combination of Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, 2*M)
            The derivative of linear combination of Gaussian basis functions w.r.t. coeffs
            (the 1st M columns) & expons (the 2nd M columns) evaluated on the grid points.
            Only returned if `deriv=True`.
        """
        if coeffs.ndim != 1 or expons.ndim != 1:
            raise ValueError("Arguments coeffs and expons should be 1D arrays.")
        if coeffs.size != expons.size:
            raise ValueError("Arguments coeffs and expons should have the same length.")

        # evaluate Gaussian basis on the grid
        matrix = np.exp(-expons[None, :] * np.power(self.points, 2)[:, None])
        # normalize Gaussian basis
        if self._normalized:
            matrix = matrix * (expons[None, :] / np.pi)**1.5
        # make linear combination of Gaussian basis on the grid
        g = np.dot(matrix, coeffs)

        # compute derivatives
        if deriv:
            dg = np.zeros((len(self._points), 2 * coeffs.size))
            # derivative w.r.t. coefficients
            dg[:, :coeffs.size] = matrix
            # derivative w.r.t. exponents
            dg[:, coeffs.size:] = - matrix * np.power(self.points, 2)[:, None] * coeffs[None, :]
            if self._normalized:
                matrix = np.exp(-expons[None, :] * np.power(self.points, 2)[:, None])
                dg[:, coeffs.size:] += 1.5 * matrix * (coeffs * expons**0.5)[None, :] / np.pi**1.5
            return g, dg
        return g
