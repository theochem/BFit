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

    def __init__(self, points, num_s, num_p, normalized=False):
        r"""
        Parameters
        ----------
        points : ndarray, (N,)
            The grid points.
        num_s : int
             Number of s-type Gaussian basis functions.
        num_p : int
             Number of p-type Gaussian basis functions.
        normalized : bool, optional
            Whether to normalize Gaussian basis functions.
        """
        if not isinstance(points, np.ndarray) or points.ndim != 1:
            print("points = ", points)
            raise TypeError("Argument points should be a 1D numpy array.")
        if not isinstance(num_s, int) or num_s < 0:
            raise TypeError("Argument num_s should be a positive integer.")
        if not isinstance(num_p, int) or num_p < 0:
            raise TypeError("Argument num_p should be a positive integer.")
        if num_s + num_p == 0:
            raise ValueError("Arguments num_s & num_p cannot both be zero!")

        self._points = points
        self.ns = num_s
        self.np = num_p
        self.normalized = normalized

    @property
    def points(self):
        """The grid points."""
        return self._points

    @property
    def num_s(self):
        """Number of s-type Gaussian basis functions."""
        return self.ns

    @property
    def num_p(self):
        """Number of p-type Gaussian basis functions."""
        return self.np

    @property
    def nbasis(self):
        """The total number of Gaussian basis functions."""
        return self.ns + self.np

    def evaluate(self, coeffs, expons, deriv=False):
        """Compute linear combination of Gaussian basis & derivatives on the grid points.

        .. math::

        Parameters
        ----------
        coeffs : ndarray, (`nbasis`,)
            The coefficients of `num_s` s-type Gaussian basis functions followed by the
            coefficients of `num_p` p-type Gaussian basis functions.
        expons : ndarray, (`nbasis`,)
            The exponents of `num_s` s-type Gaussian basis functions followed by the
            exponents of `num_p` p-type Gaussian basis functions.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, `nbasis`)
            The derivative of linear combination of Gaussian basis functions w.r.t. s-type
            coefficients, s-type exponents, p-type coefficients & p-type exponents evaluated
            on the grid points. Only returned if `deriv=True`.
        """
        if coeffs.ndim != 1 or expons.ndim != 1:
            raise ValueError("Arguments coeffs and expons should be 1D arrays.")
        if coeffs.size != expons.size:
            raise ValueError("Arguments coeffs and expons should have the same length.")
        if coeffs.size != self.nbasis:
            raise ValueError("Argument coeffs should have {0} size.".format(self.nbasis))

        # evaluate all Gaussian basis on the grid, i.e., exp(-a * r**2)
        matrix = np.exp(-expons[None, :] * np.power(self.points, 2)[:, None])

        # compute linear combination of Gaussian basis
        if self.np == 0:
            # only s-type Gaussian basis functions
            return self._eval_s(matrix, coeffs, expons, deriv)

        elif self.ns == 0:
            # only p-type Gaussian basis functions
            return self._eval_p(matrix, coeffs, expons, deriv)

        else:
            # both s-type & p-type Gaussian basis functions
            gs = self._eval_s(matrix[:, :self.ns], coeffs[:self.ns], expons[:self.ns], deriv)
            gp = self._eval_p(matrix[:, self.ns:], coeffs[self.ns:], expons[self.ns:], deriv)
            if deriv:
                return gs[0] + gp[0], np.concatenate((gs[1], gp[1]), axis=1)
            return gs + gp

    def _eval_s(self, matrix, coeffs, expons, deriv):
        """Compute linear combination of s-type Gaussian basis & derivatives on the grid points.

        .. math::

        Parameters
        ----------
        matrix : ndarray, (N, M)
             The exp(-a * r**2) array evaluated on grid points for each exponent.
        coeffs : ndarray, (M,)
            The coefficients of Gaussian basis functions.
        expons : ndarray, (M,)
            The exponents of Gaussian basis functions.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of s-type Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, 2*M)
            The derivative of linear combination of s-type Gaussian basis functions w.r.t.
            coefficients (the 1st M columns) & exponents (the 2nd M columns) evaluated on the
            grid points. Only returned if `deriv=True`.
        """
        # normalize Gaussian basis
        if self.normalized:
            matrix = matrix * (expons[None, :] / np.pi) ** 1.5
        # make linear combination of Gaussian basis on the grid
        g = np.dot(matrix, coeffs)

        # compute derivatives
        if deriv:
            dg = np.zeros((len(self._points), 2 * coeffs.size))
            # derivative w.r.t. coefficients
            dg[:, :coeffs.size] = matrix
            # derivative w.r.t. exponents
            dg[:, coeffs.size:] = - matrix * np.power(self.points, 2)[:, None] * coeffs[None, :]
            if self.normalized:
                matrix = np.exp(-expons[None, :] * np.power(self.points, 2)[:, None])
                dg[:, coeffs.size:] += 1.5 * matrix * (coeffs * expons**0.5)[None, :] / np.pi**1.5
            return g, dg
        return g

    def _eval_p(self, matrix, coeffs, expons, deriv):
        """Compute linear combination of p-type Gaussian basis & derivatives on the grid points.

        .. math::

        Parameters
        ----------
        matrix : ndarray, (N, M)
             The exp(-a * r**2) array evaluated on grid points for each exponent.
        coeffs : ndarray, (M,)
            The coefficients of Gaussian basis functions.
        expons : ndarray, (M,)
            The exponents of Gaussian basis functions.
        deriv : bool, optional
            Whether to compute derivative of Gaussian basis functions w.r.t. coefficients &
            exponents.

        Returns
        -------
        g : ndarray, (N,)
            The linear combination of p-type Gaussian basis functions evaluated on the grid points.
        dg : ndarray, (N, 2*M)
            The derivative of linear combination of p-type Gaussian basis functions w.r.t.
            coefficients (the 1st M columns) & exponents (the 2nd M columns) evaluated on the
            grid points. Only returned if `deriv=True`.
        """
        # multiply r**2 with the evaluated Gaussian basis, i.e., r**2 * exp(-a * r**2)
        matrix = matrix * np.power(self.points, 2)[:, None]

        if not self.normalized:
            # linear combination of p-basis is the same as s-basis with an extra r**2
            return self._eval_s(matrix, coeffs, expons, deriv)

        # normalize Gaussian basis
        matrix = matrix * (expons[None, :]**2.5 / np.pi**1.5) / 1.5
        # make linear combination of Gaussian basis on the grid
        g = np.dot(matrix, coeffs)
        if deriv:
            dg = np.zeros((len(self._points), 2 * coeffs.size))
            # derivative w.r.t. coefficients
            dg[:, :coeffs.size] = matrix
            # derivative w.r.t. exponents
            dg[:, coeffs.size:] = - matrix * np.power(self.points, 2)[:, None] * coeffs[None, :]
            matrix = np.exp(-expons[None, :] * np.power(self.points, 2)[:, None])
            matrix = matrix * np.power(self.points, 2)[:, None]
            dg[:, coeffs.size:] += 5 * matrix * (coeffs * expons**1.5)[None, :] / (3 * np.pi**1.5)
            return g, dg
        return g

