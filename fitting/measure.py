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
"""Deviation Measure Module."""


import numpy as np


__all__ = ["KLDivergence", "SquaredDifference"]


class SquaredDifference(object):
    """Sum of Squares Difference Class."""

    def __init__(self, density):
        """
        Parameters
        ----------
        density : ndarray, (N,)
            The exact density evaluated on the grid points.
        """
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        self.density = density

    def evaluate(self, model, deriv=False):
        r"""Evaluate squared difference b/w density & model on the grid points.

        .. math ::

        Parameters
        ----------
        model : ndarray, (N,)
            The model density evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of squared difference w.r.t. model density.

        Returns
        -------
        m : ndarray, (N,)
            The squared difference between density & model on the grid points.
        dm : ndarray, (N,)
            The derivative of squared difference w.r.t. model density evaluated on the
            grid points. Only returned if `deriv=True`.
        """
        if not isinstance(model, np.ndarray) or model.shape != self.density.shape:
            raise ValueError("Argument model should be {0} array.".format(self.density.shape))
        # compute residual
        residual = self.density - model
        # compute squared residual
        value = np.power(residual, 2)
        # compute derivative of squared residual w.r.t. model
        if deriv:
            return value, -2 * residual
        return value


class KLDivergence(object):
    """Kullback-Leibler Divergence Class."""

    def __init__(self, density, mask_value=1.e-12):
        """
        Parameters
        ----------
        density : ndarray, (N,)
            The exact density evaluated on the grid points.
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division.
        """
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        if np.any(density < 0.):
            raise ValueError("Argument density should be positive.")
        self.density = density
        self.mask_value = mask_value

    def evaluate(self, model, deriv=False):
        r"""Evaluate Kullback-Leibler divergence b/w density & model on the grid points.

        .. math ::

        Parameters
        ----------
        model : ndarray, (N,)
            The model density evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of divergence w.r.t. model density.

        Returns
        -------
        m : ndarray, (N,)
            The divergence between density & model on the grid points.
        dm : ndarray, (N,)
            The derivative of divergence w.r.t. model density evaluated on the grid points.
            Only returned if `deriv=True`.
        """
        # check model density
        if not isinstance(model, np.ndarray) or model.shape != self.density.shape:
            raise ValueError("Argument model should be {0} array.".format(self.density.shape))
        if np.any(model < 0.):
            raise ValueError("Argument model should be positive.")

        # compute ratio & replace masked values by 1.0
        # TODO: If the grid is too wide
        ratio = self.density / np.ma.masked_less_equal(model, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=1.0)
        # compute KL divergence
        value = self.density * np.log(ratio)
        # compute derivative
        if deriv:
            return value, -ratio
        return value
