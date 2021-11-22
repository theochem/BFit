# -*- coding: utf-8 -*-
# BFit - python program that fits a convex sum of
# positive basis functions to any probability distribution. .
#
# Copyright (C) 2020 The BFit Development Team.
#
# This file is part of BFit.
#
# BFit is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# BFit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
"""Measure Module."""

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["KLDivergence", "SquaredDifference"]


class Measure(ABC):
    r"""Abstract base class for the measures."""

    @abstractmethod
    def evaluate(self, density, model, deriv=False):
        r"""
        Abstract method for evaluating the measure.

        Parameters
        ----------
        density : ndarray(N,)
            The exact density evaluated on the grid points.
        model : ndarray(N,)
            The model evaluated on the same :math:`N` points that `density` is
            evaluated on.
        deriv : bool, optional
            Whether it should return the derivatives of the  measure w.r.t. to the
            model parameters.

        Returns
        -------
        m : ndarray(N,)
            The measure between density & model on the grid points.
        dm : ndarray(N,)
            The derivative of measure w.r.t. model evaluated on the
            grid points. Only returned if `deriv=True`.

        """
        raise NotImplementedError("Evaluate function should be implemented.")


class SquaredDifference(Measure):
    r"""Squared Difference Class for performing the Least-Squared method."""
    def __init__(self):
        r"""Construct the SquaredDifference class."""
        super(SquaredDifference, self).__init__()

    def evaluate(self, density, model, deriv=False):
        r"""
        Evaluate squared difference b/w density & model on the grid points.

        This is defined to be :math:`(f(x) - g(x))^2`,
        where,
            :math:`f` is the true function, and
            :math:`g` is the model function,

        Parameters
        ----------
        density : ndarray(N,)
            The exact density evaluated on the grid points.
        model : ndarray(N,)
            The model density evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of squared difference w.r.t. model density.
            Default is false.

        Returns
        -------
        m : ndarray(N,)
            The squared difference between density & model on the grid points.
        dm : ndarray(N,)
            The derivative of squared difference w.r.t. model density evaluated on the
            grid points, only returned if `deriv=True`.

        Notes
        -----
        - This class returns the squared difference at each point in the domain.
        One would need to integrate this to get the desired measure.

        Notes
        -----
        - This class does not return the Least-Squared but rather the squared difference.
            One would need to integrate this to get the Least Squared.

        """
        if not isinstance(model, np.ndarray):
            raise ValueError("Argument model should be {0} array.".format(density.shape))
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        if model.shape != density.shape:
            raise ValueError(f"Model shape {model.shape} should be the same as density"
                             f" {density.shape}.")
        if not isinstance(deriv, bool):
            raise TypeError(f"Deriv {type(deriv)} should be Boolean type.")
        # compute residual
        residual = density - model
        # compute squared residual
        value = np.power(residual, 2)
        # compute derivative of squared residual w.r.t. model
        if deriv:
            return value, -2 * residual
        return value


class KLDivergence(Measure):
    r"""Kullback-Leibler Divergence Class."""

    def __init__(self, mask_value=1.e-12):
        r"""
        Construct the Kullback-Leibler class.

        Parameters
        ----------
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division,
            and then replaced with the value of one so that logarithm of one is zero.

        """
        super(KLDivergence, self).__init__()
        self._mask_value = mask_value

    @property
    def mask_value(self):
        r"""Masking value used when evaluating the measure."""
        return self._mask_value

    def evaluate(self, density, model, deriv=False):
        r"""
        Evaluate the integrand of Kullback-Leibler divergence b/w true & model.

        .. math ::
            D(f, g) := \int_G f(x) \ln ( \frac{f(x)}{g(x)} ) dx
        where,
            :math:`f` is the true probability distribution,
            :math:`g` is the model probability distribution,
            :math:`G` is the grid being integrated on.

        If the model density is negative, then this function will return extremely large values,
        for optimization purposes.

        Parameters
        ----------
        density : ndarray(N,)
            The exact density evaluated on the grid points.
        model : ndarray(N,)
            The model density evaluated on the grid points.
        deriv : bool, optional
            Whether to return the derivative of divergence w.r.t. model density, as well.
            Default is false.

        Returns
        -------
        m : ndarray(N,)
            The Kullback-Leibler divergence between density & model on the grid points.
        dm : ndarray(N,)
            The derivative of divergence w.r.t. model density evaluated on the grid points.
            Only returned if `deriv=True`.

        Raises
        ------
        ValueError :
            If the model density is negative, then the integrand is un-defined.

        Notes
        -----
        - Values of Model density that are less than `mask_value` are masked when used in
            division and then replaced with the value of 1 so that logarithm of one is zero.
        - This class does not return the Kullback-Leibler but rather the integrand.
            One would need to integrate this to get the Least Squared.

        """
        # check model density
        if not isinstance(model, np.ndarray):
            raise ValueError("Argument model should be {0} array.".format(density.shape))
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        if model.shape != density.shape:
            raise ValueError(f"Model shape {model.shape} should be the same as density"
                             f" {density.shape}.")
        if not isinstance(deriv, bool):
            raise TypeError(f"Deriv {type(deriv)} should be Boolean type.")
        if np.any(model < 0.):
            raise ValueError("Model density should be positive.")

        # compute ratio & replace masked values by 1.0
        ratio = density / np.ma.masked_less_equal(model, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=1.0)

        # compute KL divergence
        value = density * np.log(ratio)
        # compute derivative
        if deriv:
            return value, -ratio
        return value
