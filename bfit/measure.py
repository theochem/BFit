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
import numpy as np

__all__ = ["KLDivergence", "SquaredDifference"]


class SquaredDifference:
    r"""
    Squared Difference Measure.

    This the defined to be the square of the :math:`L_2` norm:

    .. math::
        ||f - g||_2^2 = \int_G (f(x) - g(x))^2 dx
    where,
        :math:`f` is the true function,
        :math:`g` is the model function,
        :math:`G` is the domain of the grid.

    The term :math:`|f(x) - g(x)|^2` is called the Squared Difference between f and g on a point x.
    """

    def __init__(self, density):
        """
        Construct the SquaredDifference class.

        Parameters
        ----------
        density : ndarray, (N,)
            The exact density (to be fitted) evaluated on the grid points.

        """
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        self._density = density

    @property
    def density(self):
        r"""Array of size :math:`N` of the true density evaluated on `N` points."""
        return self._density

    def evaluate(self, model, deriv=False):
        r"""Evaluate squared difference b/w density & model on the grid points.

        This is defined to be :math:`(f(x) - g(x))^2`.

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


class KLDivergence:
    r"""
    Kullback-Leibler Divergence Class.

    This is defined as the integral:
    .. math::
        D(f, g) := \int_G f(x) \ln ( \frac{f(x)}{g(x)} ) dx
    where,
        :math:`f` is the true probability distribution,
        :math:`g` is the model probability distribution,
        :math:`G` is the grid.

    Attributes
    ----------
    density : ndarray(N,)
        The true function (being approximated) evaluated on `N` points.
    mask_value : float
        Values of model density `g` that are less than `mask_value` are masked when used in division
         and then replaced with the value of 1 so that logarithm of one is zero.

    Methods
    -------
    evaluate(deriv=False)
        Return the integrand :math:`f(x) \ln(f(x)/g(x))` between model and true functions.
        Note that it does not integrate the model, and so it doesn't exactly return the
        Kullback-Leibler. If `deriv` is True, then the derivative with respect to model function
        is returned.

    Notes
    -----
    - This class does not return the Kullback-Leibler but rather the integrand.
        One would need to integrate this to get the Least Squared.
    - It using masked values to handle overflow and underflow floating point precision issues. This
        is due to the division in the Kullback-Leibler formula between two probability
        distributions.

    """

    def __init__(self, density, mask_value=1.e-12):
        r"""
        Construct the Kullback-Leibler class.

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
        r"""
        Evaluate the integrand of Kullback-Leibler divergence b/w true & model.

        .. math ::
            D(f, g) := \int_G f(x) \ln ( \frac{f(x)}{g(x)} ) dx
        where,
            :math:`f` is the true probability distribution,
            :math:`g` is the model probability distribution,
            :math:`G` is the grid.

        If the model density is negative, then this function will return extremely large values,
        for optimization purposes.

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

        Notes
        -----
        - Values of Model density that are less than `mask_value` are masked when used in
            division and then replaced with the value of 1 so that logarithm of one is zero.

        """
        # check model density
        if not isinstance(model, np.ndarray) or model.shape != self.density.shape:
            raise ValueError("Argument model should be {0} array.".format(self.density.shape))
        if np.any(model < 0.):
            # If the model density is negative, then return large values for optimization to favour
            #  positive model densities.
            return np.array([100000.] * len(self.density)), np.array([100000.] * len(self.density))

        # compute ratio & replace masked values by 1.0
        ratio = self.density / np.ma.masked_less_equal(model, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=1.0)

        # compute KL divergence
        value = self.density * np.log(ratio)
        # compute derivative
        if deriv:
            return value, -ratio
        return value
