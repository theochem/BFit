# -*- coding: utf-8 -*-
# BFit is a Python library for fitting a convex sum of Gaussian
# functions to any probability distribution
#
# Copyright (C) 2020- The QC-Devs Community
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


__all__ = ["SquaredDifference", "KLDivergence", "TsallisDivergence"]


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
            Whether it should return the derivatives of the measure w.r.t. to the
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
          One would need to integrate this to get the desired least-squares.

        """
        if not isinstance(model, np.ndarray):
            raise ValueError(
                f"Argument model should be {density.shape} array."
            )
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

    def __init__(self, mask_value=1.e-12, negative_val=100000.0):
        r"""
        Construct the Kullback-Leibler class.

        Parameters
        ----------
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division,
            and then replaced with the value of one so that logarithm of one is zero.
        negative_val : (float, np.inf), optional
            Constant value that gets returned if the model density is negative
            (i.e. not a true probability distribution). Useful for optimization algorithms
            with weak constraints.

        """
        super().__init__()
        self._mask_value = mask_value
        self._negative_val = negative_val

    @property
    def mask_value(self):
        r"""Masking value used when evaluating the measure."""
        return self._mask_value

    @property
    def negative_val(self):
        r"""Value that gets returned if the model density is negative."""
        return self._negative_val

    def evaluate(self, density, model, deriv=False):
        r"""
        Evaluate the integrand of Kullback-Leibler divergence b/w true & model.

        .. math ::
            D(f, g) := \int_G f(x) \ln \bigg( \frac{f(x)}{g(x)} \bigg) dx

        where,
        :math:`f` is the true probability distribution,
        :math:`g` is the model probability distribution,
        :math:`G` is the grid being integrated on.

        If the model density is negative, then this function will return infinity.
        If :math:`g(x) < \text{mask value}` then :math:`\frac{f(x)}{g(x)} = 1`.

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
        - Values of nodel density that are less than `mask_value` are masked when used in
          division and then replaced with the value of 1 so that logarithm of one is zero.
        - This class does not return the Kullback-Leibler divergence.
          One would need to integrate this to get the KL value.

        """
        # check model density
        if not isinstance(model, np.ndarray):
            raise ValueError(
                f"Argument model should be {density.shape} array."
            )
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        if model.shape != density.shape:
            raise ValueError(f"Model shape {model.shape} should be the same as density"
                             f" {density.shape}.")
        if not isinstance(deriv, bool):
            raise TypeError(f"Deriv {type(deriv)} should be Boolean type.")
        if np.any(model < 0.):
            if deriv:
                # Add an incredibly large derivative
                return np.array([self.negative_val] * model.shape[0]), \
                       np.array([self.negative_val] * model.shape[0])
            return np.array([self.negative_val] * model.shape[0])

        # compute ratio & replace masked values by 1.0
        ratio = density / np.ma.masked_less_equal(model, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=1.0)

        # compute KL divergence
        # Add ignoring division by zero and multiplying by np.nan
        with np.errstate(divide='ignore', invalid="ignore"):
            value = density * np.log(ratio)
        # compute derivative
        if deriv:
            return value, -ratio
        return value


class TsallisDivergence(Measure):
    r"""Tsallis Divergence Class."""

    def __init__(self, alpha=1.001, mask_value=1.e-12):
        r"""
        Construct the Kullback-Leibler class.

        Parameters
        ----------
        alpha : float
            The alpha parameter of the Tsallis divergence. If it tends towards
            one, then Tsallis divergence approaches Kullback-Leibler. See Notes
            for more information.
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division,
            and then replaced with the value of one so that logarithm of one is zero.

        Notes
        -----
        - Care should be taken on :math:`\alpha`. If :math:`\alpha > 1` and it isn't too
          close to 1, then Tsallis divergence upper bounds the Kullback-Leibler and can be useful
          for optimization purposes.

        """
        self._alpha = alpha
        self._mask_value = mask_value
        if self._alpha < 0:
            raise ValueError(f"Alpha parameter {alpha} should be positive.")
        if np.abs(self._alpha - 1.0) < 1e-10:
            raise ValueError(f"Alpha parameter {alpha} shouldn't be equal to one."
                             f"Use Kullback-Leibler divergence instead.")
        super().__init__()

    @property
    def alpha(self):
        r"""Return alpha parameter of the Tsallis divergence."""
        return self._alpha

    @property
    def mask_value(self):
        r"""Return masking value used when evaluating the measure."""
        return self._mask_value

    def evaluate(self, density, model, deriv=False):
        r"""
        Evaluate the integrand of Tsallis divergence ([1]_, [2]_) on grid points.

        Defined as follows:

        .. math::
            \int_G \frac{1}{\alpha - 1} f(x) \bigg(\frac{f(x)}{g(x)}^{q- 1} - 1\bigg) dx,

        where
        :math:`f` is the true probability distribution,
        :math:`g` is the model probability distribution.
        :math:`G` is the grid being integrated on.

        Parameters
        ----------
        density : ndarray(N,)
            The exact density evaluated on the grid points.
        model : ndarray(N,)
            The model density evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of squared difference w.r.t. model density.

        Returns
        -------
        m : ndarray(N,)
            The Tsallis divergence between density & model on the grid points.
        dm : ndarray(N,)
            The derivative of divergence w.r.t. model density evaluated on the grid points.
            Only returned if `deriv=True`.

        Notes
        -----
        - This does not impose non-negativity of the model density, unlike the Kullback-Leibler.

        - Values of Model density that are less than `mask_value` are masked when used in
          division and then replaced with the value of 0.

        - As :math:`\alpha` parameter tends towards one, then this converges to the
          Kullback-Leibler. This is particularly useful for trust-region methods that don't
          impose strict constraints during the optimization procedure.

        - Care should be taken on :math:`\alpha`. If :math:`\alpha > 1` and it isn't too
          close to 1, then Tsallis divergence upper bounds the Kullback-Leibler and can be useful
          for optimization purposes.

        References
        ----------
        .. [1] Ayers, Paul W. "Information theory, the shape function, and the Hirshfeld atom."
           Theoretical Chemistry Accounts 115.5 (2006): 370-378.

        .. [2] Heidar-Zadeh, Farnaz, Ivan Vinogradov, and Paul W. Ayers. "Hirshfeld partitioning
           from non-extensive entropies." Theoretical Chemistry Accounts 136.4 (2017): 54.

        """
        # check model density
        if not isinstance(model, np.ndarray):
            raise ValueError(
                f"Argument model should be {density.shape} array."
            )
        if not isinstance(density, np.ndarray) or density.ndim != 1:
            raise ValueError("Arguments density should be a 1D numpy array.")
        if model.shape != density.shape:
            raise ValueError(f"Model shape {model.shape} should be the same as density"
                             f" {density.shape}.")
        if not isinstance(deriv, bool):
            raise TypeError(f"Deriv {type(deriv)} should be Boolean type.")

        # compute ratio & replace masked values by 1.0
        ratio = density / np.ma.masked_less_equal(model, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=0.0)
        value = density * (np.power(ratio, self.alpha - 1.0) - 1.0)
        integrand = value / (self.alpha - 1.0)
        if deriv:
            return integrand, -np.power(ratio, self.alpha)
        return integrand
