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


__all__ = ["SquaredDifference", "KLDivergence", "TsallisDivergence"]


class Measure(ABC):
    r"""Abstract base class for the measures."""

    @abstractmethod
    def evaluate(self, true, approx, deriv=False):
        r"""Abstract method for evaluating the measure between true and approximate functions.

        Parameters
        ----------
        true : ndarray(N,)
            The true function :math:`f(x)` evaluated on the grid points.
        approx : ndarray(N,)
            The approximate function :math:`g(x)` evaluated on the grid points.
        deriv : bool, optional
            Whether it should return the derivatives of the measure w.r.t. to the
            .

        Returns
        -------
        m : ndarray(N,)
            The measure between true and approximate function on the grid points.
        dm : ndarray(N,)
            The derivative of measure w.r.t. approximate function evaluated on the grid
            points, if `deriv=True`.

        """
        raise NotImplementedError("Evaluate function should be implemented.")


class SquaredDifference(Measure):
    r"""Squared Difference Class."""

    def evaluate(self, true, approx, deriv=False):
        r"""Evaluate squared difference between true and approximate functions.

        .. math ::
           (f(x) - g(x))^2

        Parameters
        ----------
        true : ndarray(N,)
            The true function :math:`f(x)` evaluated on the grid points.
        approx : ndarray(N,)
            The approximate function :math:`g(x)` evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of squared difference w.r.t. :math:`g(x)`.

        Returns
        -------
        m : ndarray(N,)
            The squared difference between true and approximate functions on the grid points.
        dm : ndarray(N,)
            The derivative of squared difference w.r.t. approximate function on the grid points,
            if `deriv=True`.

        """
        if not isinstance(approx, np.ndarray) or approx.ndim != 1:
            raise ValueError(
                f"Argument approx should be an 1D numpy array, got {type(approx)} {approx.ndim}."
            )
        if not isinstance(true, np.ndarray) or true.ndim != 1:
            raise ValueError(
                f"Argument true should be a 1D numpy array, got {type(true)} {true.ndim}."
            )
        if approx.shape != true.shape:
            raise ValueError(
                f"The shape of true and approx arguments do not match, got {true.shape} != "
                f"{approx.shape}"
            )
        if not isinstance(deriv, bool):
            raise TypeError(f"Argument deriv should be a boolean, got {type(deriv)}")
        # compute squared residual & its derivative w.r.t. approx function
        residual = true - approx
        value = np.power(residual, 2)
        if deriv:
            return value, -2 * residual
        return value


class KLDivergence(Measure):
    r"""Kullback-Leibler (KL) Divergence Class."""

    def __init__(self, mask_value=1.0e-12, negative_val=100000.0):
        r"""
        Parameters
        ----------
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division,
            and then replaced with the value of one so that logarithm of one is zero.
        negative_val : (float, np.inf), optional
            Constant value that gets returned if the approximate function is negative
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
        r"""Value that gets returned if the approximate function is negative."""
        return self._negative_val

    def evaluate(self, true, approx, deriv=False):
        r"""Evaluate the integrand of KL divergence between true and approximate functions.

        .. math ::
            f(x) \ln \bigg( \frac{f(x)}{g(x)} \bigg)

        If the approximate function is negative, then this function will return infinity.
        If :math:`g(x) < \text{mask_value}` then :math:`\frac{f(x)}{g(x)} = 1`.

        Parameters
        ----------
        true : ndarray(N,)
            The true function :math:`f(x)` evaluated on the grid points.
        approx : ndarray(N,)
            The approximate function :math:`g(x)` evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of KL divergence w.r.t. :math:`g(x)`.

        Returns
        -------
        m : ndarray(N,)
            The KL divergence between true and approximate functions on the grid points.
        dm : ndarray(N,)
            The derivative of KL divergence w.r.t. approximate function evaluated on the grid
            points, if `deriv=True`.

        Raises
        ------
        ValueError :
            If the approximate function is negative, then the integrand is un-defined.

        Notes
        -----
        - Values of approximate function that are less than `mask_value` are masked when used in
          division and then replaced with the value of 1 so that :math:`\ln(1) = 0`.

        """
        if not isinstance(approx, np.ndarray) or approx.ndim != 1:
            raise ValueError(
                f"Argument approx should be an 1D numpy array, got {type(approx)} {approx.ndim}."
            )
        if not isinstance(true, np.ndarray) or true.ndim != 1:
            raise ValueError(
                f"Argument true should be a 1D numpy array, got {type(true)} {true.ndim}."
            )
        if approx.shape != true.shape:
            raise ValueError(
                f"The shape of true and approx arguments do not match, got {true.shape} != "
                f"{approx.shape}"
            )
        if not isinstance(deriv, bool):
            raise TypeError(f"Argument deriv should be a boolean, got {type(deriv)}")

        # TODO: Add comments to better explain what is happening
        if np.any(approx < 0.0):
            if deriv:
                # Add an incredibly large derivative
                return (
                    np.array([self.negative_val] * approx.shape[0]),
                    np.array([self.negative_val] * approx.shape[0]),
                )
            return np.array([self.negative_val] * approx.shape[0])

        # compute ratio & replace masked values by 1.0
        ratio = true / np.ma.masked_less_equal(approx, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=1.0)

        # compute KL divergence and its derivative
        # Add ignoring division by zero and multiplying by np.nan
        # TODO: With masking, why do we need these ignores?
        with np.errstate(divide="ignore", invalid="ignore"):
            value = true * np.log(ratio)
        if deriv:
            return value, -ratio
        return value


class TsallisDivergence(Measure):
    r"""Tsallis Divergence Class."""

    def __init__(self, alpha=1.001, mask_value=1.0e-12):
        r"""
        Parameters
        ----------
        alpha : float
            The alpha parameter of the Tsallis divergence. When :math:`\alpha \to 1`, the Tsallis
            divergence approaches Kullback-Leibler, so `KLDivergence` measure should be used.
        mask_value : float, optional
            The elements (in the denominator) less than or equal to this value are masked in a
            division, and then replaced with the value of one so that :math:`\ln(1)=0`.

        Notes
        -----
        - Choose :math:`\alpha` carefully. If :math:`\alpha > 1` and it isn't too
          close to 1, then Tsallis divergence upper bounds the Kullback-Leibler and can be useful
          for optimization purposes.

        """
        self._alpha = alpha
        self._mask_value = mask_value
        if self._alpha < 0:
            raise ValueError(f"Alpha parameter should be positive, got {alpha}.")
        if np.abs(self._alpha - 1.0) < 1e-10:
            raise ValueError(
                f"For the case of alpha=1.0, use KLDivergence measure instead, got {alpha}"
            )
        super().__init__()

    @property
    def alpha(self):
        r"""Return alpha parameter of the Tsallis divergence."""
        return self._alpha

    @property
    def mask_value(self):
        r"""Return masking value used when evaluating the measure."""
        return self._mask_value

    def evaluate(self, true, approx, deriv=False):
        r"""Evaluate the integrand of Tsallis divergence ([1]_, [2]_).

        .. math::
            \frac{1}{\alpha - 1} f(x) \bigg(\frac{f(x)}{g(x)}^{\alpha- 1} - 1\bigg)

        Parameters
        ----------
        true : ndarray(N,)
            The true function :math:`f(x)` evaluated on the grid points.
        approx : ndarray(N,)
            The approximate function :math:`g(x)` evaluated on the grid points.
        deriv : bool, optional
            Whether to compute the derivative of Tsallis divergence w.r.t. :math:`g(x)`.

        Returns
        -------
        m : ndarray(N,)
            The Tsallis divergence between true and approximate functions on the grid points.
        dm : ndarray(N,)
            The derivative of divergence w.r.t. approximate function evaluated on the grid points,
            if `deriv=True`.

        Notes
        -----
        - This does not impose non-negativity of the approximate function, unlike `KLDivergence`.

        - Values of approximate function that are less than `mask_value` are masked when used in
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
        if not isinstance(approx, np.ndarray) or approx.ndim != 1:
            raise ValueError(
                f"Argument approx should be an 1D numpy array, got {type(approx)} {approx.ndim}."
            )
        if not isinstance(true, np.ndarray) or true.ndim != 1:
            raise ValueError(
                f"Argument true should be a 1D numpy array, got {type(true)} {true.ndim}."
            )
        if approx.shape != true.shape:
            raise ValueError(
                f"The shape of true and approx arguments do not match, got {true.shape} != "
                f"{approx.shape}"
            )
        if not isinstance(deriv, bool):
            raise TypeError(f"Argument deriv should be a boolean, got {type(deriv)}")

        # compute ratio & replace masked values by 1.0
        ratio = true / np.ma.masked_less_equal(approx, self.mask_value)
        ratio = np.ma.filled(ratio, fill_value=0.0)
        value = true * (np.power(ratio, self.alpha - 1.0) - 1.0)
        integrand = value / (self.alpha - 1.0)
        if deriv:
            return integrand, -np.power(ratio, self.alpha)
        return integrand
