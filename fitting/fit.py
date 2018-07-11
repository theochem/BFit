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
Contains the abstract base class for optimizing the Kullback-Leibler Divergence.

The point of this class is to define the necessary equations in case
one wants to implement different kinds of linear convex sums of function
using fixed-point iteration or minimizing with traditional methods provided by
'scipy.minimize'.In Addition it can work with the greedy method provided.

If one wants to implement their own linear convex sums of function. They would
have to inherit from KullbackLeibler class.
"""


import numpy as np

from fitting.measure import KLDivergence


__all__ = ["KLDivergenceSCF"]


class KLDivergenceSCF(object):
    r"""Kullback-Leiber Divergence Self-Consistent Fitting."""

    def __init__(self, grid, density, model, weights=None, mask_value=0.):
        r"""

        Parameters
        ----------


        """
        self.grid = grid
        self.density = density
        self.model = model
        self.norm = grid.integrate(density, spherical=True)
        if weights is None:
            weights = np.ones(len(density))
        self.weights = weights
        # Various methods relay on masked values due to division of small numbers.
        self._lm = self.grid.integrate(self.density * self.weights, spherical=True) / self.norm
        if self._lm == 0. or np.isnan(self._lm):
            raise RuntimeError("Lagrange multiplier cannot be {0}.".format(self._lm))
        self.measure = KLDivergence(density, mask_value=mask_value)

    @property
    def lagrange_multiplier(self):
        return self._lm

    def _update_params(self, coeffs, expons, update_coeffs=True, update_expons=False):

        if not update_coeffs and not update_expons:
            raise ValueError("At least one of args update_coeff or update_expons should be True.")
        # compute model density & its derivative
        m, dm = self.model.evaluate(coeffs, expons, deriv=True)
        # compute KL divergence & its derivative
        k, dk = self.measure.evaluate(m, deriv=True)
        # compute averages needed to update parameters
        avrg1, avrg2 = np.zeros(self.model.nbasis), np.zeros(self.model.nbasis)
        for index in range(self.model.nbasis):
            integrand = self.weights * -dk * dm[:, index]
            avrg1[index] = self.grid.integrate(integrand, spherical=True)
            if update_expons:
                avrg2[index] = self.grid.integrate(integrand * self.grid.points**2, spherical=True)
        # compute updated coeffs & expons
        if update_coeffs:
            new_coeffs = coeffs * avrg1 / self._lm
        if update_expons:
            new_expons = 1.5 * avrg1 / avrg2
        if update_coeffs and update_expons:
            return new_coeffs, new_expons
        return new_coeffs if update_coeffs else new_expons

    def run(self, init_coeffs, init_expons, c_threshold, e_threshold, d_threshold, maxiter=500):
        """Optimize the coefficients & exponents of Gaussian basis functions self-consistently.

        Parameters
        ----------
        init_coeffs : ndarray
            The initial coefficients of Gaussian basis functions.
        init_expons : ndarray
            The initial exponents of Gaussian basis functions.
        c_threshold : float
            The convergence threshold for absolute change in coefficients.
        e_threshold : float
            The convergence threshold for absolute change in exponents.
        d_threshold : float
            The convergence threshold for absolute change in divergence value.
        maxiter : int, optional
            The maximum number of iterations.

        Returns
        -------
        result : dict
            The optimization results presented as a dictionary containing:
            "x" : (ndarray, ndarray)
                The optimized coefficients and exponents.
            "success": bool
                Whether or not the optimization exited successfully.
            "fun" : ndarray
                Values of KL divergence (objective function) at each iteration.
            "performance" : ndarray
                Values of various performance measures of modeled density at each iteration,
                as computed by `goodness_of_fit()` method.
        """
        # check the shape of initial coeffs and expons
        if init_coeffs.shape != (self.model.nbasis,):
            raise ValueError("Argument init_coeffs shape != ({0},)".format(self.model.nbasis))
        if init_expons.shape != (self.model.nbasis,):
            raise ValueError("Argument init_expons shape != ({0},)".format(self.model.nbasis))

        new_coeffs, new_expons = init_coeffs, init_expons

        diff_divergence = np.inf
        max_diff_coeffs = np.inf
        max_diff_expons = np.inf

        fun, performance = [], []
        niter = 0
        while max_diff_expons > e_threshold and diff_divergence > d_threshold and maxiter > niter:

            while max_diff_coeffs > c_threshold:
                # update coeffs & compute max |coeffs_change|
                old_coeffs = new_coeffs
                new_coeffs = self._update_params(new_coeffs, new_expons, True, False)
                max_diff_coeffs = np.max(np.abs(new_coeffs - old_coeffs))
                # compute errors & update niter
                performance.append(self.goodness_of_fit(new_coeffs, new_expons))
                fun.append(performance[-1][-1])
                niter += 1

            # update expons & compute max |expons_change|
            old_expons = new_expons
            new_expons = self._update_params(new_coeffs, new_expons, False, True)
            max_diff_expons = np.max(np.abs(new_expons - old_expons))
            # compute errors & update niter
            performance.append(self.goodness_of_fit(new_coeffs, new_expons))
            fun.append(performance[-1][-1])
            niter += 1

            # compute absolute change in divergence
            diff_divergence = np.abs(performance[niter - 1][-1] - performance[niter - 2][-1])

        # check whether convergence is reached
        if maxiter < niter and diff_divergence > d_threshold:
            success = False
        else:
            success = True

        results = {"x": (new_coeffs, new_expons),
                   "fun": np.array(fun),
                   "success": success,
                   "performance": np.array(performance)}

        return results

    def goodness_of_fit(self, coeffs, expons):
        r"""Compute various measures to see how good is the fitted model.

        Parameters
        ----------
        model : ndarray, (N,)
            Value of the fitted model on the grid points.

        Returns
        -------
        model_norm : float
            Integrate(4 * pi * r**2 * model)
        l1_error : float
            Integrate(|density - model|)
        l1_error_modified : float
            Integrate(|density - model| * r**2)
        kl : float
            KL deviation between density and model
        """
        # evaluate model density
        dens = self.model.evaluate(coeffs, expons)
        # compute KL deviation measure on the grid
        value = self.measure.evaluate(dens, deriv=False)
        return [self.grid.integrate(dens, spherical=True),
                self.grid.integrate(np.abs(self.density - dens)),
                self.grid.integrate(np.abs(self.density - dens), spherical=True) / (4 * np.pi),
                self.grid.integrate(self.weights * value, spherical=True)]
