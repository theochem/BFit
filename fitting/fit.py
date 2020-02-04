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
"""Gaussian Basis Fitting Module."""


import numpy as np

from scipy.optimize import minimize

from fitting.measure import KLDivergence, SquaredDifference


__all__ = ["KLDivergenceSCF", "GaussianBasisFit"]


class BaseFit(object):
    """Base Fitting Class."""

    def __init__(self, grid, density, model, measure):
        """
        Parameters
        ----------
        grid :
            The grid class.
        density : ndarray
            The true density evaluated on the grid points.
        model :
            The Gaussian basis model density.
        measure
            The deviation measure between true density and model density.
        """
        self.grid = grid
        self.density = density
        self.model = model
        self.measure = measure

    def goodness_of_fit(self, coeffs, expons):
        r"""Compute various measures to see how good is the fitted model.

        Parameters
        ----------
        coeffs : ndarray
            The coefficients of Gaussian basis functions.
        expons : ndarray
            The exponents of Gaussian basis functions.

        Returns
        -------
        n : float
            Integral of approximate model density, i.e. norm of approximate model density.
        l1 : float
            Integral of absolute difference between density and approximate model density.
        m : float
            Integral of deviation measure between density and approximate model density.
        """
        # evaluate approximate model density
        approx = self.model.evaluate(coeffs, expons)
        # compute deviation measure on the grid
        value = self.measure.evaluate(approx, deriv=False)
        diff = np.abs(self.density - approx)
        return [self.grid.integrate(approx),
                self.grid.integrate(diff),
                np.max(diff),
                self.grid.integrate(self.weights * value)]


class KLDivergenceSCF(BaseFit):
    r"""Kullback-Leiber Divergence Self-Consistent Fitting."""

    def __init__(self, grid, density, model, weights=None, mask_value=0.):
        """
        Parameters
        ----------
        grid :
            The grid class.
        density : ndarray
            The true density evaluated on the grid points.
        model :
            The Gaussian basis model density.
        weights : ndarray, optional
            The weights of objective function at each point. If `None`, 1.0 is used.
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division.
        """
        # initialize KL deviation measure
        measure = KLDivergence(density, mask_value=mask_value)
        super(KLDivergenceSCF, self).__init__(grid, density, model, measure)
        # compute norm of density
        self.norm = grid.integrate(density)
        if weights is None:
            weights = np.ones(len(density))
        self.weights = weights
        # compute lagrange multiplier
        self._lm = self.grid.integrate(self.density * self.weights) / self.norm
        if self._lm == 0. or np.isnan(self._lm):
            raise RuntimeError("Lagrange multiplier cannot be {0}.".format(self._lm))

    @property
    def lagrange_multiplier(self):
        """The lagrange multiplier."""
        return self._lm

    def _update_params(self, coeffs, expons, update_coeffs=True, update_expons=False):
        """Compute updated coefficients & exponents of Gaussian basis functions.

        Parameters
        ----------
        coeffs : ndarray
            The initial coefficients of Gaussian basis functions.
        expons : ndarray
            The initial exponents of Gaussian basis functions.
        update_coeffs : bool, optional
            Whether to optimize coefficients of Gaussian basis functions.
        update_expons : bool, optional
            Whether to optimize exponents of Gaussian basis functions.

        Returns
        -------
        coeffs : ndarray
            The updated coefficients of Gaussian basis functions. Only returned if `deriv=True`.
        expons : ndarray
            The updated exponents of Gaussian basis functions. Only returned if `deriv=True`.
        """
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
            avrg1[index] = self.grid.integrate(integrand)
            if update_expons:
                if self.model.natoms == 1:
                    # case of AtomicGaussianDensity or MolecularGaussianDensity model with 1 atom
                    radii = np.ravel(self.model.radii)
                else:
                    # case of MolecularGaussianDensity model with more than 1 atom
                    center_index = self.model.assign_basis_to_center(index)
                    radii = self.model.radii[center_index]
                avrg2[index] = self.grid.integrate(integrand * radii**2)
        # compute updated coeffs & expons
        if update_coeffs:
            coeffs = coeffs * avrg1 / self._lm
        if update_expons:
            expons = self.model.prefactor * avrg1 / avrg2
        return coeffs, expons

    def run(self, c0, e0, opt_coeffs=True, opt_expons=True, maxiter=500, c_threshold=1.e-6,
            e_threshold=1.e-6, d_threshold=1.e-6):
        """Optimize the coefficients & exponents of Gaussian basis functions self-consistently.

        Parameters
        ----------
        c0 : ndarray
            The initial coefficients of Gaussian basis functions.
        e0 : ndarray
            The initial exponents of Gaussian basis functions.
        opt_coeffs : bool, optional
            Whether to optimize coefficients of Gaussian basis functions.
        opt_expons : bool, optional
            Whether to optimize exponents of Gaussian basis functions.
        maxiter : int, optional
            Maximum number of iterations.
        c_threshold : float
            The convergence threshold for absolute change in coefficients.
        e_threshold : float
            The convergence threshold for absolute change in exponents.
        d_threshold : float
            The convergence threshold for absolute change in divergence value.

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
        if c0.shape != (self.model.nbasis,):
            raise ValueError("Argument init_coeffs shape != ({0},)".format(self.model.nbasis))
        if e0.shape != (self.model.nbasis,):
            raise ValueError("Argument init_expons shape != ({0},)".format(self.model.nbasis))

        new_cs, new_es = c0, e0

        diff_divergence = np.inf
        max_diff_coeffs = np.inf
        max_diff_expons = np.inf

        fun, performance = [], []
        niter = 0
        while ((max_diff_expons > e_threshold or max_diff_coeffs > c_threshold) and
               diff_divergence > d_threshold) and maxiter > niter:

            # update old coeffs & expons
            old_cs, old_es = new_cs, new_es
            # update coeffs and/or exponents
            if opt_coeffs and opt_expons:
                new_cs, new_es = self._update_params(new_cs, new_es, True, True)
            elif opt_coeffs:
                new_cs, new_es = self._update_params(new_cs, new_es, True, False)
            elif opt_expons:
                new_cs, new_es = self._update_params(new_cs, new_es, False, True)
            else:
                raise ValueError("Both opt_coeffs & opt_expons are False! Nothing to optimize!")
            # compute max change in cs & expons
            max_diff_coeffs = np.max(np.abs(new_cs - old_cs))
            max_diff_expons = np.max(np.abs(new_es - old_es))
            # compute errors & update niter
            performance.append(self.goodness_of_fit(new_cs, new_es))
            fun.append(performance[-1][-1])
            niter += 1

            # compute absolute change in divergence
            if niter != 1:
                diff_divergence = np.abs(performance[niter - 1][-1] - performance[niter - 2][-1])
            print(diff_divergence, max_diff_coeffs, max_diff_expons)
            print(niter, performance[-1])
            print(new_cs, new_es)
            print("\n")

        # check whether convergence is reached
        if maxiter == niter and diff_divergence > d_threshold:
            success = False
        else:
            success = True

        results = {"x": (new_cs, new_es),
                   "fun": np.array(fun),
                   "success": success,
                   "performance": np.array(performance)}

        return results


class GaussianBasisFit(BaseFit):
    r"""Kullback-Leiber Divergence Fitting using `Scipy.Optimize` Library."""

    def __init__(self, grid, density, model, measure="KL", method="SLSQP", mask_value=0.):
        r"""
        Parameters
        ----------
        grid :
            The grid class.
        density : ndarray
            The true density evaluated on the grid points.
        model :
            The Gaussian basis model density.
        measure : str, optional
            The deviation measure between true density and model density.
        method : str, optional
            The method used for optimizing parameters.
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division.
        """
        if np.any(abs(grid.points - model.points) > 1.e-12):
            raise ValueError("The grid.points & model.points are not the same!")
        if len(grid.points) != len(density):
            raise ValueError("Argument density should have ({0},) shape.".format(len(grid.points)))
        if method.lower() not in ["slsqp"]:
            raise ValueError("Argument method={0} is not recognized!".format(method))

        self.method = method
        # assign measure to measure deviation between density & modeled density.
        if measure.lower() == "kl":
            measure = KLDivergence(density, mask_value=mask_value)
        elif measure.lower() == "sd":
            measure = SquaredDifference(density)
        else:
            raise ValueError("Argument measure={0} not recognized!".format(measure))
        super(GaussianBasisFit, self).__init__(grid, density, model, measure)

    def run(self, c0, e0, opt_coeffs=True, opt_expons=True, maxiter=1000, ftol=1.e-14):
        r"""Optimize coefficients and/or exponents of Gaussian basis functions.

        Parameters
        ----------
        c0 : ndarray
            Initial guess for coefficients of Gaussian basis functions.
        e0 : ndarray
            Initial guess for exponents of Gaussian basis functions.
        opt_coeffs : bool, optional
            Whether to optimize coefficients of Gaussian basis functions.
        opt_expons : bool, optional
            Whether to optimize exponents of Gaussian basis functions.
        maxiter : int, optional
            Maximum number of iterations.
        ftol : float, optional
            Precision goal for the value of objective function in the stopping criterion.
        """
        # set bounds, initial guess & args
        if opt_coeffs and opt_expons:
            bounds = [(1.e-12, np.inf)] * 2 * self.model.nbasis
            x0 = np.concatenate((c0, e0))
            args = ()
        elif opt_coeffs:
            bounds = [(1.e-12, np.inf)] * self.model.nbasis
            x0 = c0
            args = ("fixed_expons", e0)
        elif opt_expons:
            bounds = [(1.e-12, np.inf)] * self.model.nbasis
            x0 = e0
            args = ("fixed_coeffs", c0)
        else:
            raise ValueError("Nothing to optimize!")
        # set constraints
        constraints = [{"fun": self.const_norm, "type": "eq", "args": args}]
        # set optimization options
        options = {"ftol": ftol, "maxiter": maxiter, "disp": True}
        # optimize
        res = minimize(fun=self.func,
                       x0=x0,
                       args=args,
                       method=self.method,
                       jac=True,
                       bounds=bounds,
                       constraints=constraints,
                       options=options,
                       )
        # check successful optimization
        if not res["success"]:
            raise ValueError("Failed Optimization: {0}".format(res["message"]))
        # check constraints

        # split optimized coeffs & expons
        if opt_coeffs and opt_expons:
            coeffs, expons = res["x"][:self.model.nbasis], res["x"][self.model.nbasis:]
        elif opt_coeffs:
            coeffs, expons = res["x"], e0
        else:
            coeffs, expons = c0, res["x"]
        return coeffs, expons, res["fun"], res["jac"]

    def func(self, x, *args):
        r"""Compute objective function and its derivative w.r.t. Gaussian basis parameters.

        Parameters
        ----------
        x : ndarray
            The parameters of Gaussian basis which is being optimized
        args :
        """
        # compute linear combination of gaussian basis functions
        m, dm = self.evaluate_model(x, *args)
        # compute KL divergence
        k, dk = self.measure.evaluate(m, deriv=True)
        # compute objective function & its derivative
        obj = self.grid.integrate(k)
        d_obj = np.zeros_like(x)
        for index in range(len(x)):
            d_obj[index] = self.grid.integrate(dk * dm[:, index])
        return obj, d_obj

    def const_norm(self, x, *args):
        r"""Compute deviation in normalization constraint.

        Parameters
        ----------
        x : ndarray
            The parameters of Gaussian basis which is being optimized
        args :
        """
        norm = self.grid.integrate(self.density)
        # compute linear combination of gaussian basis functions
        m, dm = self.evaluate_model(x, *args)
        cons = norm - self.grid.integrate(m)
        return cons

    def evaluate_model(self, x, *args):
        r"""Compute deviation between density & fitted density & its derivative.

        Parameters
        ----------
        x : ndarray
            The parameters of Gaussian basis which is being optimized
        args :
        """
        # assign coefficients & exponents
        if len(args) != 0:
            if args[0] == "fixed_coeffs":
                coeffs = args[1]
                expons = x
                start, end = self.model.nbasis, 2 * self.model.nbasis
            elif args[0] == "fixed_expons":
                coeffs = x
                expons = args[1]
                start, end = 0, self.model.nbasis
            else:
                raise ValueError("Argument args is not understandable!")
        else:
            coeffs, expons = x[:self.model.nbasis], x[self.model.nbasis:]
            start, end = 0, 2 * self.model.nbasis
        # compute model density & its derivative
        m, dm = self.model.evaluate(coeffs, expons, deriv=True)
        return m, dm[:, start: end]
