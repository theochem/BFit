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
r"""Fitting Algorithms."""

from timeit import default_timer as timer
import warnings

from bfit.measure import KLDivergence, Measure, SquaredDifference
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint


__all__ = ["KLDivergenceFPI", "ScipyFit"]


class _BaseFit:
    r"""Base Fitting Class."""

    def __init__(self, grid, density, model, measure, integral_dens=None, spherical=False,
                 mask_value=1e-18):
        r"""
        Construct the base fitting class.

        Parameters
        ----------
        grid : (_BaseRadialGrid, CubicGrid)
            The grid class that contains the grid points and a integrate function.
             Located in `grid.py`
        density : ndarray(N,)
            The true density evaluated on :math:`N` grid points.
        model : (AtomicGaussianDensity, MolecularGaussianDensity)
            The Gaussian basis model density. See `model.py`.
        measure : (SquaredDifference, KLDivergence)
            The deviation measure between true density and model density. See `measure.py`.
        integral_dens : float, optional
            If this is provided, then the model is constrained to integrate to this value.
            If not, then the model is constrained to the numerical integration of the
            density. Useful when one knows the actual integration value of the density.
        spherical : bool, optional
            Whether to perform spherical integration by adding :math:`4 \pi r^2` term
            to the integrand. Only used when grid is one-dimensional and positive (radial grid).
        mask_value : float, optional
            Mask value used for calculating the Kullback-Leibler divergence. This value sets
            :math:`\log(f(x) / g(x)) = 0` when `g(x)` is less than the mask value.

        """
        if np.any(density < 0.):
            raise ValueError("Density should be positive.")
        self._grid = grid
        if not hasattr(self.grid, "points"):
            raise AttributeError(
                f"Grid class {type(self.grid)} should contain attribute 'points'."
            )
        if not (hasattr(self.grid, 'integrate') and callable(getattr(self.grid, 'integrate'))):
            raise AttributeError(
                f"Grid class {type(self.grid)} should contain method called 'integrate'."
            )
        if self.grid.points.ndim != 1 and spherical:
            raise ValueError(
                f"Spherical is true only when grid points {self.grid.points.ndim} are "
                f"one-dimensional."
            )
        self._density = density
        self._model = model
        self._measure = measure
        self._spherical = spherical
        # compute norm of density
        if integral_dens is None:
            self._integral_dens = self.integrate(density)
        else:
            self._integral_dens = integral_dens

        # Used to calculate the error measures in model only.
        self.kl_error = KLDivergence(mask_value=mask_value)
        self.ls_error = SquaredDifference()

    @property
    def grid(self):
        r"""Return grid object containing points and integration method."""
        return self._grid

    @property
    def density(self):
        r"""Return the true density evaluated on the grid points."""
        return self._density

    @property
    def model(self):
        r"""Return the Gaussian basis model density."""
        return self._model

    @property
    def measure(self):
        r"""Return the deviation measure between true density and model density."""
        return self._measure

    @property
    def integral_dens(self):
        r"""Return integration value of the density."""
        return self._integral_dens

    @property
    def spherical(self):
        """Whether to perform spherical integration."""
        return self._spherical

    def integrate(self, integrand):
        r"""
        Integrate the integrand.

        Parameters
        ----------
        integrand : ndarray(N,)
            The integrand :math:`f(x)` being integrated defined on :math:`N` points.
            If `spherical` attribute is true, then the radial component is
            integrated in spherical coordinates.

        Returns
        -------
        float :
            Returns the integration :math:`\int f(x) dx` or if spherical
            is true then returns :math:`\int_0^\infty f(r) 4 \pi r^2 dr`.

        """
        if self.spherical:
            return self.grid.integrate(integrand * 4.0 * np.pi * self.grid.points**2.0)
        return self.grid.integrate(integrand)

    def goodness_of_fit(self, coeffs, expons):
        r"""
        Compute various measures over the grid to determine the accuracy of the fitted model.

        In particular, it computes the integral of the model, the :math:`L_1` distance,
        the :math:`L_\infty` distance and attribute `measure` distance between true and model
        functions.

        Parameters
        ----------
        coeffs : ndarray
            The coefficients of Gaussian basis functions.
        expons : ndarray
            The exponents of Gaussian basis functions.

        Returns
        -------
        integral : float
            Integral of approximate model density, i.e. norm of approximate model density.
        l_1 : float
            Integral of absolute difference between density and approximate model density.
            This is defined to be :math:`L_(f, g) = \int |f(x) - g(x)| dx`.
        l_infinity : float
            The maximum absolute difference between density and approximate model density.
            This is defined to be :math:`L_\infty(f, g) = \max |f(x) - g(x)|`.
        least_squares : float
            Square of the :math:`L_2` norm between density and approximate model density.
            This is defined to be :math:`L_2^2(f, g) = \int (f(x) - g(x))^2 dx`.
        kullback_leibler : float
            Kullback-Leibler divergence between density and approximate model density.
            This is defined to be :math:`KL(f, g) = \int f(x) \log\bigg(\frac{f(x)}{g(x)}\bigg)dx`.
            If :math:`g` is negative, then this returns infinity.

        """
        # evaluate approximate model density
        approx = self.model.evaluate(coeffs, expons)
        diff = np.abs(self.density - approx)
        return [
            self.integrate(approx),
            self.integrate(diff),
            np.max(diff),
            self.integrate(self.ls_error.evaluate(self.density, approx, deriv=False)),
            self.integrate(self.kl_error.evaluate(self.density, approx, deriv=False))
        ]


class KLDivergenceFPI(_BaseFit):
    r"""
    Kullback-Leibler Divergence FPI method for fitting to a density.

    This class optimizes the following objective function using a fixed point iteration method,

    .. math::
        \min_{\{c_i\}, \{\alpha\}} \int f(x) \log \bigg(\frac{f(x)}{\sum c_i b_i(x, \alpha_i)}
         \bigg)dx + \lambda(N - \sum c_i)

    where :math:`f` is the true density to be fitted,
    :math:`\lambda` is the Lagrange multiplier,
    :math:`c_i` is the coefficient of the ith basis function
    that all sum to the integral of density function :math:`N= \int f(x)dx`, and
    :math:`\alpha_i` is the exponent of the basis function :math:`b_i`.

    """

    def __init__(self, grid, density, model, mask_value=0., integral_dens=None, spherical=False):
        r"""
        Construct the KLDivergenceFPI class.

        Parameters
        ----------
        grid : (_BaseRadialGrid, CubicGrid)
            Grid class that contains the grid points and integration methods on them.
        density : ndarray
            The true density evaluated on the grid points.
        model : (AtomicGaussianDensity, MolecularGaussianDensity)
            The Gaussian basis model density. Located in `model.py`.
        mask_value : float, optional
            The elements less than or equal to this number are masked in a division.
        integral_dens : float, optional
            If this is provided, then the model is constrained to integrate to this value.
            If not, then the model is constrained to the numerical integration of the
            density. Useful when one knows the actual integration value of the density.
        spherical : bool
            Whether to perform spherical integration by adding :math:`4 \pi r^2` term
            to the integrand. Only used when grid is one-dimensional and positive (radial grid).

        """
        # initialize KL deviation measure
        measure = KLDivergence(mask_value=mask_value)
        super().__init__(grid, density, model, measure, integral_dens, spherical, mask_value)
        # compute lagrange multiplier
        self._lm = self.integrate(self.density) / self.integral_dens
        if self._lm == 0. or np.isnan(self._lm):
            raise RuntimeError(f"Lagrange multiplier cannot be {self._lm}.")

    @property
    def lagrange_multiplier(self):
        """Lagrange multiplier of Kullback-Leibler optimization problem."""
        return self._lm

    def _update_params(self, coeffs, expons, update_coeffs=True, update_expons=False):
        r"""
        Update coefficients & exponents of the Gaussian density model.

        Parameters
        ----------
        coeffs : ndarray
            The initial coefficients of Gaussian basis functions.
        expons : ndarray
            The initial exponents of Gaussian basis functions.
        update_coeffs : bool, optional
            Whether to optimize coefficients of Gaussian basis functions.
            Default is true.
        update_expons : bool, optional
            Whether to optimize exponents of Gaussian basis functions.
            Default is true.

        Returns
        -------
        coeffs : ndarray
            The updated coefficients of Gaussian basis functions. Only returned if
            `update_coeffs=True`.
        expons : ndarray
            The updated exponents of Gaussian basis functions. Only returned if
            `update_expones=True`.

        """
        if not update_coeffs and not update_expons:
            raise ValueError("At least one of args update_coeff or update_expons should be True.")
        # compute model density & its derivative
        m, dm = self.model.evaluate(coeffs, expons, deriv=True)
        # compute KL divergence & its derivative
        _, dk = self.measure.evaluate(self.density, m, deriv=True)
        # compute averages needed to update parameters
        avrg1, avrg2 = np.zeros(self.model.nbasis), np.zeros(self.model.nbasis)
        for index in range(self.model.nbasis):
            integrand = -dk * dm[:, index]
            avrg1[index] = self.integrate(integrand)
            if update_expons:
                if self.model.natoms == 1:
                    # case of AtomicGaussianDensity or MolecularGaussianDensity model with 1 atom
                    radii = np.ravel(self.model.radii)
                else:
                    # case of MolecularGaussianDensity model with more than 1 atom
                    center_index = self.model.assign_basis_to_center(index)
                    radii = self.model.radii[center_index]
                avrg2[index] = self.integrate(integrand * radii**2)

        # compute updated coeffs & expons
        if update_coeffs:
            coeffs = coeffs * avrg1 / self._lm
        if update_expons:
            expons = self.model.prefactor * avrg1 / avrg2
        return coeffs, expons

    def run(self, c0, e0, opt_coeffs=True, opt_expons=True, maxiter=500, c_threshold=1.e-6,
            e_threshold=1.e-6, d_threshold=1.e-6, disp=False):
        r"""
        Optimize the coefficients & exponents of Gaussian basis functions via fixed-point.

        Parameters
        ----------
        c0 : ndarray
            The initial coefficients of Gaussian basis functions.
        e0 : ndarray
            The initial exponents of Gaussian basis functions.
        opt_coeffs : bool, optional
            Whether to optimize coefficients of Gaussian basis functions.
            Default is true.
        opt_expons : bool, optional
            Whether to optimize exponents of Gaussian basis functions.
            Default is true.
        maxiter : int, optional
            Maximum number of iterations.
        c_threshold : float
            The termination threshold for absolute change in coefficients. Default is 1e-6.
        e_threshold : float
            The termination threshold for absolute change in exponents. Default is 1e-6.
        d_threshold : float
            The termination threshold for absolute change in divergence value. Default is 1e-6.
        disp : bool
            If true, then at each iteration various error measures will be printed.

        Returns
        -------
        dict :
            The optimization results presented as a dictionary with keys:

            "coeffs" : ndarray
                The optimized coefficients of the Gaussian model.
            "exps" : ndarray
                The optimized exponents of the Gaussian model.
            "success": bool
                Whether the optimization exited successfully.
            "fun" : ndarray
                Values of the KL divergence (objective function) at each iteration.
            "performance" : ndarray
                Values of various performance measures of modeled density at each iteration,
                as computed by `goodness_of_fit()` method.
            "time" : float
                The time in seconds it took to complete the algorithm.

        """
        # check the shape of initial coeffs and expons
        if not isinstance(c0, np.ndarray) or not isinstance(e0, np.ndarray):
            raise TypeError("Initial coefficients or exponents should be numpy arrays.")
        if c0.shape != (self.model.nbasis,):
            raise ValueError(f"Argument init_coeffs shape != ({self.model.nbasis},)")
        if e0.shape != (self.model.nbasis,):
            raise ValueError(f"Argument init_expons shape != ({self.model.nbasis},)")

        new_cs, new_es = c0, e0

        diff_divergence = np.inf
        max_diff_coeffs = np.inf
        max_diff_expons = np.inf

        fun, performance = [], []
        niter = 0
        start = timer()

        if disp:
            # Template for the header.
            print("-" * (10 + 12 + 15 + 15 + 15 + 15 + 15 + 15 + 15 + 12))
            # Format is {identifier:width}, ^ means center it, | means put a bar in it
            template_header = (
                "|{0:^10}|{1:^12}|{2:^15}|{3:^15}|{4:^15}|{5:^16}|{6:^15}|{7:^15}|{8:^15}|"
            )
            # Print the headers
            print(template_header.format(
                "Iteration", "Integration", "L1", "L Infinity", "Least-squares",
                "Kullback-Leibler", "Change Coeffs", "Change Exps", "Change Objective")
            )
            print("-" * (10 + 12 + 15 + 15 + 15 + 15 + 15 + 15 + 15 + 12))
            # Template for float iteration
            template_iters = (
                "|{0:^10d}|{1:^12f}|{2:^15e}|{3:^15f}|{4:^15e}|{5:^16e}|{6:^15e}|{7:^15e}|{8:^15e}|"
            )
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

            if disp:
                print(template_iters.format(
                     niter, *performance[-1], max_diff_coeffs, max_diff_expons, diff_divergence)
                )

        end = timer()
        time = end - start

        # check whether convergence is reached.
        if (maxiter == niter and diff_divergence > d_threshold) or np.isnan(diff_divergence):
            success = False
        else:
            success = True

        if disp:
            print("-" * (10 + 12 + 15 + 15 + 15 + 15 + 15 + 15 + 15 + 12))
            print(f"Successful?: {success}")
            print(f"Time: {time:.2f} seconds")
            print(f"Coefficients {new_cs}")
            print(f"Exponents {new_es}")

        results = {"coeffs": new_cs,
                   "exps" : new_es,
                   "fun": np.array(fun),
                   "success": success,
                   "performance": np.array(performance),
                   "time": time}

        return results


class ScipyFit(_BaseFit):
    r"""
    Optimize least-squares or Kullback-Leibler of Gaussian functions using `Scipy.optimize`.

    Least-squares objective function w.r.t. Gaussian basis-functions is defined as

    .. math::
        \min_{\{c_i\}, \{\alpha\}} \int \bigg(f(x) - \sum c_i b_k(x, \alpha_i) \bigg)^2 dx

    The Kullback-Leibler divergence function is defined as

    .. math::
        \min_{\{c_i\}, \{\alpha\}, \sum c_i = N} \int f(x) \log \bigg(\frac{f(x)}{\sum c_i
        b_i(x, \alpha_i)} \bigg)dx,

    where :math:`f` is the density to be fitted to,
    :math:`c_i, \alpha_i` are the coefficeints and exponents of the Gaussian
    basis functions, and :math:`N = \int f(x)dx` is the integral of the density function.

    Notes
    -----
    - Note that the Kullback-Leibler between two functions :math:`f` and :math:`g` is positive
      if and only if the integrals of :math:`f` and :math:`g` are identical.  The `with_constraint`
      attribute must be True for optimizing Kullback-Leibler.

    """

    def __init__(self, grid, density, model, measure=KLDivergence, method="SLSQP", weights=None,
                 integral_dens=None, spherical=False, mask_value=1e-18):
        r"""
        Construct the ScipyFit object.

        Parameters
        ----------
        grid : (_BaseRadialGrid, CubicGrid)
            The grid class.
        density : ndarray(N,)
            The true density evaluated on the grid points.
        model : (AtomicGaussianDensity, MolecularGaussianDensity)
            The Gaussian basis model density.
        measure : bfit.measure.Measure
            The deviation measure between true density and model density.
            See bfit.measure.py for examples of measures to use.
        method : str, optional
            The method used for optimizing parameters. Default is "slsqp".
            See "scipy.optimize.minimize" for options.
        weights : ndarray, optional
            The weights of objective function at each point. If `None`, 1.0 is used.
        integral_dens : float, optional
            If this is provided, then the model is constrained to integrate to this value.
            If not, then the model is constrained to the numerical integration of the
            density. Useful when one knows the actual integration value of the density.
        spherical : bool, optional
            Whether to perform spherical integration by adding :math:`4 \pi r^2` term
            to the integrand. Only used when grid is one-dimensional and positive (radial grid).
        mask_value : float, optional
            Mask value used for calculating the Kullback-Leibler divergence. This value sets
            :math:`\log(f(x) / g(x)) = 0` when `g(x)` is less than the mask value.

        """
        if np.any(abs(grid.points - model.points) > 1.e-12):
            raise ValueError("The grid.points & model.points are not the same!")
        if len(grid.points) != len(density):
            raise ValueError(f"Argument density should have ({len(grid.points)},) shape.")
        if method.lower() not in ["slsqp", "trust-constr"]:
            raise ValueError(f"Argument method={method} is not recognized!")
        if not isinstance(measure, Measure):
            raise TypeError(f"Measure {type(measure)} needs to be a children of the class Measure.")

        self.method = method.lower()
        # Assign the weights.
        if weights is None:
            weights = np.ones(len(density))
        self.weights = weights
        super().__init__(grid, density, model, measure, integral_dens, spherical, mask_value)

    def run(self, c0, e0, opt_coeffs=True, opt_expons=True, maxiter=1000, tol=1.e-14, disp=False,
            with_constraint=True):
        r"""
        Optimize coefficients and/or exponents of Gaussian basis functions with constraint.

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
        tol : float, optional
            For slsqp. precision goal for the value of objective function in the stopping criterion.
            For trust-constr, it is precision goal for the change in the variables.
        disp : bool
            If True, then it will print the iteration errors, convergence messages from the
            optimizer and will print out various error measures at each iteration.
        with_constraint : bool
            If true, then adds the constraint that the integration of the model density must
            be equal to the constraint of true density. The default is True.

        Returns
        -------
        dict :
            The optimization results presented as a dictionary containing:

            "coeffs" : ndarray
                The optimized coefficients of the Gaussian model.
            "exps" : ndarray
                The optimized exponents of the Gaussian model.
            "success": bool
                Whether the optimization exited successfully.
            "message" : str
                Information about the cause of termination.
            "fun" : float
                Values of KL divergence (objective function) at the final iteration.
            "jacobian": ndarray
                The Jacobian of the objective function w.r.t. coefficients and exponents.
            "performance" : list
                Values of various performance measures of modeled density at each iteration,
                as computed by `_BaseFit.goodness_of_fit` method.
            "time" : float
                The time in seconds it took to optimize.

        Notes
        -----
        - The coefficients and exponents are bounded to be positive.

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
        constraints = []
        if with_constraint:
            if self.method == "slsqp":
                constraints = [{"fun": self.const_norm, "type": "eq", "args": args}]
            elif self.method == "trust-constr":
                constraints = [NonlinearConstraint(self.const_norm, 0.0, 0.0, jac="3-point")]
        # set optimization options
        if self.method == "slsqp":
            options = {"ftol": tol, "maxiter": maxiter, "disp": disp}
        elif self.method == "trust-constr":
            # If the display is true then increase verbosity.
            verbose = 0
            if disp:
                verbose = 3
            options = {"xtol": tol, "maxiter": maxiter, "disp": disp, "verbose": verbose}
        # Set callback to computing the error measures we care about
        callback = None
        if disp:
            if self.method == "slsqp":
                # Template for the header.
                print("-" * (10 + 12 + 15 + 15 + 15 + 12))
                # Format is {identifier:width}, ^ means center it, | means put a bar in it
                template_header = (
                    "|{0:^12}|{1:^15}|{2:^15}|{3:^15}|{4:^16}|"
                )
                # Print the headers
                print(template_header.format(
                    "Integration", "L1", "L Infinity", "Least-squares", "Kullback-Leibler")
                )
                print("-" * (10 + 12 + 15 + 15 + 15 + 12))
                # Template for float iteration
                template_iters = (
                    "|{0:^12f}|{1:^15e}|{2:^15f}|{3:^15e}|{4:^16e}|"
                )

                def print_results(xk):
                    if opt_coeffs and opt_expons:
                        performance = self.goodness_of_fit(xk[:len(c0)], xk[len(c0):])
                    elif opt_coeffs:
                        performance = self.goodness_of_fit(xk, e0)
                    elif opt_expons:
                        performance = self.goodness_of_fit(c0, xk)
                    print(template_iters.format(*performance))

                callback = print_results
            elif self.method == "trust-constr":
                # Template for float iteration
                template_iters = (
                    "Integration: {0:10f},  L1: {1:10e},  L Infinity: {2:10f}, "
                    "Least Squares: {3:10e},  Kullback-Liebler: {4:10e}"
                )

                def print_results(xk, _):
                    if opt_coeffs and opt_expons:
                        performance = self.goodness_of_fit(xk[:len(c0)], xk[len(c0):])
                    elif opt_coeffs:
                        performance = self.goodness_of_fit(xk, e0)
                    elif opt_expons:
                        performance = self.goodness_of_fit(c0, xk)
                    print(template_iters.format(*performance))
                callback = print_results
        # optimize
        start = timer()  # Start timer
        res = minimize(fun=self.func,
                       x0=x0,
                       args=args,
                       method=self.method,
                       jac=True,
                       bounds=bounds,
                       constraints=constraints,
                       options=options,
                       callback=callback,
                       )
        end = timer()
        time = end - start

        # check successful optimization
        if not res["success"]:
            warnings.warn(f"Failed Optimization: {res['message']}")

        # split optimized coeffs & expons
        if opt_coeffs and opt_expons:
            coeffs, expons = res["x"][:self.model.nbasis], res["x"][self.model.nbasis:]
        elif opt_coeffs:
            coeffs, expons = res["x"], e0
        else:
            coeffs, expons = c0, res["x"]

        if disp:
            # Print Final Output.
            if self.method == "slsqp":
                print("-" * (10 + 12 + 15 + 15 + 15 + 12))
            print(f"Successful?: {res['success']}")
            print(f"Time: {time:.2f} seconds")
            print("Scipy Message: " + res["message"])
            print(f"Coefficients {coeffs}")
            print(f"Exponents {expons}")

        results = {"coeffs": coeffs,
                   "exps": expons,
                   "fun": res["fun"],
                   "success": res["success"],
                   "message": res["message"],
                   "jacobian": res["jac"],
                   "performance": np.array(self.goodness_of_fit(coeffs, expons)),
                   "time": time}
        return results

    def func(self, x, *args):
        r"""Compute objective function and its derivative w.r.t. Gaussian basis parameters.

        Parameters
        ----------
        x : ndarray
            The parameters of Gaussian basis which is being optimized. Contains both the
            coefficients and exponents together in a 1-D array.
        args :
            Additional arguments to the model.

        Returns
        -------
        (float, ndarray) :
            The objective function value and its derivative wrt to coefficients and exponents.

        """
        # compute linear combination of gaussian basis functions
        m, dm = self.evaluate_model(x, *args)
        # compute KL divergence
        k, dk = self.measure.evaluate(self.density, m, deriv=True)
        # compute objective function & its derivative
        obj = self.integrate(self.weights * k)
        d_obj = np.zeros_like(x)
        for index in range(len(x)):
            d_obj[index] = self.integrate(self.weights * dk * dm[:, index])
        return obj, d_obj

    def const_norm(self, x, *args):
        r"""Compute deviation in normalization constraint :math:`\sum c_i - \int f(x) dx`.

        Parameters
        ----------
        x : ndarray
            The parameters of Gaussian basis-functions. Contains both the
            coefficients and exponents together in a 1-D array.
        args :
            Additional parameters for the model.

        Returns
        -------
        float :
            The deviation of the integrla with the normalization constant.

        """
        # compute linear combination of gaussian basis functions
        m, _ = self.evaluate_model(x, *args)
        cons = self.integral_dens - self.integrate(m)
        return cons

    def evaluate_model(self, x, *args):
        r"""
        Evaluate the model density & its derivative.

        Parameters
        ----------
        x : ndarray
            The parameters of Gaussian basis-functions. Contains both the
            coefficients and exponents together in a 1-D array.
        args :
            Additional parameters for the model.

        Returns
        -------
        float, ndarray :
            Evaluates the model density & its derivative.

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
