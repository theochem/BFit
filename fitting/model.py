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
This file designates how to define your model (e.g. gaussian basis set) to
fit to slater densities using the least squares optimization.

Hence the DensityModel is the abstract class for all models.
It also contains standard error measures to be used when fitting to get a sense
of how good the fit is.
"""


import numpy as np


__all__ = ["DensityModel", "GaussianBasisSet"]


class DensityModel(object):
    """
    This is an abstract class for the gaussian least_squares model
    used for fitting slater densities.

    Primarily used to define the cost function/objective function and
    the residual which is being minimized through least squares.
    Additionally, contains tools to define different error measures,
    as well as UGBS exponents used to define proper initial guesses
    for the Gaussian least_squares model.
    """

    def __init__(self, grid, true_model=None):
        r"""

        Parameters
        ----------
        grid : np.ndarray
               Contains the grid points for the density model.
        element : str, optional
                 The element that the slater densities are based on.
                 Used if one want's to use UGBS parameters as initial guess.
        true_model : np.ndarray
                         Pre-defined electron density in case one doesn't want
                         to use slater densities.
        Raises
        ------
        TypeError
            If an argument of an invalid type is used

        """
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid should be a numpy array.")
        if true_model is not None:
            if not isinstance(true_model, np.ndarray):
                raise TypeError("Electron least_squares should be an array.")
            if grid.shape != true_model.shape:
                raise ValueError("Electron least_squares and _grid should be the same "
                                 "size.")
        self._grid = np.ravel(np.copy(grid))
        self._true_model = np.ravel(true_model)

    @property
    def grid(self):
        return self._grid

    @property
    def true_model(self):
        return self._true_model

    def create_model(self):
        """
        """
        raise NotImplementedError("Need to implement the least_squares model")

    def cost_function(self):
        """
        """
        raise NotImplementedError("Need to implement the cost function")

    def derivative_of_cost_function(self):
        """
        """
        raise NotImplementedError("Need to Implement the derivative of cost "
                                  "function")

    def create_cofactor_matrix(self):
        pass

    def get_residual(self, *args):
        return self._true_model - self.create_model(*args)

    def integrate_model_trapz(self, approx_model):
        r"""
        Integrates the approximate model with an added r^2 over [0, inf),
        using the trapezoidal method.

        Parameters
        ----------
        approx_model : np.ndarray
                       The model obtained from fitting.

        Returns
        -------
        int : float
              Integration value of r^2 with approximate model over the _grid.
        """
        grid_squared = np.ravel(self._grid ** 2.)
        return np.trapz(y=grid_squared * approx_model, x=self._grid)

    def get_error_diffuse(self, true_model, approx_model):
        r"""
        This error measures how good the kl_divergence is between the approximate and
        true least_squares at long densities.
        # TODO: FIX THIS TO MAKE THE LATEX INLINE
        ..math::
            Given two functions, denoted f, g.
            The error is given as \int r^2 |f(r) - g(r)|dr

        Parameters
        ----------
        true_model : np.ndarray
                     The true model that is being fitted.

        approx_model : np.ndarray
                       The model obtained from fitting.

        Returns
        -------
        error : float
                A positive real number that measures how good the kl_divergence is.
        """
        abs_diff = np.absolute(np.ravel(true_model) - np.ravel(approx_model))
        error = np.trapz(y=self._grid**2 * abs_diff, x=self._grid)
        return error

    def get_integration_error(self, true_model, approx_model):
        r"""
        This error measures the difference in integration of the two models.

        ..math::
            Given two functions, denoted f, g.
            The error is given as |\int r^2f(r)dr - \int r^2g(r)dr|.

        Parameters
        ----------
        true_model : np.ndarray
                     The true model that is being fitted.

        approx_model : np.ndarray
                       The model obtained from fitting.

        Returns
        -------
        error : float
                Measures the difference in integration of the two models.
        """
        integrate_true_model = self.integrate_model_trapz(true_model)
        integrate_approx_model = self.integrate_model_trapz(approx_model)
        diff_model = integrate_true_model - integrate_approx_model
        return np.absolute(diff_model)


class GaussianBasisSet(DensityModel):
    r"""
    Defines Gaussian Basis Set with least squares formula for optimization.

    The gaussian basis set is defined based on coefficients and exponents.
    In addition, the cost-function is provided with it's derivative to be
    used for optimization routine like, slsqp, l-bfgs, and nnls, in the
    'fitting.least_squares.least_sqs'.
    """
    def __init__(self, grid, true_model):
        r"""
        Creates the class by providing formula to the DensityModel class.

        Parameters
        ----------
        grid : np.ndarray
               Radial Grid points for the basis set.

        true_model : np.ndarray
                    Electron Density to be fitted to. By default, it is the
                    slater densities where the parameters of the slater
                    densities is provided by the file path.

        """
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid should be a numpy array.")
        if not isinstance(true_model, np.ndarray):
            raise TypeError("True model should be a numpy array.")
        super(GaussianBasisSet, self).__init__(grid, true_model)

    def create_model(self, parameters, fixed_params=[], which_opti="b"):
        r"""
        Given the coefficients and exponents, creates the gaussian density
        using the default grid provided in the constructor.

        Parameters
        ----------
        parameters : np.ndarray or list
                     Parameters to be optimized. Depends on which_opti.
                     if which_opti is:
                     - 'b' contains both coefficients and exponents, resp.
                     - 'c' contains coefficients to be optimized.
                     - 'e' contains exponents to be optimized.

        fixed_params : np.ndarray or list
                     Used when which_opti is 'c' or 'e'. In any case, it contains
                     the parameters that are considered fixed.

        which_opti : str
                    Tells which parameters to optimize. Should be one of:
                    - 'b' optimize both coefficients and exponents.
                    - 'c' optimize coefficients, exponents fixed.
                    - 'e' optimize exponents, coefficients fixed.

        Returns
        -------
        gauss_dens_grid : np.ndarray
                        Array that contains the gauss. density at each pt in the
                        grid.
        """
        if which_opti == "b":
            coeffs = parameters[:len(parameters)//2]
            exps = parameters[len(parameters)//2:]
        elif which_opti == "c":
            coeffs = np.copy(parameters)
            exps = fixed_params
        elif which_opti == "e":
            coeffs = fixed_params
            exps = np.copy(parameters)
        mat = np.exp(-exps * np.power(self.grid.reshape(self.grid.size, 1), 2.))
        gauss_dens_grid = np.dot(mat, coeffs)
        return gauss_dens_grid

    def cost_function(self, parameters, fixed_params=[], which_opti='b'):
        r"""
        The least squares formula between the true electron density and
        our gaussian model.
        #TODO: Add InLine Comments Here
        It is the sum over the radial grid points, where you take the difference
        squared between the true elctron density, , and gaussian density.
        ..math::
            \sum_n^{N_points}(\rho_{true}(r) - \sum_i c_i e^(-\alpha_i r^2))^2

        Parameters
        ----------
        parameters : np.ndarray or list
                     Parameters to be optimized. Depends on which_opti.
                     if which_opti is:
                     - 'b' contains both coefficients and exponents, resp.
                     - 'c' contains coefficients to be optimized.
                     - 'e' contains exponents to be optimized.

        fixed_params : np.ndarray or list
                     Used when which_opti is 'c' or 'e'. In any case, it contains
                     the parameters that are considered fixed.

        which_opti : str
                    Tells which parameters to optimize. Should be one of:
                    - 'b' optimize both coefficients and exponents.
                    - 'c' optimize coefficients, exponents fixed.
                    - 'e' optimize exponents, coefficients fixed.

        Returns
        -------
        """
        res = self.get_residual(parameters, fixed_params, which_opti)
        residual_squared = np.power(res, 2.0)
        return np.sum(residual_squared)

    def _deriv_wrt_coeffs(self, exps, res):
        r"""
        Get Derivative of cost func. wrt to coefficients of gaussian basis set.

        Parameters
        ----------
        exps : np.ndarray
               Exponents of the gaussian basis set.

        res : np.ndarray
              Residual factor that shows up from the chain rule.

        Returns
        -------
        np.ndarray
                  Derivative wrt to coefficients
        """
        cofac = self.create_cofactor_matrix(exps)
        derivative_coeff = -res.dot(cofac)
        return derivative_coeff

    def _deriv_wrt_exps(self, coeff, exps, res):
        r"""
        Get Derivative of cost func. wrt to exponents of gaussian basis set.

        Parameters
        ----------
        coeffs : np.ndarray
                Coefficients of the gaussian basis set.
        exps : np.ndarray
               Exponents of the gaussian basis set.
        res : np.ndarray
              Residual factor that shows up from the chain rule.

        Returns
        -------
        np.ndarray
                  Derivative wrt to each exponents.
        """
        cofac = self.create_cofactor_matrix(exps)
        residual_squared = res * self._grid**2.
        derivative_exp = residual_squared.dot(cofac * coeff)
        return derivative_exp

    def derivative_of_cost_function(self, parameters, fixed_params=[], which_opti="b"):
        r"""
        Derivative of the least squares cost-function.

        Depends on which_opti to know which parameters, coeffs, exps or both,
        to take the derivative wrt to.

        Parameters
        ----------
        parameters : np.ndarray or list
                     Parameters to be optimized. Depends on which_opti.
                     if which_opti is:
                     - 'b' contains both coefficients and exponents, resp.
                     - 'c' contains coefficients to be optimized.
                     - 'e' contains exponents to be optimized.

        fixed_params : np.ndarray or list
                     Used when which_opti is 'c' or 'e'. In any case, it contains
                     the parameters that are considered fixed.

        which_opti : str
                    Tells which parameters to optimize. Should be one of:
                    - 'b' optimize both coefficients and exponents.
                    - 'c' optimize coefficients, exponents fixed.
                    - 'e' optimize exponents, coefficients fixed.

        Returns
        -------
        deriv : np.ndarray
                Sum of the derivative.
        """
        residual = self.get_residual(parameters, fixed_params, which_opti)
        f_function = 2.0 * residual
        if which_opti == "b":
            coeffs = parameters[:len(parameters)//2]
            exps = parameters[len(parameters)//2:]
            deriv = self._deriv_wrt_coeffs(exps, f_function)
            deriv = np.append(deriv,
                              self._deriv_wrt_exps(coeffs, exps, f_function))

        elif which_opti == "c":
            exps = fixed_params
            deriv = self._deriv_wrt_coeffs(exps, f_function)

        elif which_opti == "e":
            exps = np.copy(parameters)
            coeffs = fixed_params
            deriv = self._deriv_wrt_exps(coeffs, exps, f_function)
        return deriv

    def create_cofactor_matrix(self, exponents):
        r"""
        Used for Non-Negative least squares (NNLS).

        Because our approximate least_squares is composed as a sum of
        un-normalized gaussian functions. NNLS optimizes the coefficients,
        while holding the exponents fixed.

        Parameters
        ----------
        exponents : np.ndarray
                    List holding the exponents for each gaussian function.
        Returns
        -------
        cofactor : np.ndarray
                   A Matrix where rows are the points on the grid and columns
                   correspond to each gaussian basis function at that point.
        """
        g_col = self._grid.reshape((len(self._grid), 1))
        exponential = np.exp(-exponents * np.power(g_col, 2.0))
        return exponential
