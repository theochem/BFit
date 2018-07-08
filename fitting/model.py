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
r"""Module for Least Squares Fitting.

This file designates how to define your model (e.g. gaussian basis set) to fit to a probability
distribution using the least squares objective function.
"""


import numpy as np


__all__ = ["GaussianModel"]


class GaussianModel(object):
    r"""
    Defines Gaussian Basis Set with least squares formula for optimization.

    The gaussian basis set is defined based on coefficients and exponents.
    In addition, the cost-function is provided with it's derivative to be
    used for optimization routine like, slsqp, l-bfgs, and nnls, in the
    'fitting.least_squares.least_sqs'.

    Attributes
    ----------
    grid : np.ndarray
        Grid Points where the 'density' is being defined on.

    density : np.ndarray
        The probability distribution that is being fitted to, defined on the 'grid'.

    Methods
    -------
    create_model(parameters) : Computes the gaussian promolecular density based on the parameters.

    cost_function() : Computes the sum of squares of residuals based on the 'create_model' and
                      'density'.

    derivative_of_cost_function() : Computes the derivative of the 'cost_function'.

    create_cofactor_matrix() : Create the cofactor matrix used for NNLS optimization ie optimizing
                              the coefficients of the basis functions. Matrix of exponential
                              functions.

    """
    def __init__(self, grid, density):
        r"""
        Creates the class by providing formula to the DensityModel class.

        Parameters
        ----------
        grid : np.ndarray
               Radial Grid points for the basis set.

        density : np.ndarray
                    Electron Density to be fitted to. By default, it is the
                    slater densities where the parameters of the slater
                    densities is provided by the file path.

        """
        if not isinstance(grid, np.ndarray):
            raise TypeError("Grid should be a numpy array.")
        if density is not None:
            if not isinstance(density, np.ndarray):
                raise TypeError("Electron least_squares should be an array.")
            if grid.shape != density.shape:
                raise ValueError("Electron least_squares and _grid should be the same "
                                 "size.")
        self._grid = np.ravel(np.copy(grid))
        self._density = np.ravel(density)

    @property
    def grid(self):
        return self._grid

    @property
    def density(self):
        return self._density

    def get_residual(self, *args):
        return self._density - self.create_model(*args)

    def create_model(self, parameters, fixed_params=[], which_opti="b"):
        r"""
        Given the coefficients and exponents, creates the gaussian density
        using the default grid provided in the constructor.

        Parameters
        ----------
        parameters : np.ndarray
                     Parameters to be optimized. Depends on which_opti.
                     if which_opti is:
                     - 'b' contains both coefficients and exponents, resp.
                     - 'c' contains coefficients to be optimized.
                     - 'e' contains exponents to be optimized.

        fixed_params : np.ndarray
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
        float
            Computes the sum of squares of residuals over the grid points.

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
