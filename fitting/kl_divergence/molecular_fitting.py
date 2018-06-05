# -*- coding: utf-8 -*-
# A basis-set curve-fitting optimization package.
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
r"""File contains algorithms responsible for fitting a molecular density to gaussian basis-sets."""

import numpy as np
import numpy.ma as ma
from numbers import Real
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting
from fitting.radial_grid.cubic_grid import CubicGrid

__all__ = ["MolecularFitting"]


class MolecularFitting(KullbackLeiblerFitting):
    def __init__(self, grid_obj, dens_val, inte_val, mol_coords, number_of_params):
        r"""
        Parameters
        ----------
        grid_obj : CubicGrid
            This is the grid object responsible for holding the grid points and integrating.
        
        dens_val : np.ndarray(shape=(1, N))
            Holds the molecular density values, where N is the number of points.
        
        inte_val : float
            The integration of the molecular density over the entire space.
        
        mol_coords : np.ndarray(shape=(M, 3))
            The coordinates of each atom position where M is the number of atoms.
        
        number_of_params : list(shape=(1, M)) 
            The number of gaussian parameters for each atom stored in a list.
            
        """
        if not isinstance(grid_obj, CubicGrid):
            raise TypeError("Grid object should be of type CubicGrid.")
        if not isinstance(dens_val, np.ndarray):
            raise TypeError("Density values should be a numpy array.")
        if not isinstance(inte_val, Real):
            raise TypeError("Integration Value should be a number.")
        if not isinstance(mol_coords, np.ndarray):
            raise TypeError("Molecule Coordinates should be a numpy array.")
        if not isinstance(number_of_params, list):
            raise TypeError("Number of parameters should be a list.")
        if not mol_coords.ndim == 2.:
            raise ValueError("Molecular Coordinates should be a two dimensional array.")
        if not mol_coords.shape[1] == 3:
            raise ValueError("Molecule Coordinates should be three dimensional.")
        if not len(number_of_params) == mol_coords.shape[0]:
            raise ValueError("Length of number of parameters should match molecular coordinates.")
        if not dens_val.ndim == 1:
            raise ValueError("Density values should be one dimensional.")
        self.mol_coords = mol_coords
        self.number_of_params = number_of_params
        super(MolecularFitting, self).__init__(grid_obj, dens_val, inte_val)

    def get_norm_coeffs(self, coeff_arr, exp_arr):
        r"""
        Return new coefficients which is now multiplied by it's normalization factor.

        The normalization factor for the ith coefficient is (ith exponent / pi)^(3 / 2).

        Parameters
        ----------
        coeff_arr : np.array
            The gaussian coefficients.

        exp_arr : np.array
            The gaussian exponents.

        Returns
        -------
        np.array
            New normalized coefficients.

        """
        return coeff_arr * self.get_norm_consts(exp_arr)

    def get_model(self, coeffs, fparams):
        r"""
        Returns the gaussian molecular density situated at each atom's coordinates.

        Parameters
        ----------
        coeffs : np.ndarray(shape=(1, K))
            The function coefficients, where K is the number of parameters.

        fparams : np.ndarray(shape=(1, K))
            The function parameters, where K is the number of parameters.

        Returns
        -------
        np.ndarray :
            Returns the gaussian density values evaluated at each grid point.

        """
        model = np.zeros(len(self.grid_obj))
        coeffs = self.get_norm_coeffs(coeffs, fparams)

        i1 = 0
        for j, numb in enumerate(self.number_of_params):
            coeff = coeffs[i1: numb + i1]
            exp1 = fparams[i1: numb + i1]

            radius = np.zeros(len(self.grid_obj))
            radius += np.sum((self.grid_obj.grid - self.mol_coords[j])**2., axis=1)

            exponential = np.exp(-exp1 * radius.reshape((len(self.grid_obj), 1)))
            model += exponential.dot(coeff)
            i1 = numb + i1
        return model

    def get_inte_factor(self, exponent, masked_normed_gaussian, mol_coord, upt_exponent=False):
        r"""
        Returns different integration factors needed for optimizing coefficients or exponents.

        Which parameter is being optimized is specified by upt_exponent.

        Parameters
        ----------
        exponent : float
            Exponent corresponding to the coefficient being optimized or itself being optimized.

        masked_normed_gaussian : np.ndarray
            Gaussian density values. It is masked, due to division by zero.

        mol_coords : np.ndarray(shape=(M, 3))
            The coordinates of each atom position where M is the number of atoms.

        upt_exponent : bool
            If true, return factors needed for updating exponents.

        Returns
        -------
        float :
            The factors needed for optimizing coefficients and exponents.

        """
        ratio = self.ma_true_mod / masked_normed_gaussian
        grid_squared = np.sum((self.grid_obj.grid - mol_coord)**2., axis=1)
        integrand = ratio * np.ma.asarray(np.exp(-exponent * grid_squared))
        if upt_exponent:
            integrand = integrand * grid_squared
        return self._get_norm_constant(exponent) * self.grid_obj.integrate_spher(False, integrand)

    def get_mol_coord(self, index):
        r"""
        Given an integer index, return the atom's coordinates corresponding to that index.

        Parameters
        ----------
        index : int
            Index of the parameter being optimized.

        Returns
        -------
        list
            Returns a list with three elements corresponding to the atom's real coordinates.

        """
        found = -1
        i1 = 0
        for j, i in enumerate(self.number_of_params):
            if i1 <= index < i + i1:
                found = j
                break
            i1 += i
        assert found != -1.
        return self.mol_coords[found]

    def _update_coeffs_gauss(self, coeff_arr, exp_arr):
        r"""
        Update a list of gaussian coefficients, using the fixed-point iteration method.

        Parameters
        ----------
        coeff_arr : np.array
            The gaussian coefficients.

        exp_arr : np.array
            The gaussian exponents.

        Returns
        -------
        np.array
            Updated gaussian coefficients for the next iteration.

        """
        gaussian = ma.asarray(self.get_model(coeff_arr, exp_arr))
        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            mol = self.get_mol_coord(i)
            new_coeff[i] *= self.get_inte_factor(exp_arr[i], gaussian, mol)
        return new_coeff / self.lagrange_multiplier

    def _get_norm_constant(self, exponent):
        r"""
        Normalization constant for a single gaussian function integrated over real space.
        """
        return (exponent / np.pi) ** (3./2.)

    def _update_func_params(self, coeff_arr, exp_arr, with_convergence=True):
        r"""
        Updates the function parameters based on the fixed point iteration method.

        For notes on how it is derived, please see the research paper for this software.

        Parameters
        ----------
        coeff_arr : np.array
            The gaussian coefficients.

        exp_arr : np.array
            The gaussian exponents.

        with_convergence : bool
            This boolean indicates if the coefficients already converged. If it is true, less
            computation is required by subbing in the lagrange multiplier.

        Returns
        -------
        np.ndarray(shape=(1, K)):
            Optimized exponents, where K is the number of gaussian parameters.

        """
        masked_normed_gaussian = np.ma.asarray(self.get_model(coeff_arr, exp_arr)).copy()

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            mol = self.get_mol_coord(i)
            if with_convergence:
                new_exps[i] = 3. * self._lagrange_multiplier
            else:
                new_exps[i] = 3. * self.get_inte_factor(exp_arr[i], masked_normed_gaussian, mol)
            integration = self.get_inte_factor(exp_arr[i], masked_normed_gaussian, mol, True)

            new_exps[i] /= (2. * integration)
        return new_exps

    def _update_coeffs(self, coeff_arr, exp_arr):
        new_coeff = self._update_coeffs_gauss(coeff_arr, exp_arr)
        return new_coeff, coeff_arr

    def _update_exps(self, coeff_arr, exp_arr):
        new_exps = self._update_func_params(coeff_arr, exp_arr)
        return new_exps, exp_arr

    def _get_deriv_coeffs(self, coeffs, fparams):
        pass

    def _get_deriv_fparams(self, coeffs, fparams):
        pass
