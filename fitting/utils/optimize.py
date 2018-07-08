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
r"""Module responsible for optimizing functions catered for least-squares or Kullback-Leibler.

Functions
---------
- optimize_using_nnls : Optimize via nnls.
- optimize_using_nnls_valence : Optimize valence densities via nnla.
- optimize_using_slsqp : Optimize least-squares or Kullback-Leibler formula via SLSQP.
- optimize_using_l_bfgs : Optimize least-squares via L-BFGS method.

"""

from scipy.optimize import nnls, minimize, fmin_l_bfgs_b
import numpy as np
from fitting.kl_divergence.kull_leib_fitting import KullbackLeiblerFitting

__all__ = ["optimize_using_slsqp", "optimize_using_nnls", "optimize_using_l_bfgs",
           "optimize_using_nnls_valence"]


def optimize_using_nnls(true_dens, cofactor_matrix):
    r"""

    """
    b_vector = np.copy(true_dens)
    b_vector = np.ravel(b_vector)
    row_nnls_coefficients = nnls(cofactor_matrix, b_vector)
    return row_nnls_coefficients[0]


def optimize_using_nnls_valence(true_val_dens, cofactor_matrix):
    b_vector = np.copy(true_val_dens)
    b_vector = np.ravel(b_vector)
    row_nnls_coefficients = nnls(cofactor_matrix, b_vector)
    return row_nnls_coefficients[0]


def optimize_using_slsqp(density_model, initial_guess, bounds=None, *args):
    r"""Optimize the model via SLSQP.

    Works for Kullback-leibler class or DensityModel class.

    Parameters
    ----------
    density_model :

    initial_guess :

    bounds : optional

    *args : optional
        Additional arguments for the cost function defined from 'density_model'.

    Returns
    -------
    list :
        Updated parameters after optimization procedure.

    """
    const = None
    if isinstance(density_model, KullbackLeiblerFitting):
        const = {"eq": np.sum(initial_guess - density_model.norm)}

    if bounds is None:
        bounds = np.array([(0.0, np.inf)] * len(initial_guess))
    opts = {"maxiter": 100000000, "disp": True, "eps": 1e-10}
    f_min_slsqp = minimize(density_model.cost_function,
                           x0=initial_guess, method="SLSQP", bounds=bounds, args=(args),
                           jac=density_model.derivative_of_cost_function,
                           constraints=const, options=opts)
    parameters = f_min_slsqp['x']
    return parameters


def optimize_using_l_bfgs(density_model, initial_guess, bounds=None, *args):
    r"""Optimize the DensityModel class via L-BFGS.

    Parameters
    ----------
    density_model :

    initial_guess :

    bounds : optional

    Returns
    -------
    list :
        Updated parameters.

    """
    if bounds is None:
        bounds = np.array([(0.0, 1.7976931348623157e+308) for _ in range(0, len(initial_guess))],
                          dtype=np.float64)
    f_min_l_bfgs_b = fmin_l_bfgs_b(density_model.cost_function, x0=initial_guess,
                                   bounds=bounds, fprime=density_model.derivative_of_cost_function,
                                   maxfun=1500000, maxiter=1500000, factr=1e7, args=args,
                                   pgtol=1e-5)

    parameters = f_min_l_bfgs_b[0]
    return parameters
