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


import scipy
import numpy as np


__all__ = ["optimize_using_slsqp", "optimize_using_nnls", "optimize_using_l_bfgs",
           "optimize_using_nnls_valence"]


def optimize_using_nnls(true_dens, cofactor_matrix):
    b_vector = np.copy(true_dens)
    b_vector = np.ravel(b_vector)
    row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
    return row_nnls_coefficients[0]


def optimize_using_nnls_valence(true_val_dens, cofactor_matrix):
    b_vector = np.copy(true_val_dens)
    b_vector = np.ravel(b_vector)

    row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
    return row_nnls_coefficients[0]


def optimize_using_slsqp(density_model, initial_guess, bounds=None, *args):
    if bounds is None:
        bounds = np.array([(0.0, np.inf) for x in range(0, len(initial_guess))], dtype=np.float64)
    f_min_slsqp = scipy.optimize.minimize(density_model.cost_function,
                                          x0=initial_guess,
                                          method="SLSQP",
                                          bounds=bounds, args=(args),
                                          jac=density_model.derivative_of_cost_function)
    parameters = f_min_slsqp['x']
    return parameters


def optimize_using_l_bfgs(density_model, initial_guess, bounds=None, *args):
    if bounds is None:
        bounds = np.array([(0.0, 1.7976931348623157e+308) for x in range(0, len(initial_guess))],
                          dtype=np.float64)
    f_min_l_bfgs_b = scipy.optimize.fmin_l_bfgs_b(density_model.cost_function,
                                                  x0=initial_guess,
                                                  bounds=bounds,
                                                  fprime=density_model.derivative_of_cost_function,
                                                  maxfun=1500000,
                                                  maxiter=1500000,
                                                  factr=1e7,
                                                  args=args, pgtol=1e-5)

    parameters = f_min_l_bfgs_b[0]
    return parameters
