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
Unified interfaces for gaussian basis-set fitting algorithms.

Functions
---------
fit_radial_densities : minimization of least squares or kullback-leibler div
                       using either techniques or a greedy algorithm.
"""

import os
import warnings
import numpy as np

from fitting.grid import BaseRadialGrid
from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler
from fitting.density import AtomicDensity
from fitting.model import DensityModel, GaussianBasisSet
from fitting.greedy.greedy_kl import GreedyKL
from fitting.utils.plotting_utils import plot_model_densities, plot_error
from fitting.utils.greedy_utils import get_next_choices
from fitting.greedy.greedy_lq import GreedyLeastSquares


__all__ = ["fit_gaussian_densities"]


def get_hydrogen_electron_density(grid, bohr_radius=1):
    return (1. / np.pi * (bohr_radius ** 3.)) * np.exp(-2. * grid / bohr_radius)


def fit_gaussian_densities(grid, element_name=None, true_model=None, inte_val=None,
                           method="kl-divergence", options=None, density_model=None,
                           ioutput=False, iplot=False):
    """
    Fits Gaussian Densities to a true model using variety of methods provided.

    Parameters
    ----------
    grid : arr or ClenshawGrid or BaseRadialGrid or HortonGrid, optional
          The radial grid points, if arr is provided it is stored as a
          BaseRadialGrid object.

    element_name : str, optional
                  The element that is being fitted to. If this is None, then
                  the true_model must be specified.

    true_model : arr, optional
                 True model to be fitted from.

    inte_val : int, optional
               Integration of the true model with 4 pi x^2 over the entire space.
               If it isn't specified, then it is estimated through the
               trapezoidal method.

    method : str or callable, optional
        Type of solver.  Should be one of
            - 'slsqp' :ref:`(see here) <kl_divergence.least_sqs.optimize_using_slsqp>`
            - 'l-bfgs'      :ref:`(see here) <kl_divergence.least_sqs.optimize_using_l_bfgs>`
            - 'nnls'          :ref:`(see here) <kl_divergence.least_sqs.optimize_using_nnls>`
            - 'greedy-ls-sqs'   :ref:'(see here) <greedy.greedy_lq.GreedyLeastSquares>'
            - 'kl_divergence'        :ref:`(see here) <kl_divergence.kull_leib_fitting>`
            - 'greedy-kl_divergence'   :ref:`(see here) <greedy.greedy_kl.GreedyKL>`

        If not given, chosen to be one of ``BFGS``, ``L-BFGS-B``, ``SLSQP``,
        depending if the problem has constraints or bounds.


    density_model : DensityModel, optional
        If one want's to provide their own model instead of our gaussian model.
        Then a DensityModel class has to be provided, where your model and cost
        function is stored for least squares. Note if done, then only
        the methods 'nnls', 'slsqp', l-bfgs would work.


    options : dict, optional
        - 'slsqp' - {bounds=(0, np.inf), initial_guess=custom(see *)}
        - 'l-bfgs' - {bounds=(0, np.inf), initial_guess=custom(see *)}
        - 'nnls' - {initial_guess=UGBS Exponents}
        - 'kl_divergence' - {threshold_coeff, threshold_exp, initial_guess, iprint=False}
        - 'greedy-kl_divergence' - {factor, max_number_of_functions, additional_funcs,
                           threshold_coeff, threshold_exp, splitting_func}
        - 'greedy-ls-sqs' - {factor, max_numb_of_funcs, additional_funcs,
                             splitting_func, threshold_coeff, threshold_exp}

        * initial guess is obtained by optimizing coefficients using NNLS using
        UGBS s-type exponents.

    iplots : boolean, optional

    ioutput : boolean, optional

    Returns
    -------


    Notes
    -----


    References
    ----------


    Examples
    --------
    See examples folder.
    """
    if options is None:
        options = {}

    full_names = {"be": "Beryllium", "c": "Carbon", "he": "Helium", "li": "Lithium",
                  'b': "boron", "n": "nitrogen", "o": "oxygen", "f": "fluoride",
                  "ne": "neon"}

    if element_name is not None:
        element_name = element_name.lower()

    if isinstance(grid, np.ndarray):
        grid = BaseRadialGrid(grid)

    # Sets Default Density To Atomic Slater
    current_file = os.path.abspath(os.path.dirname(__file__))
    if true_model is None:
        file_path = current_file + "/data/examples/" + element_name.lower()
        true_model = AtomicDensity(file_path, grid=grid.radii).electron_density

    # Sets Default Density Model to Gaussian Density
    if density_model is None:
        file_path = None
        if element_name is not None:
            slater_file_path = "/data/examples/" + element_name.lower()
            file_path = os.path.join(current_file, slater_file_path)
        density_model = GaussianBasisSet(grid.radii, true_model=true_model)

    # Exits If Custom Density Model is not inherited from density_model
    if not isinstance(density_model, DensityModel):
        raise TypeError("Custom Density Model should be inherited from "
                        "DensityModel from density_model.py")

    # Gives Warning if you wanted a custom density model to kl_divergence related procedures.
    if method in ["kl_divergence", "greedy-kl_divergence"] and density_model is not None:
        warnings.warn("Method %s does not use custom least_squares models. Rather it uses default "
                      "gaussian least_squares" % method, RuntimeWarning)

    if method in ['slsqp', 'l-bfgs']:
        options.setdefault('initial_guess', options["initial_guess"])

    # Sets Exponents needed for NNLS to S-type UGBS
    if method == "nnls":
        options.setdefault('initial_guess', options['exps'])

    # Set Default Arguments For Greedy
    if method in ['greedy-ls-sqs', 'greedy-kl_divergence']:
        options.setdefault('factor', 2.)
        options.setdefault('max_numb_funcs', 30)
        options.setdefault('backward_elim_funcs', None)
        s_func = get_next_choices
    if method == 'kl_divergence':
        options.setdefault('eps_coeff', 1e-3)
        options.setdefault('eps_fparam', 1e-4)
        options.setdefault('iprint', False)
        options.setdefault('coeffs', options['coeffs'])
        options.setdefault('fparams', options['fparams'])

    # Method Controller
    if method == "slsqp":
        params = optimize_using_slsqp(density_model, **options)
    elif method == "l-bfgs":
        params = optimize_using_l_bfgs(density_model, **options)
    elif method == "nnls":
        mat = density_model.create_cofactor_matrix(options['initial_guess'])
        params = optimize_using_nnls(mat)
    elif method == "kl_divergence":
        fit_obj = GaussianKullbackLeibler(grid, true_model, inte_val)
        result = fit_obj.run(**options)
        params = result['x']
        error = result['errors']
    elif method == "greedy-ls-sqs":
        fit_obj = GreedyLeastSquares(grid, true_model, inte_val=inte_val,
                                     splitting_func=s_func)
        if ioutput:
            params, params_it = fit_obj.__call__(ioutput=ioutput, **options)
            error = fit_obj.errors
            exit_info = fit_obj.exit_info
        else:
            params = fit_obj.__call__(ioutput=ioutput, **options)
    elif method == "greedy-kl_divergence":
        fit_obj = GreedyKL(grid, true_model, inte_val,
                           splitting_func=s_func)
        params, params_it, exit_info = fit_obj.__call__(ioutput=ioutput, **options)
        error = fit_obj.err_arr

    if iplot:
        # Change Grid To Angstrom
        g = grid.radii.copy() * 0.5291772082999999
        model = fit_obj.get_model(params)
        plot_model_densities(fit_obj.true_model, model, g,
                             title="Fitting Plot of " + full_names[element_name],
                             element_name=element_name,
                             figure_name="model_plot_using_" + method)
        models_it = []
        for p in params_it:
            c, e = p[:len(p)//2], p[len(p)//2:]
            models_it.append(fit_obj.mbis_obj.get_model(c, e))
        plot_model_densities(fit_obj.true_model, model, g,
                             title="Fitting Plot of " + full_names[element_name],
                             element_name=element_name,
                             figure_name="greedy_model_plot_using_" + method,
                             additional_models_plots=models_it)
        plot_error(error, element_name, "Different Error Measures On " + full_names[element_name],
                   figure_name="error_plot_using_" + method)

    if ioutput:
        dire = os.path.abspath(os.path.dirname(__file__))
        file_object = open(dire + '/arguments_' + method + ".txt", "w+")
        file_object.write("Method Used " + method + "\n")
        file_object.write("Number Of Basis FUnctions: " +
                          str(len(params)//2) + "\n")
        file_object.write("Final Parameters: " + str(params) + "\n")
        file_object.write("Iteration Parameters: " + str(params_it) + "\n")
        file_object.write(str(options) + "\n")
        file_object.write("Exit Information: " + str(exit_info) + "\n")
        file_object.write("Redudandance Info: " +
                          str(fit_obj.redudan_info_numb_basis_funcs))
        file_object.close()
        np.save(dire + "/parameters_" + method + ".npy", params)
        np.save(dire + "/parameters_" + method + "_iter.npy", params_it)

    return params
