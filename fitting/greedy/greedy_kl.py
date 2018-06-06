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

"""

from fitting.kl_divergence.gaussian_kl import GaussianKullbackLeibler
from fitting.greedy.greedy_utils import get_next_choices
from fitting.greedy.greedy_strat import GreedyStrategy
import numpy as np

__all__ = ["GreedyKL"]


class GreedyKL(GreedyStrategy):
    r"""

    """
    def __init__(self, grid_obj, true_model, inte_val,
                 splitting_func=get_next_choices, eps_coeff=1e-3, eps_exp=1e-4,
                 factor=2):
        r"""

        Parameters
        ----------

        Returns
        -------

        """
        self.mbis_obj = GaussianKullbackLeibler(grid_obj, true_model, inte_val)
        self.splitting_func = splitting_func
        self.threshold_coeff = eps_coeff
        self.threshold_exp = eps_exp
        self.factor = factor
        self.successful = None
        super(GreedyKL, self).__init__()

    @property
    def true_model(self):
        return self.mbis_obj.true_model

    @property
    def inte_val(self):
        return self.mbis_obj.inte_val

    @property
    def grid_obj(self):
        return self.mbis_obj.grid_obj

    def get_model(self, params):
        return self.mbis_obj.get_model(params[:len(params)//2],
                                       params[len(params)//2:])

    def get_cost_function(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        model = self.mbis_obj.get_model(coeffs, exps)
        return self.mbis_obj.get_kullback_leibler(model)

    def get_best_one_function_solution(self):
        denom = self.grid_obj.integrate_spher(self.mbis_obj.ma_true_mod * np.power(
                self.mbis_obj.masked_grid_squared, 2.))
        exps = 3. * self.inte_val / (2. * 4. * np.pi * denom)
        return np.array([self.inte_val, exps])

    def get_next_iter_params(self, factor, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.splitting_func(factor, coeffs, exps)

    def get_optimization_routine(self, params, local=False):
        coeff_arr, exp_arr = params[:len(params)//2], params[len(params)//2:]
        if local:
            return self.mbis_obj.run(1e-2, 1e-3, coeff_arr, exp_arr, iprint=False)['x']
        return self.mbis_obj.run(self.threshold_coeff, self.threshold_exp, coeff_arr, exp_arr,
                                 iprint=False)['x']

    def get_errors_from_model(self, params):
        model = self.mbis_obj.get_model(params[:len(params) // 2],
                                        params[len(params)//2:])
        return self.mbis_obj.get_descriptors_of_model(model)
