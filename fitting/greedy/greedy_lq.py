r"""

"""

from fitting.greedy.greedy_strat import GreedyStrategy
from fitting.greedy.greedy_utils import get_next_choices
from fitting.least_squares.gaussian_density import GaussianBasisSet
from fitting.least_squares.least_sqs import optimize_using_nnls, optimize_using_slsqp
import numpy as np

__all__ = ["GreedyLeastSquares"]


class GreedyLeastSquares(GreedyStrategy):
    def __init__(self, grid_obj, true_model, splitting_func=get_next_choices,
                 factor=2):
        self.gauss_obj = GaussianBasisSet(grid_obj.radii, true_model)
        self.grid_obj = grid_obj
        self.factor = factor
        self.splitting_func = splitting_func
        super(GreedyStrategy, self).__init__()

    @property
    def true_model(self):
        return self.gauss_obj.electron_density

    @property
    def ugbs(self):
        return self.gauss_obj.UGBS_s_exponents

    def get_cost_function(self, params):
        self.gauss_obj.cost_function(params)

    def _solve_one_function_weight(self, weight):
        a = 2.0 * np.sum(weight)
        sum_of_grid_squared = np.sum(weight * np.power(self.grid_obj.radii, 2))
        b = 2.0 * sum_of_grid_squared
        sum_ln_electron_density = np.sum(weight * np.log(self.true_model))
        c = 2.0 * sum_ln_electron_density
        d = b
        e = 2.0 * np.sum(weight * np.power(self.grid_obj.radii, 4))
        f = 2.0 * np.sum(weight * np.power(self.grid_obj.radii, 2) *
                         np.log(self.true_model))
        big_a = (b * f - c * e) / (b * d - a * e)
        big_b = (a * f - c * d) / (a * e - b * d)
        coefficient = np.exp(big_a)
        exponent = - big_b
        return np.array([coefficient, exponent])

    def get_best_one_function_solution(self):
        # Minimizing weighted least squares with three different weights
        weight1 = np.ones(len(self.grid_obj.radii))
        weight3 = np.power(self.true_model, 2.)
        p1 = self._solve_one_function_weight(weight1)
        cost_func1 = self.gauss_obj.cost_function(p1)

        p2 = self._solve_one_function_weight(self.true_model)
        cost_func2 = self.gauss_obj.cost_function(p2)

        p3 = self._solve_one_function_weight(weight3)
        cost_func3 = self.gauss_obj.cost_function(p3)

        p_min = min([(cost_func1, p1), (cost_func2, p2), (cost_func3, p3)],
                    key=lambda t: t[0])

        # Minimize by analytically finding coefficient.
        val = 1e10
        if self.gauss_obj.element is not None:
            exp_choice1 = self.gauss_obj.generation_of_UGBS_exponents(1.25,
                                                                      self.ugbs)
            exp_choice2 = self.gauss_obj.generation_of_UGBS_exponents(1.5,
                                                                      self.ugbs)
            exp_choice3 = self.gauss_obj.generation_of_UGBS_exponents(1.75,
                                                                      self.ugbs)
            grid_squared = self.grid_obj.radii**2.
            best_found = None
            for exp in np.append((exp_choice1, exp_choice2, exp_choice3)):
                num = np.sum(self.true_model * np.exp(-exp * grid_squared))
                den = np.sum(np.exp(-2. * exp * grid_squared))
                c = num / den
                p = np.array([c, exp])
                p = self.get_optimization_routine(p)
                cost_func = self.gauss_obj.cost_function(p)
                if cost_func < val:
                    val = cost_func
                    best_found = p

        if p_min[0] < val:
            return p_min[1]
        return best_found

    def get_next_iter_params(self, params):
        return self.splitting_func(self.factor, params[:len(params)//2],
                                   params[len(params)//2:])

    def get_optimization_routine(self, params):
        exps = params[len(params)//2:]
        cofac_matrix = self.gauss_obj.create_cofactor_matrix(exps)
        coeffs = optimize_using_nnls(self.true_model, cofac_matrix)

        p = np.append(coeffs, exps)
        params = optimize_using_slsqp(self.gauss_obj, p)
        return params

    def get_errors_from_model(self, params):
        model = self.gauss_obj.create_model(params)
        err1 = self.gauss_obj.integrate_model_trapz(model)
        err2 = self.gauss_obj.get_integration_error(self.true_model, model)
        err3 = self.gauss_obj.get_error_diffuse(self.true_model, model)
        return [err1, err2, err3]
