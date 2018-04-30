r"""

"""

from abc import ABCMeta, abstractmethod
import numpy as np
from fitting.greedy.greedy_utils import check_redundancies

__all__ = ["GreedyStrategy"]


class GreedyStrategy(object):
    r"""

    """
    __metaclass__ = ABCMeta

    def __init__(self):
        r"""

        """
        self.errors = None
        self.redudan_info = []

    @abstractmethod
    def get_cost_function(self):
        raise NotImplementedError()

    @abstractmethod
    def get_best_one_function_solution(self):
        raise NotImplementedError()

    @abstractmethod
    def get_next_iter_params(self):
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_routine(self, *args, local=False):
        raise NotImplementedError()

    @abstractmethod
    def get_errors_from_model(self, params):
        pass

    def store_errors(self, params):
        err = self.get_errors_from_model(params)
        if self.errors is None:
            self.errors = [[x] for x in err]
        else:
            for i, x in enumerate(err):
                self.errors[i].append(x)

    def _find_best_lparams(self, param_list):
        best_local_value = 1e10
        best_local_param = None
        for param in param_list:
            local_param = self.get_optimization_routine(param, local=True)
            cost_func = self.get_cost_function(local_param)
            if cost_func < best_local_value:
                best_local_value = self.get_cost_function(local_param)
                best_local_param = local_param
        return best_local_value, best_local_param

    def __call__(self, factor, max_numb_funcs=30, backward_elim_funcs=None,
                 add_choice_funcs=None, ioutput=False):
        # Initialize all the variables
        gparams = self.get_best_one_function_solution()
        numb_one_func_params = len(gparams)
        self.store_errors(gparams)
        exit_info = ""
        numb_funcs = len(gparams) / numb_one_func_params
        prev_gval = 0
        best_gval = 1e10
        params_iter = [gparams]
        numb_redum = 0
        factor0 = factor

        # Start the greedy algorithm
        while numb_funcs < max_numb_funcs - 1 and \
              np.abs(best_gval - prev_gval) >= 1e-5 and \
              numb_redum < 5:
            choices_parameters = self.get_next_iter_params(factor, gparams)
            if callable(add_choice_funcs):
                # When you want extra choices to be added.
                # ie fitting the left-over density.
                choices_parameters.append(add_choice_funcs(gparams))

            best_lval, best_lparam = self._find_best_lparams(choices_parameters)

            numb_funcs = len(gparams) / numb_one_func_params
            coeffs, fparams = check_redundancies(gparams[:len(gparams)//2],
                                                 gparams[len(gparams)//2:])
            # If Redudancies Found
            if numb_funcs != len(coeffs):
                new_params = np.append(coeffs, fparams)
                redudan_obj = self.get_cost_function(new_params)
                # If Removing Redundancies gave a better answer.
                if redudan_obj < best_gval:
                    gparams = np.append(coeffs, fparams)
                    self.redudan_info.append([numb_funcs, len(coeffs),
                                              "Gave Better Answer"])
                    numb_funcs = len(coeffs)
                    prev_gval, best_gval = best_gval, redudan_obj
                    self.store_errors(gparams)
                    params_iter.append(gparams)
                else:
                    self.redudan_info.append([numb_funcs, len(coeffs),
                                              "Go Back a Iteration with new factor"])
                    numb_funcs -= 1
                numb_redum += 1
                factor += 5

            # Best local choice is best so far.
            elif best_lval <= best_gval:
                prev_gval, best_gval = best_gval, best_lval
                gparams = self.get_optimization_routine(best_lparam, local=False)
                self.store_errors(gparams)
                params_iter.append(gparams)
                numb_redum = 0
                factor = factor0  # Reset Factor.

            else:
                exit_info = "Next Iteration Did Not Find The Best Choice"
                break
            if backward_elim_funcs is not None:
                gparams = backward_elim_funcs(best_lparam)

        # "Next Iteration Did Not Find THe Best Choice":
        if exit_info is None:
            exit_info = self._final_exit_info(numb_funcs, max_numb_funcs,
                                              best_gval, prev_gval, numb_redum)
        return gparams, params_iter, exit_info

    def _final_exit_info(self, num_func, max_func, best_val, prev_gval, redum):
        if not num_func < max_func - 1:
            exit_info = "Max number of functions reached,  " + \
                             str(num_func)
        elif not np.abs(best_val - prev_gval) >= 1e-5:
            exit_info = "Cost function is less than some epsilon " + \
                             str(best_val - prev_gval) + " <= " + str(1e-5)
        elif not redum < 5:
            exit_info = " Number of redudancies " + str(redum) + \
                             " found in a row is more than 5, "
        return exit_info
