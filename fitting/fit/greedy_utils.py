from abc import ABCMeta, abstractmethod
from fitting.fit.mbis_total_density import TotalMBIS
import numpy as np


def get_next_possible_coeffs_and_exps(factor, coeffs, exps):
    size = exps.shape[0]
    all_choices_of_exponents = []
    all_choices_of_coeffs = []
    all_choices_of_parameters = []
    coeff_value = 100.
    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        #all_choices_of_exponents.append(exponent_array)
        #all_choices_of_coeffs.append(coefficient_array)
        all_choices_of_parameters.append(np.append(coefficient_array, exponent_array))
        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            #all_choices_of_exponents.append(exponent_array)
            #all_choices_of_coeffs.append()
            all_choices_of_parameters.append(np.append(np.append(coeffs, np.array([coeff_value])), exponent_array))
    return all_choices_of_parameters #all_choices_of_coeffs, all_choices_of_exponents

class _GreedyStrategy(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.errors = None

    @abstractmethod
    def get_cost_function(self):
        raise NotImplementedError()

    @abstractmethod
    def get_best_one_function_solution(self):
        raise NotImplementedError()

    @abstractmethod
    def get_next_iteration_of_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_routine(self):
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

    def run_greedy(self, factor, max_number_of_functions=30, additional_funcs=None,
                   ioutput=False):
        global_parameters = self.get_best_one_function_solution()
        numb_one_func_params = len(global_parameters)
        # Should Be One
        number_of_functions = len(global_parameters) / numb_one_func_params
        best_global_value = 1e10
        while number_of_functions < max_number_of_functions:
            choices_of_parameters = self.get_next_iteration_of_variables(factor, global_parameters)

            best_local_value = 1e10
            best_local_param = None
            for param in choices_of_parameters:
                local_param = self.get_optimization_routine(param)
                cost_func = self.get_cost_function(local_param)
                if cost_func < best_local_value:
                    best_local_value = self.get_cost_function(local_param)
                    best_local_param = local_param
            if best_local_value <= best_global_value:
                best_global_value = best_local_value
                global_parameters = self.get_optimization_routine(best_local_param)
                self.store_errors(global_parameters)
            else:
                print("Next Iteration Did Not Find THe Best Choice")
                import sys
                sys.exit()

            if additional_funcs is not None:
                global_parameters = additional_funcs(best_local_param)
            number_of_functions = len(global_parameters) / numb_one_func_params
        if ioutput:
            return global_parameters, self.errors
        return global_parameters


class GreedyMBIS(_GreedyStrategy):
    def __init__(self, element_name, atomic_number, grid_obj, electron_density,
                 splitting_func=get_next_possible_coeffs_and_exps, threshold_coeff=1e-3,
                 threshold_exps=1e-4, factor=2):
        self.atomic_number = atomic_number
        self.grid_obj = grid_obj
        self.mbis_obj = TotalMBIS(element_name, atomic_number, grid_obj, electron_density)
        self.splitting_func = splitting_func
        self.threshold_coeff = threshold_coeff
        self.threshold_exp = threshold_exps
        self.factor = factor
        super(GreedyMBIS, self).__init__()

    def get_cost_function(self, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        model = self.mbis_obj.get_normalized_gaussian_density(coeffs, exps)
        return self.mbis_obj.get_objective_function(model)

    def get_best_one_function_solution(self):
        denom = 4. * np.pi * self.grid_obj.integrate(self.mbis_obj.masked_electron_density
                                                     * np.power(self.mbis_obj.masked_grid_squared, 2.))
        exps = 3. * self.atomic_number / (2. * denom)
        return np.array([np.float(self.atomic_number), exps])

    def get_next_iteration_of_variables(self, factor, params):
        coeffs, exps = params[:len(params)//2], params[len(params)//2:]
        return self.splitting_func(factor, coeffs, exps)

    def get_optimization_routine(self, params):
        coeff_arr, exp_arr = params[:len(params)//2], params[len(params)//2:]
        return self.mbis_obj.run(self.threshold_coeff, self.threshold_exp, coeff_arr, exp_arr, iprint=True)

    def get_errors_from_model(self, params):
        model = self.mbis_obj.get_normalized_gaussian_density(params[:len(params)//2],
                                                              params[len(params)//2:])
        return self.mbis_obj.get_descriptors_of_model(model)



class GreedyLeastSquares(_GreedyStrategy):
    def __init__(self):
        pass

    @abstractmethod
    def get_cost_function(self):
        raise NotImplementedError()

    @abstractmethod
    def get_best_one_function_solution(self):
        raise NotImplementedError()

    @abstractmethod
    def get_next_iteration_of_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_routine(self):
        raise NotImplementedError()

class GreedyGeneral(_GreedyStrategy):
    def __init__(self):
        pass

    @abstractmethod
    def get_cost_function(self):
        raise NotImplementedError()

    @abstractmethod
    def get_best_one_function_solution(self):
        raise NotImplementedError()

    @abstractmethod
    def get_next_iteration_of_variables(self):
        raise NotImplementedError()

    @abstractmethod
    def get_optimization_routine(self):
        raise NotImplementedError()

def check_redundancies(coeffs, exps):
    new_coeffs = coeffs.copy()
    new_exps = exps.copy()

    indexes_where_they_are_same = []
    for i, alpha in enumerate(exps):
        similar_indexes = []
        for j in range(i + 1, len(exps)):
            if j not in similar_indexes:
                if np.abs(alpha - exps[j]) < 1e-2:
                    if i not in similar_indexes:
                        similar_indexes.append(i)
                    similar_indexes.append(j)
        if len(similar_indexes) != 0:
            indexes_where_they_are_same.append(similar_indexes)

    for group_of_similar_items in indexes_where_they_are_same:
        for i in  range(1, len(group_of_similar_items)):
            new_coeffs[group_of_similar_items[0]] += coeffs[group_of_similar_items[i]]

    if len(indexes_where_they_are_same) != 0:
        print("-------- Redundancies found ---------")
        print()
        new_exps = np.delete(new_exps, [y for x in indexes_where_they_are_same for y in x[1:]])
        new_coeffs = np.delete(new_coeffs, [y for x in indexes_where_they_are_same for y in x[1:]])
    assert len(exps) == len(coeffs)
    return new_coeffs, new_exps



def get_two_next_possible_coeffs_and_exps(factor, coeffs, exps):
    size = len(exps)
    all_choices_of_coeffs = []
    all_choices_of_exps = []
    coeff_value = 100.

    for index, exp in np.ndenumerate(exps):
        if index[0] == 0:
            exponent_array = np.insert(exps, index, exp / factor)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        elif index[0] <= size:
            exponent_array = np.insert(exps, index, (exps[index[0] - 1] + exps[index[0]]) / 2)
            coefficient_array = np.insert(coeffs, index, coeff_value)
        params = get_next_possible_coeffs_and_exps(factor, coefficient_array, exponent_array)
        all_choices_of_coeffs.extend(coeff2)
        all_choices_of_exps.extend(exp2)

        if index[0] == size - 1:
            exponent_array = np.append(exps, np.array([exp * factor]))
            coeff2, exp2 = get_next_possible_coeffs_and_exps(factor, np.append(coeffs, np.array([coeff_value])), exponent_array)
            all_choices_of_coeffs.extend(coeff2)
            all_choices_of_exps.extend(exp2)
    return all_choices_of_coeffs, all_choices_of_exps

if __name__ == "__main__":
    #print(get_two_next_possible_coeffs_and_exps(2., np.array([1., 2.]), np.array([0.5, 1.5]))[0])
    #print(get_two_next_possible_coeffs_and_exps(2., np.array([1., 2.]), np.array([0.5, 1.5]))[1])

    from fitting.radial_grid.radial_grid import HortonGrid
    import horton
    rtf = horton.ExpRTransform(1.0e-30, 25, 1000)
    radial_grid_2 = horton.RadialGrid(rtf)
    from fitting.radial_grid.radial_grid import HortonGrid
    radial_grid = HortonGrid(1e-80, 25, 1000)
    import os
    file_path = os.path.expanduser('~') + r"/PythonProjects/fitting/fitting/data/examples" + "/" + "be"
    from fitting.density.slater_density.atomic_slater_density import Atomic_Density
    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    greedy_mbis = GreedyMBIS("be", 4, radial_grid, atomic_density.electron_density)
    greedy_mbis.run_greedy(factor=2.)