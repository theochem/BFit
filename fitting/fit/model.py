import abc
from fitting.density.atomic_slater_density import *
from fitting.gbasis.gbasis import UGBSBasis

class DensityModel():
    __metaclass__ = abc.ABCMeta

    def __init__(self, element, grid, file_path):
        #TODO ADD
        self.check_type(element, str)
        self.check_type(file_path, str)
        self.check_type(grid, "numpy array")
        assert grid.ndim == 2

        self.element = element.lower()
        self.grid = np.copy(grid)
        self.file_path = file_path

        atomic_density_object = Atomic_Density(file_path, self.grid)
        self.electron_density = np.copy(atomic_density_object.atomic_density())
        self.integrated_total_electron_density = atomic_density_object.integrate_total_density_using_trapz()

        self.electron_density_core, self.electron_density_valence = atomic_density_object.atomic_density_core_valence()
        self.integrated_core_density = self.integrate_model_using_trapz(self.electron_density_core)
        self.integrated_valence_density = self.integrate_model_using_trapz(self.electron_density_valence)

        gbasis =  UGBSBasis(element)
        self.UGBS_s_exponents = 2.0 * gbasis.exponents('s')
        self.UGBS_p_exponents = 2.0 * gbasis.exponents('p')
        #if array is empty initialize it
        if(self.UGBS_p_exponents.size == 0.0):
            self.UGBS_p_exponents = np.copy(self.UGBS_s_exponents)
        self.check_type(self.UGBS_s_exponents, "numpy array")

    @abc.abstractmethod
    def create_model(self):
        """
        TODO
        Insert Documentation about the model
        """
        raise NotImplementedError("Please Implement Your Model")

    @abc.abstractmethod
    def cost_function(self):
        """
        TODO
        """
        raise NotImplementedError("Please Implement Your Cost Function")

    @abc.abstractmethod
    def derivative_of_cost_function(self):
        """
        TODO
        """
        raise NotImplementedError("Please Implement Your Derivative of Cost Function")

    @abc.abstractmethod
    def create_cofactor_matrix(self):
        pass

    def calculate_residual(self, *args):
        residual = np.ravel(self.electron_density) - self.create_model(*args)
        return(residual)

    def calculate_residual_based_on_core(self, *args):
        residual = np.ravel(self.electron_density_core) - self.create_model(*args)
        return residual

    def calculate_residual_based_on_valence(self, *args):
        residual = np.ravel(self.electron_density_valence) - self.create_model(*args)
        return residual

    def integrate_model_using_trapz(self, approximate_model):
        integrate = np.trapz(y=np.ravel(self.grid**2) * np.ravel(approximate_model), x=np.ravel(self.grid))
        return integrate

    def measure_error_by_integration_of_difference(self, true_model, approximate_model):
        error = np.trapz(y=np.ravel(self.grid**2) * (np.absolute(np.ravel(true_model) - np.ravel(approximate_model))), x=np.ravel(self.grid))
        return error

    def measure_error_by_difference_of_integration(self, true_model, approximate_model):
        integration_of_true_model = self.integrated_total_electron_density
        integration_of_approx_model = self.integrate_model_using_trapz(approximate_model)

        difference_of_models = integration_of_true_model - integration_of_approx_model

        return np.absolute(difference_of_models)

    def generation_of_UGBS_exponents(self, p, UGBS_exponents):
        max_number_of_UGBS = np.amax(UGBS_exponents)
        min_number_of_UGBS = np.amin(UGBS_exponents)

        def calculate_number_of_Gaussian_functions(p, max, min):
            num_of_basis_functions = np.log(2 * max / min) / np.log(p)
            return num_of_basis_functions

        num_of_basis_functions = calculate_number_of_Gaussian_functions(p, max_number_of_UGBS, min_number_of_UGBS)
        num_of_basis_functions = num_of_basis_functions.astype(int)

        new_Gaussian_exponents = np.array([min_number_of_UGBS])
        for n in range(1, num_of_basis_functions + 1):
            next_exponent = min_number_of_UGBS * np.power(p, n)
            new_Gaussian_exponents = np.append(new_Gaussian_exponents, next_exponent)

        return new_Gaussian_exponents

    @staticmethod
    def check_type(object, type_of_object):
        if(type_of_object == "numpy array"):
            assert type(object).__module__ == np.__name__
        else:
            assert isinstance(object, type_of_object)

    @staticmethod
    def check_dimensions(array, dimension):
        assert isinstance(dimension, int)
        assert array.ndim == dimension

    @staticmethod
    def plot_atomic_density(radial_grid, density_list, title, figure_name):
        #Density List should be in the form
        # [(electron density, legend reference),(model1, legend reference), ..]
        import matplotlib.pyplot as plt
        colors = ["#FF00FF", "#FF0000", "#FFAA00", "#00AA00", "#00AAFF", "#0000FF", "#777777", "#00AA00", "#00AAFF"]
        ls_list = ['-', ':', ':', '-.', '-.', '--', '--', ':', ':']
        assert isinstance(density_list, list)
        radial_grid *= 0.5291772082999999   #convert a.u. to angstrom
        for i, item in enumerate(density_list):
            dens, label = item
            # plot with log scaling on the y axis
            plt.semilogy(radial_grid, dens, lw=3, label=label, color=colors[i], ls=ls_list[i])

        #plt.xlim(0, 25.0*0.5291772082999999)
        plt.xlim(0, 9)
        plt.ylim(ymin=1e-9)
        plt.xlabel('Distance from the nucleus [A]')
        plt.ylabel('Log(density [Bohr**-3])')
        plt.title(title)
        plt.legend()
        plt.savefig(figure_name)
        plt.close()


class Fitting():
    #__metaclass__ = abc.ABCMeta
    #TODO Ask farnaz, Should this class include cost function, derivative of cost funciton and cofactor matrix?

    def __init__(self, model_object):
        assert isinstance(model_object, DensityModel)
        self.model_object = model_object

    def optimize_using_nnls(self, cofactor_matrix):
        b_vector = np.copy(self.model_object.electron_density)
        b_vector = np.ravel(b_vector)
        assert np.ndim(b_vector) == 1

        row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
        return(row_nnls_coefficients[0])

    def optimize_using_slsqp(self, initial_guess, *args):
        bounds = np.array([(0.0, np.inf) for x in range(0, len(initial_guess))], dtype=np.float64)

        f_min_slsqp = scipy.optimize.fmin_slsqp(self.model_object.cost_function, x0=initial_guess, bounds=bounds, fprime=self.model_object.derivative_of_cost_function,
                                                acc =1e-06,iter=15000, args=(args), full_output=True, iprint=-1)
        parameters = f_min_slsqp[0]
        #print(f_min_slsqp[4])
        return(parameters)

    def optimize_using_l_bfgs(self, initial_guess, *args, iprint=False):
        bounds = np.array([(0.0, 1.7976931348623157e+308) for x in range(0, len(initial_guess))], dtype=np.float64)
        #fprime=self.model_object.derivative_of_cost_function
        f_min_l_bfgs_b = scipy.optimize.fmin_l_bfgs_b(self.model_object.cost_function, x0=initial_guess, bounds=bounds, fprime=self.model_object.derivative_of_cost_function
                                                  ,maxfun=1500000, maxiter=1500000, factr=1e7, args=args, pgtol=1e-5)
        if iprint:
            print(f_min_l_bfgs_b[2]['warnflag'], "            ", f_min_l_bfgs_b[2]['task'])

        #if f_min_l_bfgs_b[2]['warnflag'] != 0:
        #        print(f_min_l_bfgs_b[2]['task'])

        parameters = f_min_l_bfgs_b[0]
        return(parameters)







    def find_best_parameter_from_analytical_coeff_and_generated_exponents(self, *args, p=1.5, optimization_algo=optimize_using_l_bfgs):
        generated_UGBS_exponents = self.model_object.generation_of_UGBS_exponents(p, self.model_object.UGBS_s_exponents)

        list_of_parameters = []
        for exponent in generated_UGBS_exponents:
            coefficient = self.analytically_find_coefficient(self.model_object.electron_density, exponent)
            parameters = np.array([coefficient, exponent])
            parameters = optimization_algo(self, parameters, *args)
            #print(exponent, parameters)
            list_of_parameters.append(parameters)

        best_parameters_found, error = self.find_best_parameter_from_list(list_of_parameters, *args)

        return best_parameters_found

    def analytically_find_coefficient(self, electron_density, exponent):
        def one_exponent_model(factor, exponent):
            exponential = np.exp(-factor * exponent * np.power(self.model_object.grid, 2.0))
            return exponential

        exponential = one_exponent_model(1.0, exponent)
        exponential =  np.multiply(np.ravel(electron_density) ,np.ravel(exponential))
        exponential2 = one_exponent_model(2.0, exponent)

        coefficient = np.sum(exponential) / np.sum(exponential2)
        if coefficient < 0.0:
            print("anaytical coefficient is negative ")
            coefficient = 0.0

        return coefficient

    def analytically_solve_objective_function(self, electron_density, weight):
        if  type(weight).__module__ == np.__name__ : assert weight.ndim == 1

        grid = np.copy(np.ravel(self.model_object.grid))
        grid_squared = np.copy(np.power(grid, 2.0))
        ln_of_electron_density = np.ma.log(np.ravel(electron_density))
        #print(np.isinf(ln_of_electron_density))
        if(np.isnan(ln_of_electron_density).any()):
            print("ISNAN in the electron density")
        a = 2.0 * np.sum(weight)
        b = 2.0 * np.sum(weight * grid_squared)
        c = 2.0 * weight * np.ravel(ln_of_electron_density)
        #c.set_fill_value(0.0)
        c = c.sum()
        if(np.isnan(c)):
            print("C IS THE ISNAN")
        d = np.copy(b)
        e = 2.0 * np.sum(weight * np.power(grid, 4.0))
        f = 2.0 * weight * grid_squared * ln_of_electron_density
        #f.set_fill_value( 0.0)
        f = f.sum()
        if(np.isinf(f)):
            print("F IS THE ISNAN")

        #print(a, b, c, d, e, f)
        A = (b * f - c * e) / (b * d - a * e)
        B = (a * f - c * d) / (a * e - b * d)
        #print(a, f, c, d, weight * grid_squared, "##ISNAN##")
        coefficient = np.exp(A)
        exponent = -B
        if exponent < 0:
            exponent = 0.0
        if(np.isnan(exponent)):
            print("ISNAN DETECTGED FROM OBJECTIVE FUNCTIOn")
        return(np.array([coefficient, exponent]))

    def find_best_parameter_from_list(self, list_of_choices, *args):
        lowest_error_found = 1e10
        best_parameter_found = None

        for choice in list_of_choices:
            model = self.model_object.create_model(choice, *args)
            error_measured = self.model_object.measure_error_by_integration_of_difference(self.model_object.electron_density, model)
            if(error_measured < lowest_error_found):
                lowest_error_found = error_measured
                best_parameter_found = np.copy(choice)

        return best_parameter_found, lowest_error_found



    def forward_greedy_algorithm(self, factor, desired_accuracy, chosen_electron_density, optimization_algo=optimize_using_l_bfgs, maximum_num_of_functions=100, *args):
        assert type(factor) is float
        assert type(desired_accuracy) is float
        assert type(maximum_num_of_functions) is int

        def next_list_of_exponents(exponents_array, factor):
            assert exponents_array.ndim == 1
            size = exponents_array.shape[0]
            all_choices_of_exponents = []

            for index, exp in np.ndenumerate(exponents_array):
                if index[0] == 0:
                    exponent_array = np.insert(exponents_array, index, exp / factor )

                elif index[0] <= size:
                    exponent_array = np.insert(exponents_array, index, (exponents_array[index[0] - 1] + exponents_array[index[0]])/2)
                all_choices_of_exponents.append(exponent_array)

                if index[0] == size - 1:
                    exponent_array = np.append(exponents_array, np.array([ exp * factor] ))
                    all_choices_of_exponents.append(exponent_array)
            return(all_choices_of_exponents)

        def removeZeroFromParameters(parameters, num_of_functions):
            coeff = parameters[:num_of_functions]
            exponents = parameters[num_of_functions:]

            indexes_where_zero_exists = np.nonzero(coeff < 1.0e-6)

            coeff = np.delete(coeff, indexes_where_zero_exists)
            exponents = np.delete(exponents, indexes_where_zero_exists)
            parameters = np.concatenate((coeff, exponents))
            return parameters

        def optimization_routine(exponents, num_of_functions, iprint=False):
            cofactor_matrix = self.model_object.create_cofactor_matrix(exponents)
            optimized_coefficients = self.optimize_using_nnls(cofactor_matrix)

            initial_guess_of_parameters = np.concatenate((optimized_coefficients, exponents))
            optimized_parameters_slsqp = self.optimize_using_slsqp(initial_guess_of_parameters, num_of_functions)
            optimized_parameters_bfgs = self.optimize_using_l_bfgs(initial_guess_of_parameters, num_of_functions)

            cost_function_slsqp = self.model_object.cost_function(optimized_parameters_slsqp, num_of_functions)
            cost_function_bfgs = self.model_object.cost_function(optimized_parameters_bfgs, num_of_functions)

            cost_function_value = None
            best_parameters_from_technique = None
            if cost_function_slsqp < cost_function_bfgs:
                cost_function_value = cost_function_slsqp
                best_parameters_from_technique = optimized_parameters_slsqp
            else:
                cost_function_value = cost_function_bfgs
                best_parameters_from_technique = optimized_parameters_bfgs

            if iprint:
                print(exponents)
                print("Optimized Coeff(NNLS) with initial exponent, resp", optimized_coefficients, exponents)
                print("Initial Guess of Parameters", initial_guess_of_parameters)
                print("Optimized Parameters SLSQP/BFGS, resp", optimized_parameters_slsqp, optimized_parameters_bfgs)
                print("Cost Function Value SLSQP/BFGS, resp", cost_function_slsqp, cost_function_bfgs, "\n")

            return cost_function_value, best_parameters_from_technique

        WEIGHTS = [np.ones(np.shape(self.model_object.grid)[0]),
                   np.ravel(chosen_electron_density),
                   np.ravel(np.power(chosen_electron_density, 2.0))]


        # Selected Best Single Gaussian Function
        best_generated_UGBS_parameter_with_analytical_coefficient = self.find_best_parameter_from_analytical_coeff_and_generated_exponents(1)
        list_of_initial_one_function_choices = [best_generated_UGBS_parameter_with_analytical_coefficient]

        for weight in WEIGHTS:
            best_analytical_parameters = self.analytically_solve_objective_function(chosen_electron_density, weight)
            list_of_initial_one_function_choices.append(best_analytical_parameters)

        best_cost_function = 1e20
        best_parameters_found = None
        for choice in list_of_initial_one_function_choices:
            exponents = choice[1:]
            cost_function_value, parameters = optimization_routine(exponents, 1, iprint=False)

            if cost_function_value < best_cost_function:
                best_cost_function = cost_function_value
                best_parameters_found = np.copy(parameters)


        # Iterate To Find the Next N+1 Gaussian Function
        local_parameter = np.copy(best_parameters_found)
        global_best_parameters = None
        counter = 1
        number_of_functions = 1
        electron_density = np.ravel(chosen_electron_density) - self.model_object.create_model(local_parameter, 1)
        print("\nStart While Loop")
        while(best_cost_function > desired_accuracy and number_of_functions < maximum_num_of_functions):
            next_list_of_choices_for_exponents = next_list_of_exponents(local_parameter[number_of_functions:], factor)

            for weight in WEIGHTS:
                best_analytical_parameters = self.analytically_solve_objective_function(electron_density, weight)
                next_choice_of_exponent = np.append(local_parameter[number_of_functions:], best_analytical_parameters[1:])
                if(np.isnan(next_choice_of_exponent).any()):
                    print("ISNAN IS DEECTED IN GREEDY ALGO")
                next_list_of_choices_for_exponents.append(next_choice_of_exponent)

            number_of_functions += 1
            local_best_cost_function_value = 1e20
            local_best_parameters = None
            for choice_of_exponents in next_list_of_choices_for_exponents:
                cost_function_value, parameters = optimization_routine(choice_of_exponents, number_of_functions)

                if cost_function_value < local_best_cost_function_value:
                    local_best_cost_function_value = cost_function_value
                    local_best_parameters = np.copy(parameters)

            #if True in (np.absolute(local_best_parameters) < 1.0e-6) and False:
            #    print("ZERO IS LOCAAAAAATEDDDDDDDDDDD")
            #    local_best_parameters = removeZeroFromParameters(local_best_parameters, number_of_functions)
            #    number_of_functions = np.shape(local_best_parameters)[0]//2


            if local_best_cost_function_value < best_cost_function:
                best_cost_function = local_best_cost_function_value
                global_best_parameters = local_best_parameters

            model = self.model_object.create_model(local_best_parameters, number_of_functions)
            integrate_error = self.model_object.measure_error_by_integration_of_difference(chosen_electron_density, model)
            integration = self.model_object.integrate_model_using_trapz(model)
            local_parameter = np.copy(local_best_parameters)
            electron_density -= model

            #density_list_for_graph = [(self.model_object.electron_density, "True Density"),
            #                                      (model, "Model Density,d=" + str(integrate_error))]
            #title = str(counter) + "_" + self.model_object.element + " Density, " + "\n d=Integrate(|True - Approx| Densities) " \
            #        ", Num Of Functions: " + str((number_of_functions))
            #self.model_object.plot_atomic_density(np.copy(self.model_object.grid), density_list_for_graph, title, self.model_object.element + "_" + str(counter))

            #if np.where(local_best_parameters == 0.0) == True:
            #    local_best_parameters = removeZeroFromParameters(local_best_parameters, number_of_functions)
            #    number_of_functions = np.shape(local_best_parameters)[0]

            print("best found", best_cost_function, "integration", integration, "integrate error", integrate_error)
            print("Counter", counter, "number of functions", number_of_functions, "\n")
            counter += 1
            #break
        return global_best_parameters, best_cost_function

    def solve_for_single_coefficients(self, electron_density, alpha_s, alpha_p):
        def repeated_sum(a):
            sum = 0
            for x in range(0, np.shape(np.ravel(self.model_object.grid))[0]):
                sum += a
            return sum
        grid_squared = np.power(np.ravel(self.model_object.grid), 2.0)
        sum_electron_density = np.sum(np.ravel(electron_density))
        a = np.ravel(np.exp(-alpha_s * grid_squared))
        sum_a = np.sum(a)
        exponential_p = np.exp(-alpha_p * grid_squared)
        b = grid_squared * exponential_p
        d = np.power(b, 2.0)
        sum_top = np.sum(np.ravel(electron_density * np.power(b, 2.0)))
        sum_d = np.sum(d)

        sum_ab = np.sum(a * np.power(b, 2.0))
        c_1 = (sum_electron_density - repeated_sum(sum_top/sum_d)) / (sum_a + repeated_sum(sum_ab/sum_d))
        d_1 = np.sum((np.ravel(electron_density) - c_1 * a)*b) / sum_d
        if(c_1 < 0):
            c_1 = 0
        if(d_1 < 0):
            d_1 = 0
        #print(c_1, d_1)
        return(c_1, d_1)

    def solve_weighted_function(self, electron_density, weight):
        sum_weight = np.sum(weight)
        grid_squared = np.power(np.ravel(self.model_object.grid), 2.0)
        grid_quad = np.power(np.ravel(self.model_object.grid), 4.0)
        sum_weight_grid_squared = - np.sum(weight * grid_squared)
        sum_weight_grid_quad = - np.sum(weight * grid_quad)

        matrix_A = np.array([[sum_weight, sum_weight_grid_squared, sum_weight, sum_weight_grid_squared],
                             [-sum_weight_grid_squared, sum_weight_grid_quad, -sum_weight_grid_squared, sum_weight_grid_quad],
                             [0, 0, 0, 0],
                             [0, 0, 0, 0]])

        ln_electron_density = (np.ma.log(np.ravel(electron_density)))
        ln_grid_squared = np.ravel(np.ma.log(np.ravel(grid_quad)))
        ln_grid_weight = np.sum(weight * ln_grid_squared)
        ln_weight = (ln_electron_density * weight)
        ln_weight_squared = ln_electron_density * weight * grid_squared
        matrix_B = np.array([[np.sum(ln_weight) - np.sum(ln_grid_weight)],
                            [np.sum(ln_weight_squared) - np.sum(ln_grid_weight)], [0], [0]])

        parameters = np.linalg.lstsq(matrix_A, matrix_B)
        print(parameters)
        parameters = np.ravel(parameters[0])
        A, B, C, D = parameters
        print(A, B, C, D)
        c_1 = np.exp(A); d_1 = np.exp(C)
        return(c_1, B, d_1, D)

    def forward_greedy_algorithm_valence(self, factor, desired_accuracy, optimizer=optimize_using_l_bfgs, maximum_num_of_functions=100, *args):
        UGBS_s = self.model_object.generation_of_UGBS_exponents(1.25, self.model_object.UGBS_s_exponents)
        UGBS_p = self.model_object.generation_of_UGBS_exponents(1.25, self.model_object.UGBS_p_exponents)

        cost_function_error = 1e10
        best_found_parameters = None
        for x in range(0, np.shape(UGBS_s)[0]):
            for c in range(0, np.shape(UGBS_p)[0]):
                cofactor_matrix = self.model_object.create_cofactor_matrix(UGBS_s[x], UGBS_p[c])
                optimized_coeff = self.optimize_using_nnls(cofactor_matrix)

                initial_guess_for_next_optimizier = np.concatenate(([optimized_coeff[0]], [UGBS_s[x]], [optimized_coeff[1]], [UGBS_p[c]]), axis=1)
                optimized_parameters = self.optimize_using_l_bfgs(initial_guess_for_next_optimizier, 1, 1)

                local_cost_function_error = self.model_object.cost_function(optimized_parameters, 1, 1)
                if local_cost_function_error < cost_function_error:
                    cost_function_error = local_cost_function_error
                    best_found_parameters = optimized_parameters

        def next_list_of_choices(parameters, factor):
            assert parameters.ndim == 1
            size = parameters.shape[0]
            #assert num <= size/2
            all_choices_of_exponents = []

            for index, exp in np.ndenumerate(parameters):
                if index[0] == 0:
                    exponent_array = np.insert(parameters, index, exp / (factor ))

                elif index[0] <= size:
                    exponent_array = np.insert(parameters, index, (parameters[index[0] - 1] + parameters[index[0]])/2)
                all_choices_of_exponents.append(exponent_array)

                if index[0] == size - 1:
                    exponent_array = np.append(parameters, np.array([ exp * factor] ))
                    all_choices_of_exponents.append(exponent_array)
            return(all_choices_of_exponents)


        number_of_functions = best_found_parameters.shape[0]/4
        num_of_s_funcs = 1
        num_of_p_funcs = 1
        global_best_parameters = None
        counter = 1;
        while desired_accuracy < cost_function_error and number_of_functions <= maximum_num_of_functions:
            s_coefficients = best_found_parameters[:num_of_s_funcs]
            s_exponents = best_found_parameters[num_of_s_funcs:2 * num_of_s_funcs]

            p_coefficients = best_found_parameters[num_of_s_funcs * 2 :num_of_s_funcs * 2 + num_of_p_funcs]
            p_exponents = best_found_parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

            next_s_exponents = next_list_of_choices(s_exponents, factor)
            next_p_exponents = next_list_of_choices(p_exponents, factor)

            how_many_s_funcs_added = 0
            how_many_p_funcs_added = 0
            local_best_parameter = None;
            local_cost_function_error = 1e10;
            for choice_s_exponents in next_s_exponents:
                cofactor_matrix = self.model_object.create_cofactor_matrix(choice_s_exponents, p_exponents)
                optimized_coeff = self.optimize_using_nnls(cofactor_matrix)

                initial_guess_for_next_optimizier = np.concatenate((optimized_coeff[0:num_of_s_funcs + 1],
                                                                    choice_s_exponents,
                                                                    optimized_coeff[num_of_s_funcs + 1:],
                                                                    p_exponents), axis=1)
                optimized_parameters = self.optimize_using_l_bfgs(initial_guess_for_next_optimizier, num_of_s_funcs + 1, num_of_p_funcs)

                current_cost_function_error = self.model_object.cost_function(optimized_parameters, num_of_s_funcs + 1, num_of_p_funcs)

                if current_cost_function_error < local_cost_function_error:
                    how_many_s_funcs_added = 1
                    local_cost_function_error = current_cost_function_error
                    local_best_parameter = optimized_parameters

            for choice_p_exponents in next_p_exponents:
                cofactor_matrix = self.model_object.create_cofactor_matrix(s_exponents, choice_p_exponents)
                optimized_coeff = self.optimize_using_nnls(cofactor_matrix)

                initial_guess_for_next_optimizier = np.concatenate((optimized_coeff[0:num_of_s_funcs],
                                                                    s_exponents,
                                                                    optimized_coeff[num_of_s_funcs:],
                                                                    choice_p_exponents), axis=1)
                optimized_parameters = self.optimize_using_l_bfgs(initial_guess_for_next_optimizier, num_of_s_funcs, num_of_p_funcs + 1)

                current_cost_function_error = self.model_object.cost_function(optimized_parameters, num_of_s_funcs, num_of_p_funcs + 1)
                if current_cost_function_error < local_cost_function_error:
                    how_many_p_funcs_added = 1
                    how_many_s_funcs_added = 0
                    local_cost_function_error = current_cost_function_error
                    local_best_parameter = optimized_parameters

            if local_cost_function_error < cost_function_error:
                cost_function_error = local_cost_function_error
                best_found_parameters = local_best_parameter
                global_best_parameters = np.copy(best_found_parameters)
            else:
                best_found_parameters = local_best_parameter
            if how_many_s_funcs_added >= 1:
                num_of_s_funcs += 1
            elif how_many_p_funcs_added >= 1:
                num_of_p_funcs += 1

            model = self.model_object.create_model(best_found_parameters, num_of_s_funcs, num_of_p_funcs)
            integrate_error = self.model_object.measure_error_by_integration_of_difference(self.model_object.electron_density_valence, model)
            print(num_of_s_funcs, num_of_p_funcs, cost_function_error, integrate_error, self.model_object.integrate_model_using_trapz(model))

            density_list_for_graph = [(self.model_object.electron_density_valence, "True Density"),
                                                  (model, "Model Density,d=" + str(integrate_error))]
            title = str(1) + "_" + self.model_object.element + " Density, " + "\n d=Integrate(|True - Approx| Densities) " \
                    ", Num Of  S Functions: " + str(num_of_s_funcs) + ", Num Of P Funcs: " + str(num_of_p_funcs)
            self.model_object.plot_atomic_density(np.copy(self.model_object.grid), density_list_for_graph, title, self.model_object.element + "_" + str(counter))
            counter += 1;
            #if np.where(local_best_parameters == 0.0) == True:
            #    local_best_parameters = removeZeroFromParameters(local_best_parameters, number_of_functions)
            #    number_of_functions = np.shape(local_best_parameters)[0]
            """
            for i in range(0, len(next_s_exponents)):
                next_s_exponents[i] = np.concatenate((next_s_exponents[i], p_exponents))
            for i in range(0, len(next_p_exponents)):
                next_p_exponents[i] = np.concatenate((s_exponents, next_p_exponents[i]))
            next_exponents = next_s_exponents + next_p_exponents
            """



    def optimize_KL_l_bfgs(self, initial_guess, *args, iprint=False):
        bounds = np.array([(0.0, 1.7976931348623157e+308) for x in range(0, len(initial_guess))], dtype=np.float64)

        f_min_l_bfgs_b = scipy.optimize.fmin_l_bfgs_b(self.model_object.cost_function, x0=initial_guess, bounds=bounds, approx_grad=True
                                                  ,maxfun=1500000, maxiter=1500000, factr=1e7, args=args, pgtol=1e-5)
        if iprint:
            print(f_min_l_bfgs_b[2]['warnflag'], "            ", f_min_l_bfgs_b[2]['task'])

        #if f_min_l_bfgs_b[2]['warnflag'] != 0:
        #        print(f_min_l_bfgs_b[2]['task'])

        parameters = f_min_l_bfgs_b[0]
        return(parameters)
