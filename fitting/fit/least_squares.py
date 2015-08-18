from fitting.density.atomic_slater_density import *
from fitting.gbasis.gbasis import UGBSBasis
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

#Ask Farnaz - The Atomic_slater_density uses an column vector, so should Density Model be column vector as well?
class DensityModel():
    def __init__(self, element_name, file_path, grid, exponents=[], change_exponents=False):
        assert isinstance(file_path, str)
        assert grid.ndim == 2 and np.shape(grid)[1] == 1

        self.grid = grid
        self.file_path = file_path
        self.electron_density = Atomic_Density(file_path, self.grid).atomic_density()

        if change_exponents:
            assert type(exponents).__module__ == np.__name__
            self.exponents = exponents
        else:
            gbasis =  UGBSBasis(element_name)
            self.exponents = 2.0 * gbasis.exponents()

    def model(self, coefficients, exponents):
        """
        Used this to compute the exponents and coefficients
        Exponents should be 1ndim list

        Shape (Number of Grid Points) Row Vector
        :param coefficients:
        :param exponents:
        :return:
        """
        assert exponents.ndim == 1
        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        assert np.shape(exponential)[0] == np.shape(self.grid)[0]
        assert np.shape(exponential)[1] == np.shape(exponents)[0]
        assert exponential.ndim == 2

        assert coefficients.ndim == 1
        assert len(coefficients) == np.shape(exponential)[1]
        gaussian_density = np.dot(exponential, coefficients) # is gaussian the right word?
        assert gaussian_density.ndim == 1
        assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]
        return(gaussian_density)

    def cofactor_matrix(self, exponents=[], change_exponents=False):
        """
        Params
        exponents is 1 dimensional array

        Returns e^(exponents * radius**2) Matrix
        Shape is (Number of Grids, Number of Exponents)
        """
        if change_exponents:
            exponential = np.exp(-2 * exponents * np.power(self.grid, 2.0))
        else:
            exponential = np.exp(-self.exponents * np.power(self.grid, 2.0))
        assert np.ndim(exponential) == 2
        return(exponential)

    def cost_function(self, coefficient, exponents=[], change_exponents=False):
        if not change_exponents:
            exponents = self.exponents
        assert type(coefficient).__module__ == np.__name__
        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        assert exponential.ndim == 2
        assert (coefficient.shape)[0] == exponential.shape[1]
        coefficient = (np.reshape(np.array(coefficient), (len(coefficient), 1)))
        gaussian_model = np.dot(exponential, coefficient)
        gaussian_model = np.ravel(gaussian_model)

        residual = np.ravel(self.electron_density) - gaussian_model
        residual_squared = np.power(residual, 2.0)

        return(np.sum(residual_squared))

    def derivative_cost_function(self, coefficient, exponents=[], change_exponents=False):
        if not change_exponents:
            exponents = self.exponents

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        coefficient = np.reshape(np.array(coefficient), (len(coefficient), 1))
        gaussian_model = np.dot(exponential, coefficient)
        gaussian_model = np.ravel(gaussian_model)
        residual = np.ravel(self.electron_density) - gaussian_model

        f_function = 2.0 * residual

        exp_derivitive = []
        for exp in exponents:
            g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
            derivative = f_function * np.ravel(g_function)
            exp_derivitive.append(np.ravel(derivative))

        assert exp_derivitive[0].ndim == 1
        assert np.shape(exp_derivitive[0])[0] == np.shape(self.grid)[0]
        return(np.sum(np.asarray(exp_derivitive), axis=1))

    def f_min_slsqp_coefficients(self, coeff_guess, exponents=[], change_exponents=False):
        if not change_exponents:
            exponents = self.exponents
        bounds = [(0, None) for x in range(0, len(coeff_guess))]

        f_min_slsqp = scipy.optimize.fmin_l_bfgs_b(self.cost_function, x0=coeff_guess, bounds=bounds, fprime=self.derivative_cost_function, args=(exponents, change_exponents), factr=10.0)
        coeff = f_min_slsqp[0]

        return(coeff)
        #grad = scipy.optimize.approx_fprime(list_initial_guess, self.cost_function, epsilon=1e-5)
        #print(f_min_slsqp, grad, self.derivative_cost_function(list_initial_guess))

    def nnls_coefficients(self, cofactor_matrix):
        b_vector = self.electron_density

        #b vector has to be one dimensional
        b_vector = np.ravel(b_vector)
        assert np.ndim(b_vector) == 1
        row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
        print(row_nnls_coefficients)
        return(row_nnls_coefficients[0])

    def integration(self, coefficients, exponents):
        # Integrate Based On Model or
        # Integrate Based on COefficients


        assert coefficients.ndim == 1
        assert exponents.ndim == 1
        electron_density = self.model(coefficients, exponents) #row vector
        integrate = np.trapz(np.ravel(self.grid**2) * np.ravel(electron_density), np.ravel(self.grid))
        return integrate

    def true_value(self, points):
        p, w = np.polynomial.laguerre.laggauss(points)
        p = np.reshape(p, (len(p), 1))
        w = np.reshape(w, (len(w), 1))
        be = Atomic_Density(self.file_path, p)

        a = be.atomic_density() * p**2 * w/ np.exp(-p)
        return(np.sum(a))

    def pauls_GA(self, factor, factor_factor, initial_guess, accuracy, maximum_exponents=50, use_slsqp=True, use_nnls=False):
        assert isinstance(maximum_exponents, int)
        assert isinstance(factor, float)
        assert isinstance(initial_guess, float) or all(isinstance(x, float) for x in initial_guess)

        def measure_error_helper(coeff, exponents):
            model = self.model(coeff, exponents)
            diff = np.absolute(np.ravel(self.electron_density) - model)
            grid = np.ravel(self.grid)
            integration = np.trapz(y=diff, x=grid)
            #integration = np.trapz(y=model, x=grid)
            #integration2 = np.trapz(y=np.ravel(self.electron_density), x=grid)
            return(integration)

        def best_UGBS_helper():
            counter = 0
            lowest_error = None
            best_UGBS = None
            for exponent in self.exponents:
                exponent = np.array([exponent])
                coefficients = self.f_min_slsqp_coefficients([initial_guess], exponents=exponent, change_exponents=True)
                error = measure_error_helper(coeff=coefficients, exponents=exponent)

                if counter == 0:
                    lowest_error = error
                    best_UGBS = exponent
                    counter += 1
                if error < lowest_error:
                    lowest_error = error
                    best_UGBS = exponent

            return(lowest_error, best_UGBS, coefficients)

        def splitting_array_helper(exp_array, factor):
            mult_copy = np.copy(exp_array)
            div_copy = np.copy(exp_array)
            next_position = 0
            for index in range(0, len(exp_array)):
                next_position += 1
                next_mult_number = exp_array[index] * factor
                next_div_number = exp_array[index] / factor

                mult_copy = np.insert(mult_copy, next_position, next_mult_number)
                div_copy = np.insert(div_copy, next_position, next_div_number)
                next_position += 1
            return(np.sort(mult_copy), np.sort(div_copy))

        def remove_zero(coeff_array, exp_array):
            new_coeff = np.copy(coeff)
            new_exp = np.copy(exp_array)
            for index, x in np.ndenumerate(coeff_array):
                if x == 0:
                    new_exp = np.delete(exp_array, index)
                    new_coeff = np.delete(coeff_array, index)
            return(new_coeff, new_exp)

        lowest_error, best_UGBS, coeff = best_UGBS_helper()
        #Ask Farnaz Regarding the x
        print("The Best UGBS Exponent is ", [x for x in best_UGBS], " that gave an error of ", lowest_error)
        if use_slsqp:
            while(lowest_error > accuracy):
                mult_exp, div_exp  = splitting_array_helper(best_UGBS, factor)
                factor *= factor_factor

                initial_guess = np.append(coeff, [coeff[len(coeff) - 1]* np.random.random() for x in range(0, len(coeff)) ])

                mult_coeff = self.f_min_slsqp_coefficients(initial_guess, exponents=mult_exp, change_exponents=True)
                div_coeff = self.f_min_slsqp_coefficients(initial_guess, exponents=div_exp, change_exponents=True)

                mult_error = measure_error_helper(mult_coeff, mult_exp)
                div_error = measure_error_helper(div_coeff, div_exp)

                if mult_error < div_error:
                    coeff = mult_coeff
                    best_UGBS = mult_exp
                    #print('mult', best_UGBS)

                else:
                    coeff = div_coeff
                    best_UGBS = div_exp
                    #print('div', best_UGBS)

                if mult_error < lowest_error:
                    lowest_error = mult_error
                    coeff = mult_coeff
                    best_UGBS = mult_exp

                elif div_error < lowest_error:
                    lowest_error = div_error
                    coeff = div_coeff
                    best_UGBS = div_exp
                #print(coeff)
                #coeff, best_UGBS = remove_zero(coeff, best_UGBS)
                print("factor: ", np.round(factor, 3), ", mult error:", np.round(mult_error, 3), ", div error:", np.round(div_error, 3), ", number of exp/coeff: ", len(best_UGBS) )
                if len(best_UGBS) > maximum_exponents and len(coeff) > maximum_exponents:
                    break;

                #print(mult_coeff)


        return(lowest_error, coeff, best_UGBS)

    def greedy_algorithm(self, step_size, step_size_factor, initial_guess, accuracy, maximum_exponents=50 , use_nnls=False, use_slsqp=False):
        assert isinstance(maximum_exponents, int)
        assert isinstance(step_size, float)
        assert isinstance(initial_guess, float) or all(isinstance(x, float) for x in initial_guess)

        exponents_array = np.asarray(initial_guess) if isinstance(initial_guess, list) else np.asarray([initial_guess])
        exponents_array_2 = np.asarray(initial_guess) if isinstance(initial_guess, list) else np.asarray([initial_guess])

        true_value = self.true_value(186)

        def helper_splitting_array(pos_array, neg_array, step_size, step_size_factor):
            """
            E.g.
                Input : np.array([0.5, 1.0, 2.0, 3]) step size = 0.5, step size factor = 0.98
                Result: (array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ,  3.5]), array([ 0.5,  1. ,  1.5,  2. ,  2.5,  3. ]), 0.49)
            """

            # Point of this is because the pos_Array is changing
            # Need a copy to add the numbers properly
            pos_copy = np.copy(pos_array)
            neg_copy = np.copy(neg_array)

            for index in range(0, len(pos_array)):
                next_number = pos_copy[index] + step_size
                if next_number > 0 and next_number not in pos_array:
                    pos_array = np.sort(np.insert(pos_array, index + 1, next_number))

            #Point of second loop is in case it goes towards negatives
            # Farnaz, Redundancy of same numbers
            for index in range(0, len(neg_array)):
                next_number = neg_copy[index] - step_size

                if next_number > 0 and next_number not in neg_array:
                    neg_array = np.sort(np.insert(neg_array, index + 1, next_number))

            step_size = step_size * step_size_factor
            return(pos_array, neg_array, step_size)

        def optimized_then_integrate(exponents_array):
            cofactor_matrix = self.cofactor_matrix(exponents_array, change_exponents=True)
            try:
                row_nnls_coefficients = self.nnls_coefficients(cofactor_matrix)
            except:
                pass
            integration = self.integration(row_nnls_coefficients, exponents_array)

            return(cofactor_matrix, row_nnls_coefficients, integration)

        if use_nnls:
            mat_cofactor_matrix = self.cofactor_matrix(exponents_array, change_exponents=True)
            assert mat_cofactor_matrix.ndim == 2

            row_nnls_coefficients = self.nnls_coefficients(mat_cofactor_matrix)
            assert row_nnls_coefficients.ndim == 1

            integration = self.integration(row_nnls_coefficients,  exponents_array)

            pos_error, diff_pos_error = None, None
            neg_error, diff_neg_error = None, None

            while np.absolute(true_value - integration) > accuracy:
                #Split Exponents
                exponents_array, exponents_array_2, step_size = helper_splitting_array(exponents_array, exponents_array_2, step_size, step_size_factor)

                pos_cofactor_matrix, pos_row_nnls_coefficients, pos_integration = optimized_then_integrate(exponents_array)
                neg_cofactor_matrix, neg_row_nnls_coefficients, neg_integration = optimized_then_integrate(exponents_array_2)

                #Calc Integration Error
                pos_error = np.absolute(true_value - pos_integration)
                neg_error = np.absolute(true_value - neg_integration)

                #Calc Difference Error
                diff_pos_error = np.sum(np.absolute(np.ravel(self.electron_density) - self.model(pos_row_nnls_coefficients, exponents_array)))
                diff_neg_error = np.sum(np.absolute(np.ravel(self.electron_density) - self.model(neg_row_nnls_coefficients, exponents_array_2)))
                new = self.model(neg_row_nnls_coefficients, exponents_array_2)
                reshape = np.reshape(new, (np.shape(new)[0], 1))
                np.set_printoptions(threshold=np.nan)


                #Compare Error
                if neg_error > pos_error:
                    integration = neg_integration
                    exponents_array_2 = np.copy(exponents_array)
                    #print("pos error", pos_error, "pos integrate", pos_integration, "true_value", true_value,"size", np.shape(exponents_array))
                else:
                    integration = pos_integration
                    exponents_array = np.copy(exponents_array_2)
                    #print("neg error", neg_error, "neg integrate", neg_integration, "true_value", true_value, "size", np.shape(exponents_array))

                #Maximum list For Condition
                if np.shape(exponents_array)[0] > maximum_exponents:
                    print("\n Maximum Shape is Reached Return Results")
                    print("neg error:", neg_error,", diff neg error:", diff_neg_error, "neg integrate:",  neg_integration,
                          "true_value:", true_value, "size:", np.shape(exponents_array), "Initial_Guess", initial_guess)
                    print("pos error:", pos_error,", diff pos error:", diff_pos_error, "pos integrate:", pos_integration,
                          "true_value:", true_value,"size:", np.shape(exponents_array), "Step Size is ", step_size)

                    return(exponents_array, neg_error, pos_error, diff_neg_error, diff_pos_error)

            return(exponents_array, neg_error, pos_error, diff_neg_error, diff_pos_error)

    def evolutionary_algorithm_one_exp(self, initial_guess, step_size_factor, accuracy, maximum_exponents=50, exponents_reducer=False):
        #assert type(initial_guess) is float
        integration_error = 0.1


        difference_error = 0.1
        best_exponents_array = None
        counter = 0;
        step_size = 5.0 #Based the step_size on the atomic number
        change_step_size = accuracy /accuracy
        plt.ion()
        plt.show()
        while(integration_error > accuracy and difference_error > accuracy):

            #step_size = initial_guess[0] * np.random.random()
            try:
                greedy_algo =  self.greedy_algorithm(step_size, step_size_factor, initial_guess, accuracy, maximum_exponents=maximum_exponents, use_nnls=True)

                exponents_array, int_neg_error, int_pos_error, diff_neg_error, diff_pos_error = greedy_algo

                if int_neg_error < int_pos_error and int_neg_error < integration_error:

                    integration_error = int_neg_error
                    best_exponents_array = exponents_array
                    initial_guess = exponents_array[ np.shape(exponents_array)[0] / 2]

                elif int_pos_error < int_neg_error and int_pos_error < integration_error:
                    integration_error = int_pos_error
                    best_exponents_array = exponents_array
                    initial_guess = exponents_array[ np.shape(exponents_array)[0] / 2]

                else:
                    print(np.absolute(int_neg_error - integration_error))
                    if np.absolute(int_neg_error - integration_error) < change_step_size:
                        print("It Converged")
                        step_size /= 10
                        change_step_size /= 10

                    initial_guess = exponents_array[ np.shape(exponents_array)[0] / 2]


    #                if counter == 100:
    #                    initial_guess = best_exponents_array[int(np.random.randint(0, np.shape(best_exponents_array)[0]))]
    #                    counter = 0

                cofactor = self.cofactor_matrix(exponents_array, change_exponents=True)
                coeff = self.nnls_coefficients(cofactor)
                model = self.model(coeff, exponents_array)
                #plt.clf()
                plt.semilogy(self.grid, model)
                plt.semilogy(self.grid, self.electron_density, 'r')
                plt.draw()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                #print(repr(ex), 2)

                pass
            counter += 1
            print(counter ,": Number of Ilterations")
        def exponent_reducer(best_exponents_array):
            def optimized_then_integrate(exponents_array):
                cofactor_matrix = self.cofactor_matrix(exponents_array, change_exponents=True)
                try:
                    row_nnls_coefficients = self.nnls_coefficients(cofactor_matrix)
                except:
                    pass
                integration = self.integration(row_nnls_coefficients, exponents_array)

                return(cofactor_matrix, row_nnls_coefficients, integration)

            true_value = be.true_value(186)
            integration = None
            while len(best_exponents_array) != 1:
                mid_point = len(best_exponents_array)/2
                first_half = best_exponents_array[0:mid_point]
                second_half = best_exponents_array[mid_point:]
                cofactor_first, first_coeff, first_inte = optimized_then_integrate(first_half)
                cofactor_sec, sec_coeff, sec_inte = optimized_then_integrate(second_half)
                first_diff = np.absolute(first_inte - true_value)
                sec_diff = np.absolute(sec_inte - true_value)
                if first_diff < sec_diff:
                    best_exponents_array = first_half
                    integration = first_inte
                elif sec_diff < first_diff:
                    best_exponents_array= second_half
                    integration = sec_inte
            return(best_exponents_array, integration)

        if exponents_reducer == True:
            best_exponents_array, integration = exponent_reducer(best_exponents_array)
            integration_error = np.absolute(self.true_value(186) - integration)

        return(integration_error, difference_error, best_exponents_array)

    def evolutionary_algorithm_list(self, initial_guess, step_size_factor, accuracy, maximum_exponents=50, exponents_reducer=False):
        #assert type(initial_guess) is float
        integration_error = 0.1

        size_initial_guess = len(initial_guess)
        difference_error = 10000.0
        best_exponents_array = None
        iterations_counter = 0;
        step_size = 11.0 #Based the step_size on the atomic number
        change_step_size = accuracy /accuracy
        plt.ion()
        plt.show()
        no_change_counter = 0;

        def exponent_reducer(best_exponents_array):
            def optimized_then_integrate(exponents_array):
                cofactor_matrix = self.cofactor_matrix(exponents_array, change_exponents=True)
                try:
                    row_nnls_coefficients = self.nnls_coefficients(cofactor_matrix)
                    model = self.model(row_nnls_coefficients, exponents_array)

                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                return(cofactor_matrix, row_nnls_coefficients, model)

            difference = None
            while len(best_exponents_array) > size_initial_guess:
                mid_point = len(best_exponents_array)/2

                first_half = best_exponents_array[0:mid_point]
                second_half = best_exponents_array[mid_point:]

                cofactor_first, first_coeff, first_model = optimized_then_integrate(first_half)
                cofactor_sec, sec_coeff, sec_model = optimized_then_integrate(second_half)

                first_diff = np.sum(np.absolute(first_model - np.ravel(be.electron_density)))
                sec_diff = np.sum(np.absolute(sec_model - np.ravel(be.electron_density)))

                if first_diff < sec_diff:
                    best_exponents_array = first_half
                    difference = first_model
                elif sec_diff < first_diff:
                    best_exponents_array= second_half
                    difference = sec_model
            return(best_exponents_array, difference)

        while(integration_error > accuracy and difference_error > accuracy):

            #step_size = initial_guess[0] * np.random.random()
            try:
                greedy_algo =  self.greedy_algorithm(step_size, step_size_factor, initial_guess, accuracy, maximum_exponents=maximum_exponents, use_nnls=True)

                exponents_array, int_neg_error, int_pos_error, diff_neg_error, diff_pos_error = greedy_algo

                if diff_neg_error < diff_pos_error and diff_neg_error < difference_error:

                    difference_error = diff_neg_error
                    best_exponents_array = exponents_array
                    print('neg won')
                    no_change_counter = 0
                    best_exponents_array, int = exponent_reducer(exponents_array)
                    initial_guess = np.squeeze(np.random.choice(np.ravel(best_exponents_array), size_initial_guess ))
                    initial_guess = initial_guess.tolist()
                    #print(best_exponents_array)


                elif diff_pos_error < diff_neg_error and diff_pos_error < difference_error:
                    difference_error = diff_pos_error
                    best_exponents_array = exponents_array
                    print('pos won')
                    no_change_counter = 0
                    best_exponents_array, int = exponent_reducer(exponents_array)
                    initial_guess = np.squeeze(np.random.choice(np.ravel(best_exponents_array), size_initial_guess ))
                    initial_guess = initial_guess.tolist()
                    #print(best_exponents_array)

                else:
                    print('neither', np.absolute(int_neg_error - integration_error))


                    no_change_counter += 1
                    if no_change_counter >= 40 or step_size < 1e-12:
                        print("Resteat")
                        size_initial_guess += 1
                        step_size == 500.0
                        best_exponents_array = exponent_reducer(exponents_array)
                        print(best_exponents_array)
                        initial_guess = np.squeeze(np.random.choice(np.ravel(best_exponents_array), size_initial_guess ))
                        initial_guess = initial_guess.tolist()
                        initial_guess = [np.random.random() * 10] + initial_guess
                        no_change_counter = 0
                    else:
                        initial_guess = np.squeeze(np.random.choice(np.ravel(best_exponents_array), size_initial_guess ))
                        initial_guess = initial_guess.tolist()

                    if no_change_counter >= 10:
                        step_size/=10









                cofactor = self.cofactor_matrix(exponents_array, change_exponents=True)
                coeff = self.nnls_coefficients(cofactor)
                model = self.model(coeff, exponents_array)
                plt.clf()
                plt.semilogx(self.grid, np.ravel(self.grid**2) * np.ravel(model))
                plt.semilogx(self.grid, np.ravel(self.grid**2) * np.ravel(self.electron_density), 'r')
                plt.draw()
            except Exception as ex:
                import traceback
                traceback.print_exc()
                #print(repr(ex), 2)
                pass

            iterations_counter += 1
            print(iterations_counter ,": Number of Ilterations")


        return(integration_error, difference_error, best_exponents_array)

file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater"
from fitting.density.radial_grid import *
radial_grid = Radial_Grid(4)
row_grid_points = radial_grid.grid_points(200, 300, [50, 75, 100])

column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
be = DensityModel('be', file_path, column_grid_points)
#print("Lowest Error", be.pauls_GA(8.0, 0.95, initial_guess=1.0, accuracy=1e-5 ,maximum_exponents=50)[0])

lowest_error, coeff, exponents = be.pauls_GA(4.0, 0.75, initial_guess=1.0, accuracy=1e-5 ,maximum_exponents=25)

print("lowest error", lowest_error, "\n coeff", coeff)

if lowest_error < 0.5:
    model = be.model(coeff, exponents)
    plt.plot(be.grid, model)
    plt.plot(be.grid, np.ravel(be.electron_density))
    plt.show()
    print("Integration: ", np.trapz(y=model, x=np.ravel(be.grid)))



def plot_atomic_desnity(radial_grid, density_list, title, figure_name):
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
    plt.xlim(0, 7.5)
    plt.ylim(ymin=1e-8)
    plt.xlabel('Distance from the nucleus [A]')
    plt.ylabel('Log(density [Bohr**-3])')
    plt.title(title)
    plt.legend(loc=0)
    plt.savefig(figure_name)
    plt.close()

import sys
sys.exit()