from fitting.density.atomic_slater_density import *
from fitting.gbasis.gbasis import UGBSBasis
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

#Ask Farnaz - The Atomic_slater_density uses an column vector, so should Density Model be column vector as well?
class DensityModel():
    def __init__(self, element_name, file_path, grid, change_exponents=False):
        assert isinstance(file_path, str)
        assert grid.ndim == 2 and np.shape(grid)[1] == 1

        self.grid = grid
        self.file_path = file_path
        self.electron_density = Atomic_Density(file_path, self.grid).atomic_density()

        if change_exponents:
            pass
        else:
            gbasis =  UGBSBasis(element_name)
            self.exponents = 2.0 * gbasis.exponents()


    def model(self, coefficients, exponents):
        """
        Used this to computer the exponents and coefficients
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

    def cost_function(self, coefficient):
        cofactor_matrix = self.cofactor_matrix()

        gaussian_model = coefficient * cofactor_matrix
        gaussian_model = np.sum(gaussian_model, axis=1)

        residual = np.ravel(self.electron_density) - gaussian_model
        assert residual.ndim == 1
        residual_squared = np.power(residual, 2.0)

        return residual_squared

    def derivitive_cost_function(self):
        pass

    def f_min_slsqp_coefficients(self, list_initial_guess, ):
        bounds1=[(0, None) for x in range(0, len(self.exponents))]
        derivitive = self.derivitive_cost_function()

        #f_min_slsqp = scipy.optimize.fmin_slsqp(self.cost_function, list_initial_guess, bounds=bounds, fprime_eqcons=self.derivitive_cost_function())
        SLSQP = scipy.optimize.minimize(self.cost_function, x0=list_initial_guess, method='SLSQP', jac=True, bounds=bounds1)
        print(SLSQP)

    def nnls_coefficients(self, cofactor_matrix):
        b_vector = self.electron_density

        #b vector has to be one dimensional
        b_vector = np.ravel(b_vector)
        assert np.ndim(b_vector) == 1
        row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)

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


x0 = np.linspace(0, 1, num = len(be.exponents))
coeffs = be.nnls_coefficients( be.cofactor_matrix())

integration_error, difference_error, best_exponents_array = be.evolutionary_algorithm_one_exp(initial_guess=[5.0, 200.0, 500.0, 300.0, 200.0, 100.0, 300.0] , step_size_factor=0.98, accuracy=1e-10, exponents_reducer=True)

cofactor = be.cofactor_matrix(best_exponents_array, change_exponents=True)
coef = be.nnls_coefficients(cofactor)

np.set_printoptions(threshold=np.nan)

model = be.model(coef, best_exponents_array)
print("Best Exponent Array:", best_exponents_array)
print("Integration_Error:", integration_error)
print("Final Integration:", be.integration(coef, best_exponents_array))
#print(np.concatenate((np.reshape(model, (len(model), 1)), be.electron_density), axis=1))
#plt.plot(be.grid, (model), 'r')
#plt.plot(be.grid, np.sort(be.electron_density), 'g')
#plt.show()

density_list = [[np.reshape(model, (len(model), 1)),"Model"], [be.electron_density,"Electron Density"]]
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

plot_atomic_desnity(be.grid, density_list, "Hey", "Yo")
import sys
sys.exit()

r"""
c =exp = np.array([  0.04690497 ,  0.08862487 , 11.52134028  ,11.52438861,  11.52786841,
  11.53160574 , 11.53543023 , 11.53918749 , 11.54274849 , 11.54601524,
  11.54892286 , 11.5514382  , 11.55355592 , 11.55529294 , 11.55668213,
  11.55776611 , 11.55859181 , 11.55920608 , 11.55965256 , 11.55996971,
  11.56018995 , 11.5603395  , 11.56043881 , 11.56050332 , 11.56054432,
  11.56056982 , 11.56058533 , 11.56059457 , 11.56059995 , 11.56060303,
  11.56060474 , 11.56060568 , 11.56060672 , 11.56060698,  11.56060711,
  11.56060725])

cofactor= be.cofactor_matrix(c, change_exponents=True)
nnls_coeff = be.nnls_coefficients(cofactor)

model = be.model(nnls_coeff, exp)
print(np.shape(be.electron_density), np.shape(model))
print(np.sum(np.absolute(np.ravel(be.electron_density) - model)))
plt.plot(be.electron_density, be.grid)
plt.plot(model, be.grid)
plt.show()"""









r"""

#############################################################################################################################
########################################## FMIN_L_BFGS_B ####################################################################
#############################################################################################################################
be_UGBS = np.concatenate(( [0.02*1.95819475**25, 0.02*1.95819475**26, 0.02*1.95819475**27, 0.02*1.95819475**28, 0.02*1.95819475**29,
                0.02*1.95819475**30, 0.02*1.95819475**31], be_UGBS))
length_UGBS = np.shape(be_UGBS)[0]

def total_density_cost_function(x, *args):
    UGBS, grid, rho = args[0], args[1], args[2]
    gaussian = x * np.exp(np.longdouble(-1.9999999999 * UGBS * grid**2)) # 503 x 25
    sum_gaussians = np.sum(gaussian, axis=1)
    sum_gaussians = np.reshape(sum_gaussians, (len(sum_gaussians), 1))
    residual = rho - sum_gaussians
    sum_squared_residuals = np.sum(residual**2)
    return sum_squared_residuals
print(total_density_cost_function(0, be_UGBS, Be_grid, be_rho) == np.sum(be_rho**2 ))

# Initial Guess
x0 = np.linspace(0, 1, num = length_UGBS)
#This Is The Optimization that is Used
optimized = scipy.optimize.fmin_l_bfgs_b(total_density_cost_function, x0, args=(be_UGBS, Be_grid, be_rho),  bounds=[(0, None) for x in range(0, length_UGBS)], approx_grad=True )
coeff = optimized[0]

# Another Grid that takes small increments towards 10
grid = np.arange(0.0, 10.0, 0.0001)
def f(opt_coeffs, grid):
    grid = np.reshape(grid, (len(grid), 1))  #100000 x 1
    print(np.shape(coeff * np.exp(be_UGBS * -2 * grid**2)))    #(100000 x 25)
    return np.sum(coeff * np.exp(be_UGBS * -2 * grid**2), axis = 1)

density_approx = f(coeff, Be_grid)   #opt_coeffs refers to the optimized coefficients
# turn the density approximate into 1D array, if yours is 2D array (i.e. column array)
density_approx = np.ravel(density_approx)
density_resized = np.reshape(density_approx, (len(density_approx), 1))
print(np.shape(Be_grid), np.shape(density_resized))


#############################################################################################################################
########################################## NON-NEGATIVE LEAST SQUARES #######################################################
#############################################################################################################################

def cofactor_matrix(UGBS, grid):
    inner = UGBS * grid**2 # 503 x 25
    gaussian = np.exp(-2 * inner) # 503 x 25
    return gaussian

# This uses clenshaw grid and electron density
cofactor_mat = cofactor_matrix(be_UGBS, Be_grid)
print("The Shape of A/Cofactor Matrix is ", np.shape(cofactor_mat))
NNLS_coeff = scipy.optimize.nnls(cofactor_mat, be_rho)[0]
print("The shape of cofactor and b", cofactor_mat.ndim, be_rho.ndim)
print("These Are The Coefficients Obtained From f_min_BFGS\n", coeff, "\n")
print("These are the Coefficients Obtained from NNLS\n", NNLS_coeff, "\n")
print( np.shape(NNLS_coeff), np.shape(Be_grid))

def plot_function(coefficients, UGBS, grid):
    inner = UGBS * grid**2
    gaussian = np.dot(np.exp(-2 * inner), NNLS_coeff)
    return gaussian

NNLS_density_approx = plot_function(NNLS_coeff, NNLS_coeff, np.reshape(Be_grid, (len(Be_grid), 1)))
print("Shape", np.shape(NNLS_density_approx))
density_approx = np.reshape(density_approx, (len(density_approx), 1)) # this is for the plot

#Plotting
log_grid = np.log(np.ravel(Be_grid))
grid = np.ravel(Be_grid)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
ax.plot(grid, (density_approx), "b", label="F_MIN_BFGS_B")
ax.plot(grid, (NNLS_density_approx), "r--", label="NNLS")
ax.plot(grid, (be_rho), "g", label="True Density ")
ax.legend(bbox_to_anchor=(1.00, 1), loc=2, borderaxespad=0.)
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
ax.text(1.2, (0.1 + 0.75)/2, "T - F_Min_BFGS_B " + str(np.round(np.asscalar(np.absolute(np.sum(be_rho - density_approx))), decimals=2))
                            + "\n T - NNLS " + str(np.round(np.asscalar(np.absolute(np.sum(be_rho - NNLS_density_approx))), decimals=2))
                            + "\n F_Min - NNLS " + str(np.round(np.asscalar(np.absolute(np.sum(density_approx - NNLS_density_approx))), decimals=2)),
        transform=ax.transAxes, fontsize=14,
        verticalalignment='center', horizontalalignment='center', bbox=props)

plt.show()

#Integration
print("Integration of Beryllium for NNLS Using Clenshaw Grid is ", np.trapz(np.ravel(Be_grid**2) * np.ravel(NNLS_density_approx),
                                                                            np.ravel(Be_grid)))
print("Integration Of Beryllium for f_Min_LGBS Using Clenshaw Grid is ", np.trapz(np.ravel(Be_grid**2 * density_resized), np.ravel(Be_grid)) )       #number of electrons by integrating the approximate model

#Taking The Difference Between the Techniques
print("The Difference between True and F_Min_BFGS_B ", np.absolute(np.sum(be_rho - density_approx)))
NNLS_density_approx = np.reshape(NNLS_density_approx, (len(NNLS_density_approx), 1))
print("The Difference between True and NNLS ", np.absolute(np.sum(be_rho - NNLS_density_approx)))
print("The Difference between F_Min_BFGS_B and NNLS ", np.sum(density_approx - NNLS_density_approx))
"""
