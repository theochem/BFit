from fitting.density.atomic_slater_density import *
from fitting.gbasis.gbasis import UGBSBasis
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

#Ask Farnaz - The Atomic_slater_density uses an column vector, so should Density Model be column vector as well?
class Model():
    def __init__(self, element_name, file_path, grid, exponents=[], change_exponents=False):
        assert isinstance(file_path, str)
        assert grid.ndim == 2 and np.shape(grid)[1] == 1

        self.grid = grid;
        self.grid.flags.writeable = False; #Immutable Array

        self.file_path = file_path

        atomic_density_obj = Atomic_Density(file_path, self.grid)
        self.electron_density = atomic_density_obj.atomic_density()
        self.electron_density.flags.writeable = False; #Immutable Array
        self.electron_density_core, self.electron_density_valence = atomic_density_obj.atomic_density_core_valence()

        if change_exponents:
            assert type(exponents).__module__ == np.__name__
            self.exponents = exponents
        else:
            gbasis =  UGBSBasis(element_name)
            self.exponents = 2.0 * gbasis.exponents()
            assert type(self.exponents).__module__ == np.__name__
        self.nbasis = self.exponents.shape[0]

    def model(self, coefficients, exponents):
        """
        Used this to compute the exponents and coefficients
        Exponents should be 1ndim list

        Shape (Number of Grid Points) Row Vector
        :param coefficients:
        :param exponents:
        :return:
        """
        assert type(coefficients).__module__ == np.__name__
        assert type(exponents).__module__ == np.__name__
        assert exponents.ndim == 1
        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        assert np.shape(exponential)[0] == np.shape(self.grid)[0]
        assert np.shape(exponential)[1] == np.shape(exponents)[0]
        assert exponential.ndim == 2

        assert coefficients.ndim == 1
        assert (coefficients.shape)[0] == exponential.shape[1]
        gaussian_density = np.dot(exponential, coefficients) # is gaussian the right word?
        assert gaussian_density.ndim == 1
        assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]
        return(gaussian_density)

    def model_valence(self, s_coefficients, s_exponents, p_coefficients, p_exponents):
        #Assert all arguments are numpy arrays
        assert type(s_coefficients).__module__ == np.__name__
        assert type(s_exponents).__module__ == np.__name__
        assert type(p_coefficients).__module__ == np.__name__
        assert type(p_exponents).__module__ == np.__name__

        assert s_exponents.ndim == 1; assert p_exponents.ndim == 1;
        s_exponential = np.exp(-s_exponents * np.power(self.grid, 2.0))
        p_exponential = np.exp(-p_exponents * np.power(self.grid, 2.0))
        assert np.shape(s_exponential)[0] == np.shape(self.grid)[0]
        assert np.shape(p_exponential)[0] == np.shape(self.grid)[0]
        assert np.shape(s_exponential)[1] == np.shape(s_exponents)[0]
        assert np.shape(p_exponential)[1] == np.shape(p_exponents)[0]

        assert s_exponential.ndim == 2; assert p_exponential.ndim == 2
        assert s_coefficients.ndim == 1
        assert (s_coefficients.shape)[0] == s_exponential.shape[1]
        s_gaussian_model = np.dot(s_exponential, s_coefficients)
        p_gaussian_model = np.dot(p_exponential, p_coefficients)
        p_gaussian_model = np.ravel(p_gaussian_model)  * np.ravel(np.power(self.grid, 2.0))
        assert s_gaussian_model.ndim == 1
        assert np.shape(s_gaussian_model)[0] == np.shape(self.grid)[0]
        return(s_gaussian_model + p_gaussian_model)

    def residual(self, coefficients, exponents, core=False, valence=False, ln=False, weight=1.0):
        assert core != True or valence != True
        model = self.model(coefficients, exponents)
        if core:
            return(np.ravel(self.electron_density_core) - model)
        elif ln:
            return(weight * (np.ravel(np.log(self.electron_density)) - np.ravel(np.log(model))))
        elif valence:
            return(np.ravel(self.electron_density_valence) - model)
        else:
            return(weight * (np.ravel(self.electron_density) - model))

    def residual_valence(self, s_coefficients, s_exponents, p_coefficients, p_exponents):
        model = self.model_valence(s_coefficients, s_exponents, p_coefficients, p_exponents)
        return(np.ravel(self.electron_density_valence) - model)

    def cofactor_matrix(self, exponents=[], change_exponents=False):
        """
        Params
        exponents is 1 dimensional array

        Returns e^(exponents * radius**2) Matrix
        Shape is (Number of Grids, Number of Exponents)
        """

        if change_exponents:
            exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
            assert(exponential.shape[1] == len(np.ravel(exponents)))
        else:
            exponential = np.exp(-self.exponents * np.power(self.grid, 2.0))
            assert(exponential.shape[1] == len(np.ravel(self.exponents)))

        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)

    def cost_function_valence_both(self, parameters, num_of_s_basis, num_of_p_basis):
        assert type(parameters).__module__ == np.__name__
        assert isinstance(num_of_s_basis, int)
        assert isinstance(num_of_p_basis, int)

        s_coefficients = parameters[:num_of_s_basis]
        s_exponents = parameters[num_of_s_basis:2 * num_of_s_basis]

        p_coefficients = parameters[num_of_s_basis * 2 :num_of_s_basis * 2 + num_of_p_basis]
        p_exponents = parameters[num_of_s_basis * 2 + num_of_p_basis:]

        residual = self.residual_valence(s_coefficients, s_exponents, p_coefficients, p_exponents)
        residual_squared = np.power(residual, 2.0)

        return(np.sum(residual_squared))

    def cost_function(self, coefficient, exponents=[], change_exponents=False, core=False, valence=False, ln=False):
        if not change_exponents:
            exponents = self.exponents
        residual = self.residual(coefficient, exponents, core=core, valence=valence, ln=ln)
        residual_squared = np.power(residual, 2.0)

        return(np.sum(residual_squared))

    def cost_function_both(self, parameters, num_of_basis, core=False, valence=False, ln=False, weight=1.0):
        assert type(parameters).__module__ == np.__name__
        assert isinstance(num_of_basis, int)
        assert not (core == True and valence == True)
        coeffs = parameters[:num_of_basis]
        exponents = parameters[num_of_basis:]

        residual = self.residual(coeffs, exponents, core=core, valence=valence, ln=ln, weight=weight)
        residual_squared = np.power(residual, 2.0)

        return(np.sum(residual_squared))

    def derivative_cost_function(self, coefficient, exponents=[], change_exponents=False, core=False, valence=False):
        if not change_exponents:
            exponents = self.exponents

        residual = self.residual(coefficient, exponents, core=core, valence=valence)

        f_function = 2.0 * residual
        derivative_exp = []

        for exp in exponents:
            g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
            derivative = f_function * np.ravel(g_function)
            derivative_exp.append(np.ravel(derivative))
        assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
        derivative = np.asarray(derivative)
        assert derivative_exp[0].ndim == 1
        return(np.sum(np.asarray(derivative_exp), axis=1))

    def derivative_cost_function_both(self, parameters, num_of_basis, core=False, valence=False):
        coefficients = parameters[:num_of_basis]
        exponents = parameters[num_of_basis:]

        residual = self.residual(coefficients, exponents, core=core, valence=valence)

        f_function = 2.0 * residual
        derivative = []

        def derivative_coeff_helper():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_exp_helper():
            derivative_exp = []
            for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                exponent = exponents[index]
                g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -f_function * np.ravel(g_function)
                derivative_exp.append(np.ravel(derivative))
            assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
            return(derivative_exp)

        derivative_coeff = derivative_coeff_helper()
        derivative_exp = derivative_exp_helper()
        derivative = derivative + derivative_coeff
        derivative = derivative + derivative_exp

        return(np.sum(derivative, axis=1))

    def derivative_cost_function_valence(self, parameters, num_of_s_basis, num_of_p_basis):
        s_coefficients = parameters[:num_of_s_basis]
        s_exponents = parameters[num_of_s_basis:2 * num_of_s_basis]

        p_coefficients = parameters[num_of_s_basis * 2 :num_of_s_basis * 2 + num_of_p_basis]
        p_exponents = parameters[num_of_s_basis * 2 + num_of_p_basis:]
        residual = self.residual_valence(s_coefficients, s_exponents, p_coefficients, p_exponents)

        f_function = 2.0 * residual
        derivative = []

        def derivative_coeff_helper():
            derivative_s_coeff = []
            for exp in s_exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_s_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_s_coeff[0])[0] == np.shape(self.grid)[0]
            derivative_p_coeff = []
            for exp in p_exponents:
                g_function = -1.0 * np.power(self.grid, 2.0) * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_p_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_s_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_s_coeff, derivative_p_coeff)

        def derivative_exp_helper():
            derivative_s_exp = []
            for index, coeff in np.ndenumerate(np.ravel(s_coefficients)):
                exponent = s_exponents[index]
                g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -f_function * np.ravel(g_function)
                derivative_s_exp.append(np.ravel(derivative))
            assert np.shape(derivative_s_exp[0])[0] == np.shape(self.grid)[0]
            derivative_p_exp = []
            for index, coeff in np.ndenumerate(np.ravel(p_coefficients)):
                exponent = p_exponents[index]
                g_function = -coeff * np.power(self.grid, 4.0) * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -f_function * np.ravel(g_function)
                derivative_p_exp.append(np.ravel(derivative))
            return(derivative_s_exp, derivative_p_exp)

        derivative_s_coeff, derivative_p_coeff = derivative_coeff_helper()
        derivative_s_exp, derivative_p_exp = derivative_exp_helper()
        derivative = derivative + derivative_s_coeff + derivative_s_exp
        derivative = derivative + derivative_p_coeff + derivative_p_exp

        return(np.sum(derivative, axis=1))

    def f_min_slsqp_coefficients(self, initial_guess, factr=1e7, exponents=[], opt_coeff=True, opt_both=False, change_exponents=False, core=False, valence=False, use_slsqp=True):
        if opt_both==True: opt_coeff=False
        assert not (opt_coeff == True and opt_both == True)
        assert opt_coeff == True or opt_both == True
        assert type(initial_guess).__module__ == np.__name__

        if not change_exponents:
            exponents = np.copy(self.exponents)
        bounds = np.array([(0.0, np.inf) for x in range(0, len(initial_guess))], dtype=np.float64)

        if opt_coeff:
            if(use_slsqp):
                f_min_slsqp = scipy.optimize.fmin_slsqp(self.cost_function, x0=initial_guess, bounds=bounds, fprime=self.derivative_cost_function,
                                                         acc = 1e-10,iter=1500000, args=(exponents, change_exponents, core, valence), full_output=True, iprint=-1)
                parameters = f_min_slsqp[0]
                print(f_min_slsqp[4])
            elif(not use_slsqp):
                f_min_slsqp = scipy.optimize.fmin_l_bfgs_b(self.cost_function, x0=initial_guess, bounds=bounds, fprime=self.derivative_cost_function
                                                       ,maxfun=1500000, maxiter=1500000, factr=factr, args=(exponents, change_exponents, core, valence), pgtol=1e-7)
                parameters = f_min_slsqp[0]
                print(f_min_slsqp[2]['warnflag'], "            ", f_min_slsqp[2]['task'])

                if f_min_slsqp[2]['warnflag'] != 0:
                    print(f_min_slsqp[2]['task'])
        elif opt_both:
            assert initial_guess.shape[0] % 2 == 0
            num_of_basis = int(initial_guess.shape[0]/2)
            f_min_slsqp = scipy.optimize.fmin_slsqp(self.cost_function_both, x0=initial_guess, bounds=bounds, fprime=self.derivative_cost_function_both,
                                                    acc=1e-12, iter=150000, args=(num_of_basis, core, valence), full_output=True, iprint=0)
            #print(f_min_slsqp[4])

            #f_min_slsqp = scipy.optimize.differential_evolution(self.cost_function_both, bounds=bounds, args=(num_of_basis, core, valence))
            #print(f_min_slsqp['success'])

            #f_min_slsqp = scipy.optimize.fmin_tnc(self.cost_function_both, x0=initial_guess, fprime=self.derivative_cost_function_both, bounds=bounds, args=(num_of_basis,))
            #print(f_min_slsqp[2])


            #f_min_slsqp = scipy.optimize.fmin_l_bfgs_b(self.cost_function_both, x0=initial_guess, bounds=bounds, fprime=self.derivative_cost_function_both
            #                                           ,maxfun=1500000, maxiter=1500000, factr=factr, args=(num_of_basis, core, valence), pgtol=1e-7)
            #print(f_min_slsqp[2]['warnflag'], "            ", f_min_slsqp[2]['task'])

            #if f_min_slsqp[2]['warnflag'] != 0:
            #    print(f_min_slsqp[2]['task'])
            parameters = f_min_slsqp[0]


        return(parameters)

    def nnls_coefficients(self, cofactor_matrix):
        b_vector = np.copy(self.electron_density)

        #b vector has to be one dimensional
        b_vector = np.ravel(b_vector)
        assert np.ndim(b_vector) == 1
        row_nnls_coefficients = scipy.optimize.nnls(cofactor_matrix, b_vector)
        #print(row_nnls_coefficients)
        return(row_nnls_coefficients[0])

    def integration(self, coefficients, exponents):
        # Integrate Based On Model or
        # Integrate Based on COefficients

        assert coefficients.ndim == 1
        assert exponents.ndim == 1
        electron_density = self.model(coefficients, exponents) #row vector
        integrate = np.trapz(np.ravel(np.power(self.grid,2.0)) * np.ravel(electron_density), np.ravel(self.grid))
        return integrate

    def true_value(self, points, core=False, valence=False):
        assert core != True or valence != True
        p, w = np.polynomial.laguerre.laggauss(points)
        p = np.reshape(p, (len(p), 1))
        w = np.reshape(w, (len(w), 1))
        be = Atomic_Density(self.file_path, p)
        if core:
            a = be.atomic_density_core()[0] * p**2 * w/ np.exp(-p)
        elif valence:
            a = be.atomic_density_core()[1] * p**2 * w/ np.exp(-p)
        else:
            a = be.atomic_density() * p**2 * w/ np.exp(-p)
        return(np.sum(a))

    def pauls_new_GA(self, factor, accuracy, maximum_exponents=50, opt_both=True, core=False):
        assert isinstance(maximum_exponents, int)
        assert isinstance(factor, float)
        assert isinstance(accuracy, float)
        FACTR = 1e12

        def analytical_solver(electron_density, weight=1):
            a = 2.0 * self.grid.shape[0]
            sum_of_grid_squared = np.sum(np.ravel(np.power(self.grid, 2)))
            b = 2.0 * sum_of_grid_squared
            sum_ln_electron_density = np.sum(np.ravel(np.log(electron_density)))
            c = 2.0 * sum_ln_electron_density
            d = b
            e = 2.0 * np.sum(np.ravel(np.power(self.grid, 4)))
            f = 2.0 * np.sum(np.ravel(np.power(self.grid, 2)) * np.ravel(np.log(electron_density)))
            A = (b * f - c * e)/ (b * d - a * e)
            B = (a * f - c * d)/ (a * e - b * d)
            coefficient = np.exp(A)
            exponent = -B
            return(coefficient , exponent)

        def measure_error_helper(coeff, exponents, core=core, ln=False, weight=1.0):
            model = self.model(coeff, exponents)
            if core:
                residual = self.residual(coeff, exponents, core=True)
                diff = np.absolute(self.residual(coeff, exponents, core=True))
            else:
                residual = self.residual(coeff, exponents, ln=ln, weight=weight)
                diff = np.absolute(residual)
            grid = np.ravel(self.grid)
            integration = np.trapz(y=diff * np.ravel(np.power(self.grid, 2)), x=grid)

            #inte = self.integration(coeff, exponents)
            #true_val = self.true_value(100)
            #integration = np.absolute(inte-true_val)
            return(integration)

        weights = [1.0, np.ravel(self.electron_density), np.ravel(self.electron_density)**2]

        #Find Best One Gaussian Function
        initial_coefficient, initial_exponent = analytical_solver(self.electron_density, weight=weights[0])
        initial_parameters = np.array([initial_coefficient, initial_exponent])
        parameters = self.f_min_slsqp_coefficients(initial_guess=initial_parameters, factr=FACTR, opt_both=True, core=core)
        print(initial_parameters, parameters)
        error_one = measure_error_helper(np.array([initial_coefficient]), np.array([initial_exponent]), ln=True, weight=1.0)
        error_weight2 = measure_error_helper(np.array([initial_coefficient]), np.array([initial_exponent]), ln=True, weight=weights[1])
        error_weight3 = measure_error_helper(np.array([initial_coefficient]), np.array([initial_exponent]), ln=True, weight=weights[2])
        print(error_one, error_weight2, error_weight3)

        radial_grid = self.grid *  0.5291772082999999

        plt.semilogy(radial_grid, self.model(np.array([initial_coefficient]), np.array([initial_exponent])), 'b',  lw=3)
        plt.semilogy(radial_grid, self.electron_density, 'r',  lw=3)
        plt.semilogy(radial_grid, weights[1] * self.model(np.array([initial_coefficient]), np.array([initial_exponent])), 'g',  lw=3)
        plt.semilogy(radial_grid, weights[2] * self.model(np.array([initial_coefficient]), np.array([initial_exponent])), 'o',  lw=3)
        plt.semilogy(radial_grid, self.model(parameters[0:1], parameters[1:]), 'y',  lw=3)
        plt.xlim(0, 7.5)
        plt.ylim(ymin=1e-8)
        plt.xlabel('Distance from the nucleus [A]')
        plt.ylabel('Log(density [Bohr**-3])')
        plt.title("Fitting The Core Electron Density")
        plt.show()

        # Do While Loop

    def pauls_GA_valence(self, factor, accuracy, opt_Both=True):
        assert isinstance(factor, float)
        assert isinstance(accuracy, float)

    def old_pauls_GA(self, factor, factor_factor, initial_guess, accuracy, maximum_exponents=50, use_slsqp=True, use_nnls=False):
        assert isinstance(maximum_exponents, int)
        assert isinstance(factor, float)
        assert isinstance(initial_guess, float) or all(isinstance(x, float) for x in initial_guess)

        def measure_error_helper(coeff, exponents):
            model = self.model(coeff, exponents)
            diff = np.absolute(np.ravel(self.electron_density) - model)
            grid = np.ravel(self.grid)
            integration = np.trapz(y=diff * np.ravel(np.power(self.grid, 2)), x=grid)
            #integration = np.trapz(y=model, x=grid)
            #integration2 = np.trapz(y=np.ravel(self.electron_density), x=grid)
            return(integration)

        def best_UGBS_helper():
            counter = 0
            lowest_error = None
            best_UGBS = None
            for exponent in self.exponents:
                exponent = np.array([exponent])
                if use_nnls:
                    cofactor = self.cofactor_matrix(exponent, change_exponents=True)
                    coefficients = self.nnls_coefficients(cofactor)
                elif use_slsqp:
                    coefficients = self.f_min_slsqp_coefficients([initial_guess], exponents=exponent, change_exponents=True)
                    #print(exponent, coefficients)
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

        print("The Best UGBS Exponent is ", [x for x in best_UGBS], " that gave an error of ", lowest_error, " with coefficient", coeff)
        while(lowest_error > accuracy):
            mult_exp, div_exp  = splitting_array_helper(best_UGBS, factor)
            factor *= factor_factor

            initial_guess = np.append(coeff, [np.absolute(coeff[len(coeff) - 1]* np.random.random()) for x in range(0, len(coeff)) ])
            if use_slsqp:
                mult_coeff = self.f_min_slsqp_coefficients(initial_guess, exponents=mult_exp, change_exponents=True)
                div_coeff = self.f_min_slsqp_coefficients(initial_guess, exponents=div_exp, change_exponents=True)
            elif use_nnls:
                mult_cofac = self.cofactor_matrix(exponents=mult_exp, change_exponents=True)
                div_cofac = self.cofactor_matrix(exponents=div_exp, change_exponents=True)

                mult_coeff = self.nnls_coefficients(mult_cofac)
                div_coeff = self.nnls_coefficients(div_cofac)

            mult_error = measure_error_helper(mult_coeff, mult_exp)
            div_error = measure_error_helper(div_coeff, div_exp)

            if mult_error < div_error:
                coeff = mult_coeff
                best_UGBS = mult_exp
                print('mult', best_UGBS)

            else:
                coeff = div_coeff
                best_UGBS = div_exp
                print('div', best_UGBS)

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

    def particle_swarm_optimization(self, parameters, coeff_guess, num_of_particles, num_of_basis):
        list_of_points = np.linspace(0, 1000, num=num_of_particles)

        # Initialize All of the Particles
        list_of_particles = []
        index_of_one = np.where(parameters == coeff_guess)[0][0]
        for point in list_of_points:
            test_para = np.copy(parameters)
            test_para[index_of_one] = point
            list_of_particles.append(test_para)

        #import time
        #plt.ion()
        #plt.show()
        #plt.scatter([x[0] for x in list_of_particles], [x[1] for x in list_of_particles])
        #plt.draw()
        #time.sleep(5)
        #plt.cla()


        #Measure Fitness of Particles
        gbest, gvalue = 1e10, None
        gparticle = None
        particles_info = {} #Format Of (Best Fitness Calculated, Best Position of the Particle, Velocity of Particle)
        index = 0;
        for particle in list_of_particles:
            fitness_of_particle = self.cost_function_both(particle, num_of_basis=num_of_basis)
            particles_info[index] = [fitness_of_particle, particle[index_of_one], 0.]
            if fitness_of_particle < gbest and (particle >= 0).all():
                gbest = fitness_of_particle
                gvalue = particle[index_of_one]
                gparticle = particle
            index += 1

        #Particle Swarm Technique
        for xy in range(0, 200):
            index = 0
            for particle in list_of_particles:
                #Extract Information Regarding Particle
                particles_initial_position = particle[index_of_one]
                bfitness_of_particle, particle_best_position, particle_velocity = particles_info[index] #Gives Information Regarding Particle

                #Calculate Velocity and New Position of Particle
                weight1 = 0.5; weight2 = 0.50; weight3 = 0.75
                velocity = weight1 * particle_velocity + weight2 * np.random.random() * (particle_best_position - particles_initial_position) + weight3 * np.random.random() * (gvalue - particles_initial_position)
                new_position = particles_initial_position + velocity
                if new_position >0 :
                    particle[index_of_one] = new_position
                else:
                    velocity *= -0.5
                    new_position = particles_initial_position + velocity

                # Calculate Fitness
                new_fitness = self.cost_function_both(particle, num_of_basis=num_of_basis)

                # Update new fitness and new best position for that particle
                if new_fitness <  bfitness_of_particle and new_position > 0:
                    particles_info[index][0] = new_fitness
                    particles_info[index][1] = new_position
                    particles_info[index][2] = velocity
                    if new_fitness < gbest:
                        gbest = new_fitness
                        gvalue = particle[index_of_one]
                        gparticle = np.copy(particle)

                # Update Velocity
                if new_position > 0:
                    particles_info[index][2] = velocity
                #print(particle, particles_info[index])
                index += 1
            #plt.scatter([x[0] for x in list_of_particles], [x[1] for x in list_of_particles])
            #plt.ylabel("sds")
            #plt.xlabel("sds")
            #plt.ylim(0.0, 100)
            #plt.draw()
            #time.sleep(1)
            #plt.cla()



        #print(particles_info)
        #print("GBest", gbest, "GParticle", gparticle,"G Value", gvalue,)
        return(gparticle)

    def pauls_GA(self, factor, accuracy, factr=1e7, maximum_exponents=25, opt_both=True, opt_coeff=False, core=False, valence=False):
        if opt_coeff == True: opt_both = False
        assert isinstance(maximum_exponents, int)
        assert isinstance(factor, float)
        assert isinstance(accuracy, float)

        def measure_error_helper(coeff, exponents, core=core):
            model = self.model(coeff, exponents)
            if core:
                diff = np.absolute(np.ravel(self.electron_density_core) - model)
            else:
                diff = np.absolute(np.ravel(self.electron_density) - model)
            grid = np.ravel(self.grid)
            integration = np.trapz(y=diff * np.ravel(np.power(self.grid, 2)), x=grid)

            #inte = self.integration(coeff, exponents)
            #true_val = self.true_value(100)
            #integration = np.absolute(inte-true_val)
            return(integration)

        def split_coeff_exp(parameters, factor, coeff_guess, num):
            assert parameters.ndim == 1
            size = parameters.shape[0]
            #assert num <= size/2
            if parameters.shape[0] == 2:
                coeff, exponent = parameters
                mult_exponent = np.copy(np.array(exponent))
                div_exponent = np.copy(np.array(exponent))

                for x in range(1, num + 1):
                    new_factor = factor ** x
                    mult_exponent = ( np.append(mult_exponent, np.array([exponent * new_factor])) )
                    div_exponent = ( np.sort(np.append(div_exponent, np.array([exponent / new_factor])) ))

                mult_coeff = np.array([coeff] + [1.0 for x in range(0, num)])
                div_coeff = np.array([1.0 for x in range(0, num)] + [coeff])

                mult_params = np.append(mult_coeff, mult_exponent)
                div_params = np.append(div_coeff, div_exponent)

                return([mult_params, div_params])

            elif num == 1:
                all_parameters = []
                exponent = parameters[size/2:]
                coeff = parameters[0:size/2]
                for index, exp in np.ndenumerate(exponent):
                    if index[0] == 0:
                        for x in range(1, num + 1):
                            exponent_array = np.insert(exponent, index, exp / (factor * x))
                            coefficient_array = np.insert(coeff, index, coeff_guess)

                    elif index[0] <= size/2 - 1:
                        exponent_array = np.insert(exponent, index, (exponent[index[0] - 1] + exponent[index[0]])/2)
                        coefficient_array = np.insert(coeff, index, coeff_guess)
                    all_parameters.append(np.append(coefficient_array, exponent_array))
                    if index[0] == size/2 - 1:
                        exponent_array = np.append(exponent, np.array([ exp * factor] ))
                        coefficient_array = np.append(coeff, np.array([coeff_guess]))
                        all_parameters.append(np.append(coefficient_array, exponent_array))
                return(all_parameters)
            else:
                all_parameters = []
                exponent = parameters[size/2:]
                coeff = parameters[0:size/2]

                for time in range(0, num + int(size/2)):
                    exponent_original = np.copy(exponent)
                    coeff_original = np.copy(coeff)
                    #print(time)
                    if time == 0:
                        for x in range(0, num):
                            new_factor = factor ** (x + 1)
                            exponent_original = np.insert(exponent_original, time, exponent[0]/ new_factor)
                            coeff_original = np.insert(coeff_original, time, coeff_guess)

                        #print(coeff_original, exponent_original)
                    elif time > 0:
                        if time < num :
                            for x in range(0, num - time):
                                new_factor = factor ** (x + 1)
                                exponent_original = np.insert(exponent_original, 0, exponent[0] / new_factor)
                                coeff_original = np.insert(coeff_original, 0, coeff_guess)
                            position_to_add = 1
                            position_of_first = np.where(exponent_original==exponent[0])[0]
                            position_to_avg = 1
                            for x in range(num - time, num):
                                exponent_original = np.insert(exponent_original, position_of_first + position_to_add, (exponent[position_to_avg ] + exponent[position_to_avg - 1])/2)
                                coeff_original = np.insert(coeff_original, position_of_first + position_to_add, coeff_guess)
                                position_to_add += 2
                                position_to_avg += 1
                            #print(coeff_original, exponent_original)
                        elif time == num:
                            num_of_avgs = int(size/2 - 1)
                            position_to_add = 1
                            position_of_first = np.where(exponent_original==exponent[0])[0]
                            position_to_avg = 1
                            if num == size/2:
                                num2 = num - 1
                            else:
                                num2 = num
                            for x in range(0, num2):
                                exponent_original = np.insert(exponent_original, position_of_first + position_to_add, (exponent[position_to_avg] + exponent[position_to_avg - 1]) /2)
                                coeff_original = np.insert(coeff_original, position_of_first + position_to_add, coeff_guess)
                                position_to_add += 2
                                position_to_avg += 1
                            for x in range(0, num - num_of_avgs):
                                new_factor = factor ** (x + 1)
                                exponent_original = np.append(exponent_original, exponent[-1]*new_factor)
                                coeff_original = np.append(coeff_original, coeff_guess)
                            #print(coeff_original, exponent_original)

                        elif time > num and time != num + int(size/2) - 1:
                            exponent_original = np.flipud(exponent_original) #Reverses it
                            coeff_original = np.flipud(exponent_original)
                            for x in range(0, time - int(size/2) + 1):
                                new_factor = factor ** (x + 1)
                                exponent_original = np.insert(exponent_original, 0, exponent[-1] * new_factor)
                                coeff_original = np.insert(coeff_original, 0, coeff_guess)
                            position = -1
                            for x in range(time - int(size/2) + 1, num):
                                position_of_first = np.where(exponent_original==exponent[position])[0]
                                exponent_original = np.insert(exponent_original, position_of_first + 1, (exponent[position] + exponent[position - 1])/2)
                                coeff_original = np.insert(coeff_original, position_of_first + 1, coeff_guess)
                                position -= 1
                            exponent_original = (np.flipud(exponent_original))
                            coeff_original = np.flipud(coeff_original)
                            #print(coeff_original, exponent_original)
                    if time == num + int(size/2) - 1:
                        for x in range(0, num):
                            new_factor = factor ** (x + 1)
                            exponent_original = np.append(exponent_original, exponent[-1] * new_factor)
                            coeff_original = np.append(coeff_original, coeff_guess)
                        #print(coeff_original, exponent_original)
                    assert coeff_original.shape[0] == exponent_original.shape[0]
                    all_parameters.append(np.append(coeff_original , exponent_original))
                return(all_parameters)

        def analytical_solver(electron_density):
            a = 2.0 * self.grid.shape[0]
            sum_of_grid_squared = np.sum(np.ravel(np.power(self.grid, 2)))
            b = 2.0 * sum_of_grid_squared
            sum_ln_electron_density = np.sum(np.ravel(np.log(electron_density)))
            c = 2.0 * sum_ln_electron_density
            d = b
            e = 2.0 * np.sum(np.ravel(np.power(self.grid, 4)))
            f = 2.0 * np.sum(np.ravel(np.power(self.grid, 2)) * np.ravel(np.log(electron_density)))
            A = (b * f - c * e)/ (b * d - a * e)
            B = (a * f - c * d)/ (a * e - b * d)
            coefficient = np.exp(A)
            exponent = - B
            return(coefficient, exponent)

        initial_guess = np.array([1.0, 1.0])
        parameters = self.f_min_slsqp_coefficients(initial_guess=initial_guess, factr=factr, opt_both=True, core=core, valence=valence)
        print("Single, Best Coeff and Exponents", parameters)

        lowest_error = 1.0e5
        counter = 0
        coeff_guess = 1.0
        size = 0
        #plt.ion()
        #plt.show()
        while(lowest_error > accuracy and size/2 <= maximum_exponents):
            all_parameters = split_coeff_exp(parameters, factor, coeff_guess, 1)
            print(all_parameters)
            size = all_parameters[0].shape[0]
            lowest_error_parameters = 1.0e5
            best_parameters_from_all = None

            for set_of_parameters in all_parameters:
                size = set_of_parameters.shape[0]

                if opt_both:
                    optimized_params = self.f_min_slsqp_coefficients(initial_guess=set_of_parameters, opt_both=True, factr=factr, core=core, valence=valence)
                    coeff, exp = optimized_params[0:size/2], optimized_params[size/2:]
                    error_of_params = measure_error_helper(coeff, exp)

                elif opt_coeff:
                    cofac = self.cofactor_matrix(set_of_parameters[size/2:], change_exponents=True)
                    nnls_coeff = self.nnls_coefficients(cofac)
                    optimized_params = np.append(nnls_coeff, set_of_parameters[size/2:], factr=factr)
                    error_of_params = measure_error_helper(nnls_coeff, set_of_parameters[size/2:])

                if error_of_params < lowest_error_parameters:
                    lowest_error_parameters = error_of_params
                    best_parameters_from_all = optimized_params
                    #0.1944 - 1e9, 0.1268 - 1e8, 0.0525 - 1e7, 0.1526 - 1e6, 0.0482 - 1e5,
            if lowest_error_parameters < lowest_error:
                lowest_error = lowest_error_parameters

            parameters = best_parameters_from_all
            counter+=1
            size = parameters.shape[0]
            coeff,exp = parameters[0:size/2], parameters[size/2:]
            print("The Parameters", [parameters])
            inte = self.integration(coeff, exp)
            true = self.true_value(100, core=core, valence=valence)
            #radial_grid = self.grid *  0.5291772082999999
            #plt.semilogy(radial_grid, self.model(coeff, exp),  lw=3)
            #plt.semilogy(radial_grid, self.electron_density, 'r',  lw=3)
            #plt.xlim(0, 7.5)
            #plt.ylim(ymin=1e-8)
            #plt.xlabel('Distance from the nucleus [A]')
            #plt.ylabel('Log(density [Bohr**-3])')
            #plt.title("Fitting The Core Electron Density")
            #plt.draw()
            print("Iteration", counter, "  lowest error of iteration", lowest_error_parameters, "  size of exp", size/2, )
            print("Integration of best COeff ", inte, "True Value: ", true, "   Difference: ", np.absolute(inte-true),  "\n")
        return(parameters ,lowest_error)


file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\be.slater"
from fitting.density.radial_grid import *
radial_grid = Radial_Grid(4)
row_grid_points = radial_grid.grid_points(200, 300, [50, 75, 100])

column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
be = Model('be', file_path, column_grid_points)
#be.cost_function(np.array([1.0, 2.0]), np.array([1.0, 2.0]), change_exponents=True, core=True, valence=False)

#best_particle = be.particle_swarm_optimization(np.array([403, 1.0]), 1.0, num_of_particles=10, num_of_basis=1)
#be.pauls_GA(2.0, 1e-5, factr=1e7, opt_coeff=False, opt_both=True, core=False)
#be.pauls_new_GA(factor=2.0, accuracy=0.01)
#print(be.model_p_gaussians(np.array([1.]), np.array([2.]), np.array([3.]), np.array([4.])))

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

    #plt.xlim(0, 25.0*0.5291772082999999)x
    plt.xlim(0, 7.5)
    plt.ylim(ymin=1e-8)
    plt.xlabel('Distance from the nucleus [A]')
    plt.ylabel('Log(density [Bohr**-3])')
    plt.title(title)
    plt.legend(loc=0)
    plt.savefig(figure_name)
    plt.close()


#print(np.min(be.electron_density_core + be.electron_density_valence - be.electron_density))
#plot_atomic_desnity(be.grid, density_list=[(be.electron_density,"True Den"), (be.electron_density_core, "Core"), (be.electron_density_valence, "valence")], title="Bery", figure_name="Be")



#import sys
#sys.exit()