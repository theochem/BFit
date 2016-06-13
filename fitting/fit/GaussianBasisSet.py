from fitting.fit.model import *
import numpy as np


#TODO CHANGE GRID TO BEING ONE DIMENSIONAL
class GaussianTotalBasisSet(DensityModel):
    def __init__(self, element_name, grid, file_path):
        DensityModel.__init__(self, element_name, grid, file_path)

    def create_model(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        assert parameters.ndim == 1
        def check_dimension_and_shape():
            assert exponents.ndim == 1
            assert np.shape(exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(exponential)[1] == np.shape(exponents)[0]
            assert exponential.ndim == 2
            assert coefficients.ndim == 1
            assert (coefficients.shape)[0] == exponential.shape[1]
            assert gaussian_density.ndim == 1
            assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianTotalBasisSet, self).calculate_residual(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianTotalBasisSet, self).calculate_residual(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianTotalBasisSet, self).calculate_residual(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)

        residual_squared = np.power(residual, 2.0)
        return(np.sum(residual_squared))

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_wrt_exponents():
                derivative_exp = []
                for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                    exponent = exponents[index]
                    g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                    derivative = -f_function * np.ravel(g_function)
                    derivative_exp.append(np.ravel(derivative))
                assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
                return(derivative_exp)

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianTotalBasisSet, self).calculate_residual(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

            f_function = 2.0 * residual
            derivative = []

            derivative_coeff = derivative_wrt_coefficients()
            derivative_exp = derivative_wrt_exponents()
            derivative = derivative + derivative_coeff
            derivative = derivative + derivative_exp

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianTotalBasisSet, self).calculate_residual(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_coefficients()

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianTotalBasisSet, self).calculate_residual(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_exponents()
        return(np.sum(derivative, axis=1))

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)



class GaussianTotalKL(DensityModel):
    def __init__(self, element_name, grid, file_path):
        DensityModel.__init__(self, element_name, grid, file_path)

    def cost_function(self,parameters, exponents=[], optimize_coeff=True):
        assert parameters.ndim == 1
        coefficients = np.copy(parameters)
        exponents = np.ravel(np.array(exponents))

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)

        #Ignore dvision by zero
        old_err_state = np.seterr(divide='raise')
        ignored_states = np.seterr(**old_err_state)

        electron_density = np.ravel(np.copy(self.electron_density))
        gaussian_density_divided_by_electron_density = np.divide(gaussian_density , electron_density)
        assert electron_density.shape[0] == gaussian_density.shape[0]
        assert gaussian_density_divided_by_electron_density.shape[0] == electron_density.shape[0]

        log_of_above = np.log(gaussian_density_divided_by_electron_density)
        log_of_above = np.nan_to_num(log_of_above)
        multiply_together = np.multiply(gaussian_density , log_of_above)
        multiply_together = np.nan_to_num(multiply_together)
        integrate_it = np.trapz(y=np.ravel(multiply_together), x=np.ravel(self.grid))
        return integrate_it

    def derivative_of_cost_function(self, parameters, exponents=[], optimize_coeff=True):
        derivative_of_all_coefficients = np.empty(parameters.shape[0])

        coefficients = np.copy(parameters)
        exponents = exponents

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        log_gaussian_density = np.log(gaussian_density)
        log_gaussian_density = np.nan_to_num(log_gaussian_density)
        log_electron_density = np.ravel(np.nan_to_num(np.log(self.electron_density)))
        for i in range(0, parameters.shape[0]):
            derivative = 0

            exponent_alpha = np.exp(-exponents[i] * np.power(np.ravel(self.grid), 2.0))
            derivative += np.trapz(y=np.multiply(exponent_alpha, log_gaussian_density), x=np.ravel(self.grid))
            derivative += np.sqrt(np.pi) / exponents[i]
            derivative -= np.trapz(y=np.ravel(np.multiply(log_electron_density, exponent_alpha)), x=np.ravel(self.grid))
            derivative_of_all_coefficients[i] = derivative

        return derivative_of_all_coefficients

class GaussianCoreBasisSet(DensityModel):
    def __init__(self, element_symbol, grid, file_path):
        DensityModel.__init__(self, element_symbol, grid, file_path)

    def create_model(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        assert parameters.ndim == 1
        def check_dimension_and_shape():
            assert exponents.ndim == 1
            assert np.shape(exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(exponential)[1] == np.shape(exponents)[0]
            assert exponential.ndim == 2
            assert coefficients.ndim == 1
            assert (coefficients.shape)[0] == exponential.shape[1]
            assert gaussian_density.ndim == 1
            assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)

        residual_squared = np.power(residual, 2.0)
        return(np.sum(residual_squared))

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True, optimize_coeff=False, optimize_exp=False):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_wrt_exponents():
                derivative_exp = []
                for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                    exponent = exponents[index]
                    g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                    derivative = -f_function * np.ravel(g_function)
                    derivative_exp.append(np.ravel(derivative))
                assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
                return(derivative_exp)

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(parameters, num_of_basis_funcs, [], [], optimize_both, optimize_coeff, optimize_exp)

            f_function = 2.0 * residual
            derivative = []

            derivative_coeff = derivative_wrt_coefficients()
            derivative_exp = derivative_wrt_exponents()
            derivative = derivative + derivative_coeff
            derivative = derivative + derivative_exp

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(coefficients, num_of_basis_funcs, exponents, [], optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_coefficients()

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianCoreBasisSet, self).calculate_residual_based_on_core(exponents, num_of_basis_funcs, [], coefficients, optimize_both, optimize_coeff, optimize_exp)
            f_function = 2.0 * residual
            derivative = derivative_wrt_exponents()
        return(np.sum(derivative, axis=1))

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)


class GaussianValenceBasisSet(DensityModel):
    def __init__(self, element_symbol, grid, file_path):
        DensityModel.__init__(self, element_symbol, grid, file_path)

    def create_model(self, parameters, num_of_s_funcs, num_of_p_funcs, optimize_both=True, optimize_coeff=False, optimize_exp=False):
        s_coefficients = parameters[:num_of_s_funcs]
        s_exponents = parameters[num_of_s_funcs: 2 * num_of_s_funcs]

        p_coefficients = parameters[num_of_s_funcs * 2 : num_of_s_funcs * 2 + num_of_p_funcs]
        p_exponents = parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

        s_exponential = np.exp(-s_exponents * np.power(self.grid, 2.0))
        p_exponential = np.exp(-p_exponents * np.power(self.grid, 2.0))

        def check_type_and_dimensions():
            assert type(s_coefficients).__module__ == np.__name__
            assert type(s_exponents).__module__ == np.__name__
            assert type(p_coefficients).__module__ == np.__name__
            assert type(p_exponents).__module__ == np.__name__

            assert s_exponents.ndim == 1; assert p_exponents.ndim == 1;
            assert np.shape(p_exponential)[1] == np.shape(p_exponents)[0]
            assert np.shape(s_exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(p_exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(s_exponential)[1] == np.shape(s_exponents)[0]

            assert s_exponential.ndim == 2; assert p_exponential.ndim == 2
            assert s_coefficients.ndim == 1
            assert (s_coefficients.shape)[0] == s_exponential.shape[1]

            assert s_gaussian_model.ndim == 1
            assert np.shape(s_gaussian_model)[0] == np.shape(self.grid)[0]


        s_gaussian_model = np.dot(s_exponential, s_coefficients)
        p_gaussian_model = np.dot(p_exponential, p_coefficients)
        p_gaussian_model = np.ravel(p_gaussian_model)  * np.ravel(np.power(self.grid, 2.0))

        check_type_and_dimensions()
        return(s_gaussian_model + p_gaussian_model)

    def cost_function(self, parameters, num_of_s_funcs, num_of_p_funcs, optimize_both=True, optimize_coeff=False, optimize_exp=False):
        assert type(parameters).__module__ == np.__name__
        assert isinstance(num_of_s_funcs, int)
        assert isinstance(num_of_p_funcs, int)

        s_coefficients = parameters[:num_of_s_funcs]
        s_exponents = parameters[num_of_s_funcs:2 * num_of_s_funcs]

        p_coefficients = parameters[num_of_s_funcs * 2 :num_of_s_funcs * 2 + num_of_p_funcs]
        p_exponents = parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

        residual = super(GaussianValenceBasisSet, self).calculate_residual_based_on_valence(parameters, num_of_s_funcs, num_of_p_funcs)
        residual_squared = np.power(residual, 2.0)

        return(np.sum(residual_squared))

    def derivative_of_cost_function(self, parameters, num_of_s_funcs, num_of_p_funcs, optimize_both=True, optimize_coeff=False, optimize_exp=False):
        s_coefficients = parameters[:num_of_s_funcs]
        s_exponents = parameters[num_of_s_funcs:2 * num_of_s_funcs]

        p_coefficients = parameters[num_of_s_funcs * 2 :num_of_s_funcs * 2 + num_of_p_funcs]
        p_exponents = parameters[num_of_s_funcs * 2 + num_of_p_funcs:]

        residual = super(GaussianValenceBasisSet, self).calculate_residual_based_on_valence(parameters, num_of_s_funcs, num_of_p_funcs)
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

    def create_cofactor_matrix(self, s_exponents, p_exponents):
        exponential_s = np.exp(-1.0 * s_exponents * np.power(self.grid, 2.0))
        assert(exponential_s.shape[1] == len(np.ravel(s_exponents)))
        assert(exponential_s.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential_s) == 2

        exponential_p = np.power(self.grid, 2.0) * np.exp(-1.0 * p_exponents * np.power(self.grid, 2.0))
        assert(exponential_p.shape[1] == len(np.ravel(p_exponents)))
        assert(exponential_p.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential_p) == 2

        exponential = np.concatenate((exponential_s, exponential_p), axis=1)
        assert exponential_p.shape[1] + exponential_s.shape[1] == len(np.ravel(s_exponents)) + len(np.ravel(p_exponents))
        assert(exponential_p.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)


if __name__ == "__main__":
    # Grab Element and the file path
    ELEMENT = "BE"
    ATOMIC_NUMBER = 4
    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT + ".slater"

    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 200; NUMBER_OF_DIFFUSED_PTS = 300
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    # Create a total gaussian basis set
    be = GaussianTotalBasisSet(ELEMENT, column_grid_points, file_path)
    fitting_object = Fitting(be)

    c = fitting_object.optimize_using_l_bfgs(np.array([np.random.random()*10 for x in range(0, be.UGBS_s_exponents.shape[0])]),be.UGBS_s_exponents.shape[0],np.array(be.UGBS_s_exponents),[], False, True, False)
    param = np.concatenate((c, be.UGBS_s_exponents))

    model = be.create_model(param, param.shape[0]/2)
    print(be.measure_error_by_integration_of_difference(be.electron_density, model))
    print(be.measure_error_by_integration_of_difference(be.electron_density, model))
    print(be.integrate_model_using_trapz(model), be.integrate_model_using_trapz(be.electron_density))
    # Save Results
    """
    import os
    directory_to_save_results = os.path.dirname(__file__).rsplit('/', 2)[0] + "/fitting/examples"
    #be.graph_and_save_the_results(directory_to_save_results)
    """
    #KL
    beKL = GaussianTotalKL(ELEMENT, column_grid_points, file_path)
    fitting_objectKL = Fitting(beKL)
    print(c)
    noise = np.random.normal(0, 1, c.shape[0])
    coeff = fitting_objectKL.optimize_using_l_bfgs(noise, np.array(beKL.UGBS_s_exponents))
    print(coeff)
    param = np.concatenate((coeff, beKL.UGBS_s_exponents))
    model = be.create_model(param, param.shape[0]/2)
    print(be.measure_error_by_integration_of_difference(be.electron_density, model))
    print(be.measure_error_by_integration_of_difference(be.electron_density, model))
    print(be.integrate_model_using_trapz(model), be.integrate_model_using_trapz(be.electron_density))

    # Fit Model Using Greedy Algorithm
    #fitting_object.forward_greedy_algorithm(2.0, 0.01, np.copy(be.electron_density), maximum_num_of_functions=100)
    #fitting_object.find_best_UGBS_exponents(1, p=1.5)
    #fitting_object.analytically_solve_objective_function(be.electron_density, 1.0)
    #be.generation_of_UGBS_exponents(1.25)


    # CORE
    #be_core = GaussianCoreBasisSet(ELEMENT, column_grid_points, file_path)
    #Create fitting object based on core and fit the model using the greedy algorithm
    #fitting_object_core = Fitting(be_core)
    #fitting_object_core.forward_greedy_algorithm(2.0, 0.01, np.copy(be_core.electron_density_core),maximum_num_of_functions=100)


    # VALENCE
    #be_valence = GaussianValenceBasisSet(element, column_grid_points, file_path)
    #fitting_object_valence = Fitting(be_valence)
    #fitting_object_valence.forward_greedy_algorithm_valence(2.0, 0.01)
    #fitting_object_valence.forward_greedy_algorithm(2.0, 0.01, np.copy(be_valence.electron_density_core), maximum_num_of_functions=100)

    """
    c1 = [x for x in range(0, 100)]
    WEIGHTS = [np.ones(np.shape(fitting_object_valence.model_object.grid)[0]), np.ravel(elec_d), np.ravel(np.power(elec_d, 2.0))]
    d1 = [x for x in range(0, 100)]

    def func(c1, d1):
        grid = np.copy(fitting_object_valence.model_object.grid)
        electron_density = np.copy(fitting_object_valence.model_object.electron_density_valence)
        weights = np.ravel(WEIGHTS[2])
        UGBS_S = fitting_object_valence.model_object.UGBS_s_exponents
        alpha_s = UGBS_S[np.random.randint(0, np.shape(UGBS_S)[0])]
        UGBS_p = fitting_object_valence.model_object.UGBS_p_exponents
        alpha_p = UGBS_p[np.random.randint(0, np.shape(UGBS_p)[0])]

        hey = np.log(electron_density) - np.log(c1 * np.exp(-alpha_s*np.power(grid, 2.0)) + d1*np.power(grid, 2.0) * np.exp(-alpha_p*np.power(grid,2.0)))
        weights = np.reshape(weights, (np.shape(weights)[0], 1))
        weights = np.repeat(weights, 100, axis=1)
        print(weights.shape)
        as2 = weights * hey
        as2 *= np.exp(-alpha_s*np.power(grid, 2.0)) / (c1 * np.exp(-alpha_s*np.power(grid, 2.0)) + d1*np.power(grid, 2.0) * np.exp(-alpha_p*np.power(grid,2.0)))

        return np.nansum(as2,axis = 0)


    as2 = func(c1, d1)


    plt.plot(c1, as2, 'r--')
    plt.show()
    """