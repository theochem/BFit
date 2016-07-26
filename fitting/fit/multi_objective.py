import numpy as np
from fitting.fit.model import DensityModel, Fitting

#########
# Changing teh Gaussian Total Basis set
# so it matches with scipy.minimize

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
        return(np.sum(residual_squared), self.derivative_of_cost_function(parameters,num_of_basis_funcs))

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

class GaussianTotalIntegrationObjective(DensityModel):
    def __init__(self, element_name, grid, file_path, atomic_number,lam=0.8):
        DensityModel.__init__(self, element_name, grid, file_path)
        self.lam = lam
        self.atomic_number = atomic_number

    def create_model(self,parameters, num_of_basis_funcs):
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

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]


        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]

        residual = super(GaussianTotalIntegrationObjective, self).calculate_residual(parameters, num_of_basis_funcs)

        integration_value = np.sum( coefficients * np.sqrt(np.pi / exponents) / ( 4 * exponents))
        residual_squared = np.power(residual, 2.0) + self.lam * (integration_value - self.atomic_number)**2
        return(np.sum(residual_squared), self.derivative_of_cost_function(parameters,num_of_basis_funcs))

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function) + inner_function * self.lam * (1/(4 * exp)) * np.sqrt(np.pi/exp)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_wrt_exponents():
            derivative_exp = []
            for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                exponent = exponents[index]
                g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -f_function * np.ravel(g_function)  +\
                                inner_function * self.lam * (-3 * coeff / 8) * np.sqrt(np.pi) * exponent**(-5/2)
                derivative_exp.append(np.ravel(derivative))
            assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
            return(derivative_exp)

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]

        residual = super(GaussianTotalIntegrationObjective, self).calculate_residual(parameters, num_of_basis_funcs)

        f_function = 2.0 * residual
        inner_function = 2. * (np.sum(coefficients * np.sqrt(np.pi) / (4 * np.power(exponents, 3/2))) - self.atomic_number)
        derivative = []

        derivative_coeff = derivative_wrt_coefficients()
        derivative_exp = derivative_wrt_exponents()
        derivative = derivative + derivative_coeff
        derivative = derivative + derivative_exp

        return(np.sum(derivative, axis=1))

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)

class GaussianSecondObjTrapz(DensityModel):
    def __init__(self, element_name, grid, file_path, atomic_number,lam=0.8):
        DensityModel.__init__(self, element_name, grid, file_path)
        self.lam = lam
        self.atomic_number = atomic_number

    def create_model(self,parameters, num_of_basis_funcs):
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

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]


        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]

        residual = super(GaussianSecondObjTrapz, self).calculate_residual(parameters, num_of_basis_funcs)

        gaussian_dens = self.create_model(parameters, num_of_basis_funcs)
        integration_value = np.trapz(y=np.abs(gaussian_dens - np.ravel(self.electron_density))\
                                       * np.ravel(np.power(self.grid, 2.)), x=np.ravel(self.grid))
        residual_squared = np.power(residual, 2.0) + self.lam * integration_value
        return np.sum(residual_squared)

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs):
        pass

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return(exponential)

class GaussianSecondObjSquared(DensityModel):
    def __init__(self, element_name, grid, file_path, atomic_number,lam=0.8, lam2=1.):
        DensityModel.__init__(self, element_name, grid, file_path)
        self.lam = lam
        self.atomic_number = atomic_number
        self.lam2 = lam2

    def create_model(self,parameters, num_of_basis_funcs):
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

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]


        exponential = np.exp(-exponents * np.power(self.grid, 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return(gaussian_density)

    def cost_function(self, parameters, num_of_basis_funcs):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]

        residual = super(GaussianSecondObjSquared, self).calculate_residual(parameters, num_of_basis_funcs)

        gaussian_dens = self.create_model(parameters, num_of_basis_funcs)
        integration_value = np.trapz(y=(gaussian_dens - np.ravel(self.electron_density))**2\
                                       * np.ravel(np.power(self.grid, 2.)), x=np.ravel(self.grid))
        residual_squared = self.lam2 * np.power(residual, 2.0) + self.lam * integration_value
        return(np.sum(residual_squared), self.derivative_of_cost_function(parameters, num_of_basis_funcs))

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = self.lam2 * f_function * np.ravel(g_function) +\
                                self.lam * 2 * np.trapz(y=(mod - np.ravel(self.electron_density))\
                                * (np.exp(-exp * np.ravel(np.power(self.grid, 2.))) )\
                                * np.ravel(np.power(self.grid, 2.)), x=np.ravel(self.grid))
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return(derivative_coeff)

        def derivative_wrt_exponents():
            derivative_exp = []
            for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                exponent = exponents[index]
                g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                derivative = -self.lam2 * f_function * np.ravel(g_function)  +\
                                self.lam * 2 * np.trapz(y=(mod - np.ravel(self.electron_density))\
                                * (np.exp(-exponent * np.ravel(np.power(self.grid, 2.))) ) \
                                * (-1 *coeff)\
                                * np.ravel(np.power(self.grid, 4.)), x=np.ravel(self.grid))
                derivative_exp.append(np.ravel(derivative))
            assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
            return(derivative_exp)

        coefficients = parameters[:num_of_basis_funcs]
        exponents = parameters[num_of_basis_funcs:]

        residual = super(GaussianSecondObjSquared, self).calculate_residual(parameters, num_of_basis_funcs)

        f_function = 2.0 * residual
        derivative = []
        mod = self.create_model(parameters, num_of_basis_funcs)
        derivative_coeff = derivative_wrt_coefficients()
        derivative_exp = derivative_wrt_exponents()
        derivative = derivative + derivative_coeff
        derivative = derivative + derivative_exp

        return(np.sum(derivative, axis=1))


    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
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
    NUMBER_OF_CORE_POINTS = 500; NUMBER_OF_DIFFUSED_PTS = 600
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    # Create a total gaussian basis set
    be = GaussianTotalBasisSet(ELEMENT, column_grid_points, file_path)
    fitting_object = Fitting(be)

    FACTOR = 10000.
    THRESHOLD = 0.00001
    LAMD = 0.25
    print("FACTOR - ", FACTOR, "LAMBDA - ", LAMD)
    print("Electron Density at r = 0 - ", be.electron_density[0])
    #coeffs = fitting_object.forward_greedy_algorithm(FACTOR, THRESHOLD, be.electron_density)
    #print(coeffs)

    changed_obj_function = GaussianTotalIntegrationObjective(ELEMENT, column_grid_points, file_path,
                                                             atomic_number=ATOMIC_NUMBER,
                                                             lam=LAMD)
    fitting_changed = Fitting(changed_obj_function)

    #coeffs = fitting_changed.forward_greedy_algorithm(FACTOR, THRESHOLD, changed_obj_function.electron_density)
    #print(coeffs)


    second_obj_func_added = GaussianSecondObjTrapz(ELEMENT, column_grid_points, file_path,
                                                             atomic_number=ATOMIC_NUMBER,
                                                             lam=LAMD)
    fitting_sec_obj_func = Fitting(second_obj_func_added)
    #coeffs = fitting_sec_obj_func.forward_greedy_algorithm(FACTOR, THRESHOLD, second_obj_func_added.electron_density)

    print("Squared Sec Objective Function")
    sec_obj_func_squared = GaussianSecondObjSquared(ELEMENT, column_grid_points, file_path,
                                                             atomic_number=ATOMIC_NUMBER,
                                                             lam=LAMD)
    fit_sec_obj_func = Fitting(sec_obj_func_squared)
    parameters = fit_sec_obj_func.forward_greedy_algorithm(FACTOR, 50., sec_obj_func_squared.electron_density)
    print(parameters)