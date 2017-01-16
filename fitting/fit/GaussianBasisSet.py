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



    """
    LIST_OF_ATOMS = ["be", "LI", "HE", "B", "C", "N", "O", "F", "NE", "CU", "BR", "AG"]
    ATOMIC_NUMBER_LIST = [4, 3, 2, 5, 6, 7, 8 ,9, 10, 29, 35, 47]
    from fitting.density.radial_grid import *
    import os
    for i, atom_name in enumerate(LIST_OF_ATOMS):
        atomic_number = ATOMIC_NUMBER_LIST[i]
        print(atomic_number, atom_name)
        file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples" + "\\" + atom_name

        # Create Grid Object
        NUMBER_OF_CORE_POINTS = 400; NUMBER_OF_DIFFUSED_PTS = 500
        #row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
        radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMBER_OF_CORE_POINTS,  NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
        row_grid_pts = np.copy(radial_grid.radii)
        column_grid_pts = np.reshape(row_grid_pts, (len(row_grid_pts), 1))

        # Create Total Gaussian Basis Set and Fitting Objects
        total_gaussian_basis_set = GaussianTotalBasisSet(atom_name, column_grid_pts, file_path)
        fitting_object = Fitting(total_gaussian_basis_set)



    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 400; NUMBER_OF_DIFFUSED_PTS = 500
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    # Create a total gaussian basis set
    be = GaussianTotalBasisSet(ELEMENT, column_grid_points, file_path)
    fitting_object = Fitting(be)


    # Fit Model Using Greedy Algorithm
    #fitting_object.forward_greedy_algorithm(2.0, 0.01, np.copy(be.electron_density), maximum_num_of_functions=100)
    #fitting_object.find_best_UGBS_exponents(1, p=1.5)
    #fitting_object.analytically_solve_objective_function(be.electron_density, 1.0)
    #be.generation_of_UGBS_exponents(1.25)


    # CORE
    be_core = GaussianCoreBasisSet(ELEMENT, column_grid_points, file_path)
    fitting_object_core = Fitting(be_core)

    def plot_core_density():
        plt.semilogy(np.ravel(be_core.grid), np.ravel(be_core.electron_density),  label="Electron Density")
        plt.semilogy(np.ravel(be_core.grid), np.ravel(be_core.electron_density_core), label="Core Electron Density")
        plt.legend()
        plt.title("Log Plot of Electron Density & Core Density")
        plt.xlabel("Radius from nucleus")
        plt.savefig("Core Electron Density Plot")
        plt.show()

    initial_exps = np.array([  2.50000000e-02 ,  5.00000000e-02 ,  1.00000000e-01 ,  2.00000000e-01,
                               4.00000000e-01 ,  8.00000000e-01 ,  1.60000000e+00 ,  3.20000000e+00,
                               6.40000000e+00 ,  1.28000000e+01 ,  2.56000000e+01 ,  5.12000000e+01,
                               1.02400000e+02 ,  2.04800000e+02 ,  4.09600000e+02 ,  8.19200000e+02,
                               1.63840000e+03 ,  3.27680000e+03 ,  6.55360000e+03 ,  1.31072000e+04,
                               2.62144000e+04 ,  5.24288000e+04 ,  1.04857600e+05 ,  2.09715200e+05,
                               4.19430400e+05 ,  8.38860800e+05 ,  1.67772160e+06 ,  3.35544320e+06,
                               6.71088640e+06 ,  1.34217728e+07 ,  2.68435456e+07 ,  5.36870912e+07,
                               1.07374182e+08 ,  2.14748365e+08 ,  4.29496730e+08 ,  8.58993459e+08,
                               1.71798692e+09 ,  3.43597384e+09])[20:]
    cofactor_matrix = be_core.create_cofactor_matrix(initial_exps)
    nnls_coeffs = fitting_object_core.optimize_using_nnls(cofactor_matrix)
    slsqp_opt_parameters = fitting_object_core.optimize_using_slsqp(np.append(nnls_coeffs, initial_exps), len(initial_exps))

    coeffs = np.array([  0.00000000e+00,   0.00000000e+00 ,  0.00000000e+00,   4.25912844e-01,
                       0.00000000e+00 ,  0.00000000e+00,   0.00000000e+00  , 3.09449679e+00,
                       2.91030895e+01 ,  5.23047943e+01,   7.34205506e+01,   6.75374558e+01,
                       5.95145093e+01 ,  4.37592889e+01,   3.35704286e+01,   2.34014079e+01,
                       1.72925903e+01 ,  1.18801679e+01,   8.73220442e+00,   5.94357796e+00,
                       4.39272469e+00 ,  2.95990732e+00,   2.20835184e+00,   1.47305069e+00,
                       1.10841180e+00 ,  7.35833088e-01,   5.50702168e-01,   3.76937065e-01,
                       2.59418902e-01 ,  2.12612761e-01,   9.56166882e-02,   1.50902095e-01,
                       1.59128983e-03 ,  9.79221884e-02,   2.79737616e-02,   0.00000000e+00,
                       0.00000000e+00 ,  6.44357952e-02])
    exps = np.array(\
                  [  2.50000000e-02 ,  5.00000000e-02 ,  1.00000000e-01 ,  2.00000000e-01,
                    4.00000000e-01 ,  8.00000000e-01 ,  1.60000000e+00 ,  3.20000000e+00,
                    6.40000000e+00 ,  1.28000000e+01 ,  2.56000000e+01 ,  5.12000000e+01,
                    1.02400000e+02 ,  2.04800000e+02 ,  4.09600000e+02 ,  8.19200000e+02,
                    1.63840000e+03 ,  3.27680000e+03 ,  6.55360000e+03 ,  1.31072000e+04,
                    2.62144000e+04 ,  5.24288000e+04 ,  1.04857600e+05 ,  2.09715200e+05,
                    4.19430400e+05 ,  8.38860800e+05 ,  1.67772160e+06 ,  3.35544320e+06,
                    6.71088640e+06 ,  1.34217728e+07 ,  2.68435456e+07 ,  5.36870912e+07,
                    1.07374182e+08 ,  2.14748365e+08 ,  4.29496730e+08 ,  8.58993459e+08,
                    1.71798692e+09 ,  3.43597384e+09])
    final_parameters = np.append(coeffs,exps)
    previous_model = be.create_model(final_parameters, 38)
    FACTOR = 1000.
    THRESHOLD = 0.0001
    print(fitting_object.forward_greedy_algorithm(FACTOR, THRESHOLD, chosen_electron_density=be.electron_density))
    #new_coeffs = fitting_object.forward_greedy_algorithm_from_initial_params(FACTOR, THRESHOLD, be.electron_density, coeffs, exps)
    #print(new_coeffs)

    # VALENCE
    #be_valence = GaussianValenceBasisSet(ELEMENT, column_grid_points, file_path)
    #fitting_object_valence = Fitting(be_valence)
    #fitting_object_valence.forward_greedy_algorithm_valence(2.0, 0.01)
    #fitting_object_valence.forward_greedy_algorithm(2.0, 0.01, np.copy(be_valence.electron_density_core), maximum_num_of_functions=100)
    """

