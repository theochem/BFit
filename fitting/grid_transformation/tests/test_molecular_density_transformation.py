import unittest
from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
from fitting.grid_transformation.molecular_density_transformation import Molecular_Density_Transformation
from fitting.density.radial_grid import Radial_Grid
from fitting.fit.model import Fitting
import numpy as np
from scipy.integrate import quad, dblquad, tplquad
import sympy as sp
from sympy.abc import a, b, c, d, r, x, y, z
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nose.plugins.attrib import attr

class Default_Molecular_Density_Transformation_One_Be_Atom(unittest.TestCase):
    def set_up_grid(self):
        ATOMIC_NUMBER = 4
        radial_grid = Radial_Grid(ATOMIC_NUMBER)

        NUMBER_OF_CORE_POINTS = 200; NUMBER_OF_DIFFUSED_PTS = 300
        row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
        column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
        return row_grid_points, column_grid_points

    def set_up_fitting_objects(self, element, file_path, column_grid_points):
        be_obj = GaussianTotalBasisSet(element, column_grid_points, file_path)
        fitting_obj = Fitting(be_obj)
        return be_obj, fitting_obj

    def set_up_fitting_parameters(self, number_of_exps, element_obj, fitting_obj):
        exponents = element_obj.UGBS_s_exponents[0:number_of_exps]
        cofactor_matrix = element_obj.create_cofactor_matrix(exponents)
        coeffs = fitting_obj.optimize_using_nnls(cofactor_matrix)
        parameters = fitting_obj.optimize_using_l_bfgs(np.append(coeffs,exponents), len(coeffs))
        coeffs = parameters[0:len(coeffs)]
        exponents = parameters[len(coeffs):]
        return coeffs, exponents

    def setUp(self):
        ELEMENT = "BE"
        file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT + ".slater"
        row_grid_points, column_grid_points = self.set_up_grid()

        self.be, self.fitting_object = self.set_up_fitting_objects(ELEMENT, file_path, column_grid_points)
        self.coeffs_25, self.exps_25 = self.set_up_fitting_parameters(25, self.be, self.fitting_object)

        number_of_coordinates = 2
        displacement_vector =  np.array([0 for x in range(0, len(self.coeffs_25))])
        self.molecular_dens_25_coeffs = Molecular_Density_Transformation(np.reshape(self.coeffs_25, (-1, len(self.coeffs_25))),
                                                          np.reshape(self.exps_25, (-1, len(self.coeffs_25))), number_of_coordinates,
                                                         displacement_vector)

        number_of_coordinates = 1
        self.coeff_2, self.exps_2 = self.set_up_fitting_parameters(2, self.be, self.fitting_object)
        self.molecular_dens_2_coeffs = Molecular_Density_Transformation(np.reshape(self.coeff_2[:len(self.coeff_2)], (-1, len(self.coeff_2))),
                                                          np.reshape(self.exps_2, (-1, 2)), number_of_coordinates,
                                                          displacement_vector)


class Test_Integration_of_One_Gaussian_Function(Default_Molecular_Density_Transformation_One_Be_Atom):
    def test_1D_integration_of_one_gaussian(self):
        gaussian_coefficient = self.coeffs_25[int(np.random.random() * 25)]
        gaussian_exponent = self.exps_25[int(np.random.random() * 25)]

        self.molecular_dens_25_coeffs.dimension = 1
        one_d_integration = self.molecular_dens_25_coeffs.integrate_a_gaussian_over_the_reals(gaussian_coefficient, gaussian_exponent)
        scipy_integrate = quad(lambda  x: gaussian_coefficient * np.exp(-gaussian_exponent * x**2), -np.inf, np.inf)
        assert np.abs(one_d_integration - scipy_integrate[0]) < 1e-8

    def test_2D_integration_of_one_gaussian(self):
        gaussian_coefficient = self.coeffs_25[int(np.random.random() * 25)]
        gaussian_exponent = self.exps_25[int(np.random.random() * 25)]

        self.molecular_dens_25_coeffs.dimension = 2
        two_d_integration = self.molecular_dens_25_coeffs.integrate_a_gaussian_over_the_reals(gaussian_coefficient, gaussian_exponent)
        scipy_integration = dblquad(lambda x, y : gaussian_coefficient * np.exp(-gaussian_exponent * np.sqrt(x**2 + y**2)**2), -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
        assert np.abs(two_d_integration - scipy_integration[0]) < 1e-8

    @attr(speed='slow')
    def test_3D_integration_one_gaussian(self):
        gaussian_coefficient = self.coeffs_25[int(np.random.random() * 25)]
        gaussian_exponent = self.exps_25[int(np.random.random() * 25)]

        self.molecular_dens_25_coeffs.dimension = 3
        three_d_integration = self.molecular_dens_25_coeffs.integrate_a_gaussian_over_the_reals(gaussian_coefficient, gaussian_exponent)

        scipy_integration = tplquad(lambda x, y, z : gaussian_coefficient * np.exp(-gaussian_exponent * np.sqrt(x**2 + y**2 + z**2)**2),
                                                                                   -np.inf, np.inf, lambda x: -np.inf, lambda  x: np.inf,
                                                                                   lambda x, y: -np.inf, lambda x , y:np.inf)
        assert np.abs(three_d_integration - scipy_integration[0]) < 1e-8

class Test_Integration_of_ProMolecular_Density(Default_Molecular_Density_Transformation_One_Be_Atom):
    def test_1D_integration_of_promolecular_density(self):
        self.molecular_dens_25_coeffs.dimension = 1
        integration_1_d_space = self.molecular_dens_25_coeffs.integrate_promolecular_density()

        #Analytical Solution
        analytically_sol = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            analytically_sol += self.coeffs_25[index_gaussian] * (np.pi / self.exps_25[index_gaussian])**0.5
        assert integration_1_d_space == analytically_sol

        #Scipy Solution
        scipy_solution = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            scipy_solution += quad(lambda x: self.coeffs_25[index_gaussian] * np.exp(-self.exps_25[index_gaussian] * np.sqrt(x**2)**2),
                                      -np.inf, np.inf)[0]
        assert np.abs(scipy_solution - integration_1_d_space) < 1e-9

    def test_2D_integration_of_promolecular_density(self):
        integration_2_d_space = self.molecular_dens_25_coeffs.integrate_promolecular_density()

        #Analytical Solution
        analytically_sol = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            analytically_sol += self.coeffs_25[index_gaussian] * (np.pi / self.exps_25[index_gaussian])
        assert integration_2_d_space == analytically_sol

        #Scipy Solution
        scipy_solution = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            scipy_solution += dblquad(lambda x, y : self.coeffs_25[index_gaussian] * np.exp(-self.exps_25[index_gaussian] * np.sqrt(x**2 + y**2)**2),
                                      -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
        assert np.abs(scipy_solution - integration_2_d_space) < 1e-9

    @attr(speed="slow")
    def test_3D_integration_of_promolecular_density(self):
        self.molecular_dens_25_coeffs.dimension = 3
        integration_3_d_space = self.molecular_dens_25_coeffs.integrate_promolecular_density()

        #Analyticall Solution
        analytically_sol = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            analytically_sol += self.coeffs_25[index_gaussian] * (np.pi / self.exps_25[index_gaussian])**(3/2)
        assert integration_3_d_space == analytically_sol

        #Scipy Solution
        scipy_solution = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            if self.coeffs_25[index_gaussian] != 0.0 and self.coeffs_25[index_gaussian] != 0.0:
                scipy_solution += tplquad(lambda x, y, z : self.coeffs_25[index_gaussian] * np.exp(-self.exps_25[index_gaussian] * np.sqrt(x**2 + y**2 + z**2)**2),
                                                                                       -np.inf, np.inf, lambda x: -np.inf, lambda  x: np.inf,
                                                                                       lambda x, y: -np.inf, lambda x , y:np.inf)[0]
        assert np.abs(scipy_solution - integration_3_d_space) < 1e-4

class Test_Helper_Functions_for_Transformation_Coords(Default_Molecular_Density_Transformation_One_Be_Atom):
    def test_norm_one_dimension(self):
        self.molecular_dens_25_coeffs.dimension = 1
        x_value = 5.
        index = 0 #index has to be less than dimension which is one
        norm_value = self.molecular_dens_25_coeffs.get_norm(x_value, index)
        assert norm_value == 0.

    def test_norm_two_dimensions(self):
        self.molecular_dens_25_coeffs.dimension = 2
        x_value = [5., 12.]
        index = 0
        norm_value = self.molecular_dens_25_coeffs.get_norm(x_value, index)
        assert norm_value == 0.

        index = 1
        self.molecular_dens_25_coeffs.displacement_vector = [10., 50.]
        norm_value = self.molecular_dens_25_coeffs.get_norm(x_value, index)
        assert norm_value == (10. - 5.)**2


    def test_norm_three_dimensions(self):
        self.molecular_dens_25_coeffs.dimension = 3
        x_value = [10., 25. ,50.]
        index = 0
        assert self.molecular_dens_25_coeffs.get_norm(x_value, index) == 0

        index = 1
        self.molecular_dens_25_coeffs.displacement_vector = [52., -12., -5.]
        norm_value = self.molecular_dens_25_coeffs.get_norm(x_value, index)
        assert norm_value == (52. - 10.)**2

        index = 2
        norm_value = self.molecular_dens_25_coeffs.get_norm(x_value, index)
        assert norm_value == (52. - 10.)**2 + (25. + 12.)**2

class Test_Numerator_Of_Transformation_Method_In_One_Dimensions(Default_Molecular_Density_Transformation_One_Be_Atom):
    def check_numerator_of_transformation_one_dimension(self, point):
        """This test is done using sympy
        """
        self.molecular_dens_2_coeffs.dimension = 1
        self.molecular_dens_2_coeffs.displacement_vector = [1.]

        coordinate = upper_bound = point
        number_of_coefficients = 2
        coefficients = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents = self.molecular_dens_2_coeffs.exponents_per_atom[0]

        ################
        #Sympy Solution
        ################
        func_1 = coefficients[0] * sp.exp(- exponents[0] * (x - self.molecular_dens_2_coeffs.displacement_vector[0])**2)
        func_2 = coefficients[1] * sp.exp(- exponents[1] * (x - self.molecular_dens_2_coeffs.displacement_vector[0])**2)
        sympy_solution = sp.integrate(func_1, (x, -sp.oo, upper_bound)).evalf() +  sp.integrate(func_2, (x, -sp.oo, upper_bound)).evalf()

        #####################
        #Analytical Solution
        #####################
        prefactor_1 = coefficients[0] * 0.5 * np.sqrt(np.pi / exponents[0])
        prefactor_2 = coefficients[1] * 0.5 * np.sqrt(np.pi / exponents[1])
        coordinate = upper_bound
        norm_of_radius_1 = (coordinate - self.molecular_dens_2_coeffs.displacement_vector[0])**2
        index = 0
        analytical_solution = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1, exponents[0], norm_of_radius_1, coordinate, index)
        analytical_solution += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2, exponents[1], norm_of_radius_1,coordinate, index)

        assert np.abs(analytical_solution - sympy_solution) < 1e-4

    def test_numerator_of_transformation_one_dimension_positive_big_number(self):
        number = 10000.0
        self.check_numerator_of_transformation_one_dimension(number)

    def test_numerator_of_transformation_one_dimension_positive_small_number(self):
        number = 0.005
        self.check_numerator_of_transformation_one_dimension(number)

    def test_numerator_of_transformation_one_dimension_negative_big_number(self):
        number = -10000.0
        self.check_numerator_of_transformation_one_dimension(number)

    def test_numerator_of_transformation_one_dimension_negative_small_number(self):
        number = -0.00005
        self.check_numerator_of_transformation_one_dimension(number)

class Test_Numerator_Of_Transformation_Method_In_Two_Dimensions(Default_Molecular_Density_Transformation_One_Be_Atom):
    def check_numerator_of_transformation_two_dimensions(self, point):
        self.molecular_dens_2_coeffs.dimension = 2
        self.molecular_dens_2_coeffs.displacement_vector = [1., 2.]
        coefficients = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents = self.molecular_dens_2_coeffs.exponents_per_atom[0]

        ####################
        # Sympy Solution
        ####################
        func_1 = coefficients[0] * sp.exp(- exponents[0] * ((x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2))
        func_2 = coefficients[1] * sp.exp(- exponents[1] * ((x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2))
        transformed_point_x = sp.integrate(func_1, (x, -sp.oo, point[0]), (y, -sp.oo, sp.oo)) + sp.integrate(func_2, (x, -sp.oo, point[0]), (y, -sp.oo, sp.oo))

        func_1_at_new_point = coefficients[0] * sp.exp(- exponents[0] * ((transformed_point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 \
                                                                         + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2))
        func_2_at_new_point = coefficients[1] * sp.exp(- exponents[1] * ((transformed_point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 \
                                                                         + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2))
        transformed_point_y = sp.integrate(func_1_at_new_point, (y, -sp.oo, point[1])) + sp.integrate(func_2_at_new_point, (y, -sp.oo, point[1]))

        #print(sp.integrate(c*sp.exp(-a *((x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (y-self.molecular_dens_2_coeffs.displacement_vector[1])**2)), (x, y)))

        #####################
        # Analytical Solution
        #####################
        prefactor_1 = coefficients[0] * 0.5 * (np.pi / exponents[0])
        prefactor_2 = coefficients[1] * 0.5 * (np.pi / exponents[1])
        prefactor_1_y = coefficients[0] * 0.5 * (np.pi / exponents[0])**(1/2)
        prefactor_2_y = coefficients[1] * 0.5 * (np.pi / exponents[1])**(1/2)

        norm_of_radius_1 = (point[0] - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (point[1] - self.molecular_dens_2_coeffs.displacement_vector[1])**2
        point_x = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1, exponents[0], norm_of_radius_1, point[0], 0)
        point_x += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2, exponents[1], norm_of_radius_1, point[0], 0)

        norm_of_radius_1 = (point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (point[1] - self.molecular_dens_2_coeffs.displacement_vector[1])**2
        point_y = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1_y, exponents[0], norm_of_radius_1, point[1], 1)
        point_y += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2_y, exponents[1], norm_of_radius_1, point[1], 1)

        assert np.abs(point_x - transformed_point_x.evalf()) < 1e-5
        assert  np.abs(point_y - transformed_point_y.evalf()) < 1e-5

    def test_numerator_of_transformed_two_dimensions_big_x_vector(self):
        point = [1000., 3.]
        self.check_numerator_of_transformation_two_dimensions(point)

    def test_numerator_of_transformed_two_dimensions_big_y_vector(self):
        point = [3., 1000.]
        self.check_numerator_of_transformation_two_dimensions(point)

    def test_numerator_of_transformation_two_dimensions_big_vector(self):
        point = [2000., 1000.]
        self.check_numerator_of_transformation_two_dimensions(point)

    def test_numerator_of_transformation_two_dimensions_small_vector(self):
        point = [0.05, 0.5]
        self.check_numerator_of_transformation_two_dimensions(point)

    def test_numerator_of_transformation_two_dimensions_neg_big_vector(self):
        point = [-1000., -2000.]
        self.check_numerator_of_transformation_two_dimensions(point)

    def test_numerator_of_transformation_two_dimensions_neg_small_vector(self):
        point = [-0.5, -0.001]
        self.check_numerator_of_transformation_two_dimensions(point)

class Test_Numerator_Of_Transformation_Method_In_Three_Dimensions(Default_Molecular_Density_Transformation_One_Be_Atom):
    @attr(speed="slow")
    def test_numerator_of_transformation_in_three_dimensions(self):
        self.molecular_dens_2_coeffs.dimension = 3
        self.molecular_dens_2_coeffs.displacement_vector = [1., 2., 5.]
        coefficients = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents = self.molecular_dens_2_coeffs.exponents_per_atom[0]
        point = [10., 20., 5.]
        ####################
        # Sympy Solution
        ####################
        func_1 = coefficients[0] * sp.exp(- exponents[0] * ((x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2\
            + (z - self.molecular_dens_2_coeffs.displacement_vector[2])**2))
        func_2 = coefficients[1] * sp.exp(- exponents[1] * ((x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2)\
            + (z - self.molecular_dens_2_coeffs.displacement_vector[2])**2)
        transformed_point_x = sp.integrate(func_1, (x, -sp.oo, point[0]), (y, -sp.oo, sp.oo), (z, -sp.oo, sp.oo)) + sp.integrate(func_2, (x, -sp.oo, point[0]), (y, -sp.oo, sp.oo),
            (z, -sp.oo, sp.oo))

        func_1_at_new_x_point = coefficients[0] * sp.exp(- exponents[0] * ((transformed_point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 \
                                                                         + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                                                                         + (z - self.molecular_dens_2_coeffs.displacement_vector[2])**2))
        func_2_at_new_x_point = coefficients[1] * sp.exp(- exponents[1] * ((transformed_point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 \
                                                                         + (y - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                                                                         + (z - self.molecular_dens_2_coeffs.displacement_vector[2])**2))
        transformed_point_y = sp.integrate(func_1_at_new_x_point, (y, -sp.oo, point[1]), (z, -sp.oo, sp.oo)) + sp.integrate(func_2_at_new_x_point, (y, -sp.oo, point[1]), (z, -sp.oo, sp.oo))

        func_1_at_new_y_point = coefficients[0] * sp.exp(- exponents[0] * ((transformed_point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 \
                                                                         + (transformed_point_y - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                                                                         + (z - self.molecular_dens_2_coeffs.displacement_vector[2])**2))
        func_2_at_new_y_point = coefficients[1] * sp.exp(- exponents[1] * ((transformed_point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 \
                                                                         + (transformed_point_y - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                                                                         + (z - self.molecular_dens_2_coeffs.displacement_vector[2])**2))
        transformed_point_z = sp.integrate(func_1_at_new_y_point, (z, -sp.oo, point[2])) + sp.integrate(func_2_at_new_y_point, -sp.oo, point[2])

        ###################
        # Analytical Sol
        ###################
        prefactor_1_x = coefficients[0] * 0.5 * (np.pi / exponents[0])**(3/2)
        prefactor_2_x = coefficients[1] * 0.5 * (np.pi / exponents[1])**(3/2)
        prefactor_1_y = coefficients[0] * 0.5 * (np.pi / exponents[0])
        prefactor_2_y = coefficients[1] * 0.5 * (np.pi / exponents[1])
        prefactor_1_z = coefficients[0] * 0.5 * (np.pi / exponents[0])**(1/2)
        prefactor_2_z = coefficients[1] * 0.5 * (np.pi / exponents[1])**(1/2)

        norm_of_radius_1 = (point[0] - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (point[1] - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                        + (point[2] - self.molecular_dens_2_coeffs.displacement_vector[2])**2
        point_x = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1_x, exponents[0], norm_of_radius_1, point[0], 0)
        point_x += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2_x, exponents[1], norm_of_radius_1, point[0], 0)

        norm_of_radius_1 = (point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (point[1] - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                        + (point[2] - self.molecular_dens_2_coeffs.displacement_vector[2])**2
        point_y = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1_y, exponents[0], norm_of_radius_1, point[0], 1)
        point_y += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2_y, exponents[1], norm_of_radius_1, point[0], 1)

        norm_of_radius_1 = (point_x - self.molecular_dens_2_coeffs.displacement_vector[0])**2 + (point_y - self.molecular_dens_2_coeffs.displacement_vector[1])**2 \
                        + (point[2] - self.molecular_dens_2_coeffs.displacement_vector[2])**2
        point_z = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1_z, exponents[0], norm_of_radius_1, point[0], 2)
        point_z += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2_z, exponents[1], norm_of_radius_1, point[0], 2)

        print(point_x, point_y, point_z, transformed_point_x, transformed_point_y, transformed_point_z)



class Test_Molecular_Density_Transformation_One_Be_Atom(Default_Molecular_Density_Transformation_One_Be_Atom):
    def test_new_thetas_two_basis_funcs_one_dimensions(self):
        self.molecular_dens_2_coeffs.dimension = 1
        self.molecular_dens_2_coeffs.displacement_vector = [0.0]
        x_values = [-5, -1, -.5, .1, .5, 1, 2, 3]
        coefficients_atom = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents_atom = self.molecular_dens_2_coeffs.exponents_per_atom[0]


        for x in x_values:
            theta_1_expected = self.molecular_dens_2_coeffs.transform_coordinates_to_hyper_cube([x])
            scipy_solution_num = 0
            scipy_solution_den = 0
            for index_gaussian in range(0, len(coefficients_atom)):
                scipy_solution_num += quad(lambda x: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2)**2),
                                          -np.inf, x)[0]
                scipy_solution_den += quad(lambda x: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2)**2),
                                          -np.inf, np.inf)[0]
            assert np.abs(theta_1_expected - scipy_solution_num/scipy_solution_den) < 1e-2

    def test_new_thetas_two_basis_funcs_two_dimensions(self):
        self.molecular_dens_2_coeffs.dimension = 2
        self.molecular_dens_2_coeffs.displacement_vector = [0.0, 0.0]
        x_values = [[-5, -5], [-1, -0.9], [-.5, .6],[.05, .01],[.0025, .009], [.1, .2], [.5, .5], [2, 1]]
        coefficients_atom = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents_atom = self.molecular_dens_2_coeffs.exponents_per_atom[0]


        for pt in x_values:
            theta_1, theta_2 = self.molecular_dens_2_coeffs.transform_coordinates_to_hyper_cube(pt)

            scipy_solution_theta1_num, scipy_solution_theta2_num = 0, 0
            scipy_solution_theta1_den, scipy_solution_theta2_den = 0, 0

            for index_gaussian in range(0, len(coefficients_atom)):

                scipy_solution_theta1_num += dblquad(lambda x, y: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2)**2),
                                          -np.inf, pt[0], lambda x: -np.inf, lambda x: np.inf)[0]

                scipy_solution_theta1_den += dblquad(lambda x, y : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2)**2),
                                      -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]


            assert np.abs(theta_1 - scipy_solution_theta1_num / scipy_solution_theta1_den) < 1e-2

            for index_gaussian in range(0, len(coefficients_atom)):

                scipy_solution_theta2_num += quad(lambda y: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(theta_1**2 + y**2)**2),
                                          -np.inf, pt[1])[0]

                scipy_solution_theta2_den += quad(lambda y : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(theta_1**2 + y**2)**2),
                                      -np.inf, np.inf)[0]

            assert  np.abs(theta_2 - scipy_solution_theta2_num / scipy_solution_theta2_den) < 1e-2

    @attr(speed="slow")
    def test_new_thetas_two_basis_funcs_three_dimensions(self):
        self.molecular_dens_2_coeffs.dimension = 3
        self.molecular_dens_2_coeffs.displacement_vector = [.0, .0, .0]
        x_values = [[-1, -0.9, -0.8], [-.5, -0.2, -.4],[.05, .01, 0.01],[.0025, .009, .006], [.1, .2, .15], [.5, .5, .5]]
        coefficients_atom = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents_atom = self.molecular_dens_2_coeffs.exponents_per_atom[0]

        for pt in x_values:
            theta_1, theta_2, theta_3 = self.molecular_dens_2_coeffs.transform_coordinates_to_hyper_cube(pt)
            scipy_solution_theta1_num, scipy_solution_theta2_num, scipy_solution_theta3_num = 0, 0, 0
            scipy_solution_theta1_den, scipy_solution_theta2_den, scipy_solution_theta3_den  = 0, 0, 0

            #THETA_1
            for index_gaussian in range(0, len(coefficients_atom)):

                scipy_solution_theta1_num += tplquad(lambda x, y, z: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2 + z**2)**2),
                                          -np.inf, pt[0], lambda x: -np.inf, lambda x: np.inf,  lambda x, y: -np.inf, lambda x , y: np.inf)[0]

                scipy_solution_theta1_den += tplquad(lambda x, y, z : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2 + z**2)**2),
                                                                                       -np.inf, np.inf, lambda x: -np.inf, lambda  x: np.inf,
                                                                                       lambda x, y: -np.inf, lambda x , y: np.inf)[0]

            assert np.abs(theta_1 - scipy_solution_theta1_num /scipy_solution_theta1_den) < 1e-3

            #THETA_2
            for index_gaussian in range(0, len(coefficients_atom)):

                scipy_solution_theta2_num += dblquad(lambda y, z: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(theta_1**2 + y**2 + z**2)**2),
                                          -np.inf, pt[1], lambda x: -np.inf, lambda x: np.inf)[0]

                scipy_solution_theta2_den += dblquad(lambda y, z : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(theta_1**2 + y**2 + z**2)**2),
                                      -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]

            assert np.abs(theta_2 - scipy_solution_theta2_num / scipy_solution_theta2_den) < 1e-3

            #THETA_3
            for index_gaussian in range(0, len(coefficients_atom)):
                scipy_solution_theta3_num += quad(lambda z: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(theta_1**2 + theta_2**2 + z**2)**2),
                                          -np.inf, pt[2])[0]

                scipy_solution_theta3_den += quad(lambda z : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(theta_1**2 + theta_2**2 + z**2)**2),
                                      -np.inf, np.inf)[0]

            assert np.abs(theta_3 - scipy_solution_theta3_num / scipy_solution_theta3_den) < 1e-3

if __name__ == "__main__":
    unittest.main()