import unittest
from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
from fitting.grid_transformation.molecular_density_transformation import Molecular_Density_Transformation
from fitting.density.radial_grid import Radial_Grid
from fitting.fit.model import Fitting
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad, dblquad, tplquad, nquad, trapz

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

    def test_3D_integration_of_promolecular_density(self):
        self.molecular_dens_25_coeffs.dimension = 3
        integration_3_d_space = self.molecular_dens_25_coeffs.integrate_promolecular_density()

        #Analyticall Solution
        analytically_sol = 0
        for index_gaussian in range(0, len(self.coeffs_25)):
            analytically_sol += self.coeffs_25[index_gaussian] * (np.pi / self.exps_25[index_gaussian])**(3/2)
        assert integration_3_d_space == analytically_sol

        #Scipy Solution
        #scipy_solution = 0
        #for index_gaussian in range(0, len(coefficients_atom)):
        #    if coefficients_atom[index_gaussian] != 0.0 and exponents_atom[index_gaussian] != 0.0:
        #        scipy_solution += tplquad(lambda x, y, z : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2 + z**2)**2),
        #                                                                               -np.inf, np.inf, lambda x: -np.inf, lambda  x: np.inf,
        #                                                                               lambda x, y: -np.inf, lambda x , y:np.inf)[0]
        #assert np.abs(scipy_solution - integration_3_d_space) < 1e-4

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

    def test_numerator_of_transformation_one_dimension(self):
        #scipy solution
        self.molecular_dens_2_coeffs.dimension = 1
        self.molecular_dens_2_coeffs.displacement_vector = [1.]

        upper_bound = 0.00000000001
        number_of_coefficients = 2
        scipy_solution = 0
        coefficients = self.molecular_dens_2_coeffs.coefficients_per_atom[0]
        exponents = self.molecular_dens_2_coeffs.exponents_per_atom[0]

        from scipy import integrate
        for i in range(0, number_of_coefficients):
            print(exponents, coefficients)
            scipy_solution += quad(lambda x: coefficients[i] * np.exp(-exponents[i] * (x - self.molecular_dens_2_coeffs.displacement_vector[0])**2),
                               -np.inf, upper_bound)[0]

        prefactor_1 = coefficients[0] * 0.5 * np.sqrt(np.pi / exponents[0])
        prefactor_2 = coefficients[1] * 0.5 * np.sqrt(np.pi / exponents[1])
        coordinate = upper_bound
        norm_of_radius_1 = (coordinate - self.molecular_dens_2_coeffs.displacement_vector[0])**2
        index = 0
        analytical_solution = self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_1, exponents[0], coordinate, norm_of_radius_1, index)
        analytical_solution += self.molecular_dens_2_coeffs.get_numerator_for_conditional_distribution(prefactor_2, exponents[1], coordinate, norm_of_radius_1, index)
        print(analytical_solution, scipy_solution)



    def test_denominator_of_transformation(self):
        pass

class Test_Molecular_Density_Transformation_One_Be_Atom(Default_Molecular_Density_Transformation_One_Be_Atom):
    def test_new_thetas_two_basis_funcs(self):
        #print(self.molecular_dens_2_coeffs.dimension)
        self.molecular_dens_2_coeffs.displacement_vector = [0.0]
        #print(self.molecular_dens_2_coeffs.number_of_atoms)
        #print(self.molecular_dens_2_coeffs.new_weights([10.]))

        """
        X = np.arange(-100, 100, 0.01)
        Y = []
        for p in X:
            Y.append(self.molecular_dens_2_coeffs.transform_coordinates_to_hyper_cube([p]))
        print(Y)
        plt.title("Grid Transformation from R to [0, 1] for Be of Two Gaussian Basis Funcs. Centered at 0")
        plt.plot(X, np.array(Y), "bo")
        plt.xlabel("X-values in R")
        plt.ylabel("Theta_1 values")
        plt.show()

        X = np.arange(-0.5, 0.5, 0.001)
        Y = []
        for p in X:
            Y.append(self.molecular_dens_2_coeffs.transform_coordinates_to_hyper_cube([p]))
        print(Y)
        plt.title("Grid Transformation from R to [0, 1] for Be of Two Gaussian Basis Funcs. Centered at 0")
        plt.plot(X, np.array(Y), "bo")
        plt.xlabel("X-values in R")
        plt.ylabel("Theta_1 values")
        plt.show()
        """

        ##########################
        ###### 2-Dimensions ######
        ##########################
        self.molecular_dens_2_coeffs.dimension = 2
        self.molecular_dens_2_coeffs.displacement_vector = [0.0, 0.0]
        #print(self.molecular_dens_2_coeffs.new_weights([0.0, 0.0]))


        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x = y = np.arange(-10.0, 10.0, .1)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.molecular_dens_2_coeffs.new_weights([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Zs = zs[:,0].reshape(X.shape)
        Zz = zs[:,1].reshape(X.shape)

        ax.scatter(X, Y, Zs,c='r', marker='o')
        ax.scatter(X, Y, Zz, c="b", marker='o')
        plt.show()

        x = y = np.arange(-10.0, 10.0, .01)
        X, Y = np.meshgrid(x, y)
        zs = np.array([self.molecular_dens_2_coeffs.new_weights([x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])
        Zs = zs[:,0].reshape(X.shape)
        Zz = zs[:,1].reshape(X.shape)

        plt.plot(Zs, Zz, "ro")
        plt.title("Grid Trans. from R to [0,1]^2 for Be of Two Gaussian Basis Funcs, Centered at 0")
        plt.xlabel("Theta_1(x_1)")
        plt.ylabel("Theta_2(x_2)")
        plt.show()
        """





        #########################
        ##### 3-Dimensions ######
        #########################
        self.molecular_dens_2_coeffs.dimension = 3
        self.molecular_dens_2_coeffs.displacement_vector = [0.0, 0.0, 0.0]
        #print(self.molecular_dens_2_coeffs.new_weights([0.0, 0.0, 10000.0]))
        """
        x = y = z = np.arange(-0.5, 0.5, .03)
        X, Y, Z = np.meshgrid(x, y, z)
        zs = np.array([self.molecular_dens_2_coeffs.new_weights([x,y,z]) for x,y,z in zip(np.ravel(X), np.ravel(Y), np.ravel(Z))])
        Zx = zs[:,0]
        Zy = zs[:,1]
        Zz = zs[:,2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Zx, Zy, Zz, c='r', marker='o')
        plt.title("Grid Trans. from R to [0,1]^3 for Be of Two Gaussian Basis Funcs, Cent. At 0")
        ax.set_xlabel("Theta_1")
        ax.set_ylabel("Theta_2")
        ax.set_zlabel("Theta_3")
        plt.show()
        """
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