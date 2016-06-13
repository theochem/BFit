from nose.tools import assert_equals, assert_not_equal
import unittest
from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
from fitting.grid_transformation.molecular_density_transformation import Molecular_Density_Transformation
from fitting.density.radial_grid import Radial_Grid
from fitting.fit.model import Fitting
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad, dblquad, tplquad, nquad

class Test_Molecular_Density_Transformation_One_Atom(unittest.TestCase):

    def setUp(self):
        ELEMENT = "BE"
        ATOMIC_NUMBER = 4
        file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT + ".slater"

        radial_grid = Radial_Grid(ATOMIC_NUMBER)

        NUMBER_OF_CORE_POINTS = 200; NUMBER_OF_DIFFUSED_PTS = 300
        row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
        column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

        self.be = GaussianTotalBasisSet(ELEMENT, column_grid_points, file_path)
        self.fitting_object = Fitting(self.be)
        cofactor_matrix = self.be.create_cofactor_matrix(self.be.UGBS_s_exponents)
        coeff = self.fitting_object.optimize_using_nnls(cofactor_matrix)
        self.coeff_exp = self.fitting_object.optimize_using_l_bfgs(np.concatenate((coeff, self.be.UGBS_s_exponents)), len(coeff))

        self.elec_dens = self.be.create_model(self.coeff_exp, len(coeff))

        print(self.be.integrated_total_electron_density)
        print(self.be.integrate_model_using_trapz(self.elec_dens))
        print(self.coeff_exp[:len(coeff)], "Coefficients")
        print(self.coeff_exp[len(coeff):], "Exp")
        #plt.plot(self.be.electron_density)
        #plt.plot(self.elec_dens)
        #plt.show()

        number_of_coordinates = 2
        displacement_vector =  np.array([0 for x in range(0, len(coeff))])
        self.molecular_dens_25_coeffs = Molecular_Density_Transformation(self.elec_dens, np.reshape(self.coeff_exp[:len(coeff)], (-1, len(coeff))),
                                                          np.reshape(self.coeff_exp[len(coeff):], (-1, len(coeff))), number_of_coordinates,
                                                         displacement_vector)

        number_of_coordinates = 1
        cofactor_matrix = self.be.create_cofactor_matrix(self.be.UGBS_s_exponents[0:2])
        coeff = self.fitting_object.optimize_using_nnls(cofactor_matrix)
        self.coeff_exp_2 = self.fitting_object.optimize_using_l_bfgs(np.concatenate((coeff, self.be.UGBS_s_exponents[0:2])), 2)
        self.elec_dens_2_coeffs = self.be.create_model(self.coeff_exp_2, 2)
        self.molecular_dens_2_coeffs = Molecular_Density_Transformation(self.elec_dens_2_coeffs,np.reshape(self.coeff_exp_2[:2], (-1,2)),
                                                          np.reshape(self.coeff_exp_2[2:], (-1, 2)), number_of_coordinates,
                                                         displacement_vector)

    def test_integration_one_gaussian_over_reals(self):
        # This Test is Done Using Scipy
        gaussian_coefficient = self.coeff_exp[0]
        gaussian_exponent = self.coeff_exp[len(self.coeff_exp)//2]
        print(gaussian_exponent, gaussian_coefficient)

        ###################################
        ####### 1-Dimension ##############
        ##################################
        self.molecular_dens_25_coeffs.dimension = 1
        one_d_integration = self.molecular_dens_25_coeffs.integrate_of_one_gaussian_over_reals(gaussian_coefficient, gaussian_exponent)
        scipy_integrate = quad(lambda  x: gaussian_coefficient * np.exp(-gaussian_exponent * x**2), -np.inf, np.inf)
        assert np.abs(one_d_integration - scipy_integrate[0]) < 1e-8

        ###################################
        ###### 2-Dimensions ###############
        ###################################
        self.molecular_dens_25_coeffs.dimension = 2
        one_d_integration = self.molecular_dens_25_coeffs.integrate_of_one_gaussian_over_reals(gaussian_coefficient, gaussian_exponent)
        scipy_integraiton = dblquad(lambda x, y : gaussian_coefficient * np.exp(-gaussian_exponent * np.sqrt(x**2 + y**2)**2), -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
        assert np.abs(one_d_integration - scipy_integraiton[0]) < 1e-8

        ###################################
        ###### 3-Dimensions ###############
        ###################################
        self.molecular_dens_25_coeffs.dimension = 3
        one_d_integration = self.molecular_dens_25_coeffs.integrate_of_one_gaussian_over_reals(gaussian_coefficient, gaussian_exponent)

        scipy_integraiton = tplquad(lambda x, y, z : gaussian_coefficient * np.exp(-gaussian_exponent * np.sqrt(x**2 + y**2 + z**2)**2),
                                                                                   -np.inf, np.inf, lambda x: -np.inf, lambda  x: np.inf,
                                                                                   lambda x, y: -np.inf, lambda x , y:np.inf)
        print(nquad(lambda x, y, z : gaussian_coefficient * np.exp(-gaussian_exponent * np.sqrt(x**2 + y**2 + z**2)**2),
                    [[-np.inf, np.inf], [-np.inf, np.inf], [-np.inf, np.inf]]))

        print(scipy_integraiton, one_d_integration)


    def test_integration_entire_space(self):
        ##############################
        ####### 2-Dimensions #########
        ##############################
        integration_2_d_space = self.molecular_dens_25_coeffs.integration_of_molecular_density_over_space()

        analytically_sol = 0
        coefficients_atom = self.molecular_dens_25_coeffs.coefficients_per_atom[0]
        exponents_atom = self.molecular_dens_25_coeffs.exponents_per_atom[0]
        for index_gaussian in range(0, len(coefficients_atom)):
            analytically_sol += coefficients_atom[index_gaussian] * (np.pi / exponents_atom[index_gaussian])
        assert integration_2_d_space == analytically_sol

        scipy_solution = 0
        for index_gaussian in range(0, len(coefficients_atom)):
            scipy_solution += dblquad(lambda x, y : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2)**2),
                                      -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)[0]
        assert np.abs(scipy_solution - integration_2_d_space) < 1e-9


        ##############################
        ####### 1-Dimensions #########
        ##############################
        self.molecular_dens_25_coeffs.dimension = 1
        integration_1_d_space = self.molecular_dens_25_coeffs.integration_of_molecular_density_over_space()

        analytically_sol = 0
        for index_gaussian in range(0, len(coefficients_atom)):
            analytically_sol += coefficients_atom[index_gaussian] * (np.pi / exponents_atom[index_gaussian])**0.5
        assert integration_1_d_space == analytically_sol

        scipy_solution = 0
        for index_gaussian in range(0, len(coefficients_atom)):
            scipy_solution += quad(lambda x: coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2)**2),
                                      -np.inf, np.inf)[0]
        assert np.abs(scipy_solution - integration_1_d_space) < 1e-9

        ##############################
        ####### 3-Dimensions #########
        ##############################
        self.molecular_dens_25_coeffs.dimension = 3
        integration_3_d_space = self.molecular_dens_25_coeffs.integration_of_molecular_density_over_space()

        analytically_sol = 0
        for index_gaussian in range(0, len(coefficients_atom)):
            analytically_sol += coefficients_atom[index_gaussian] * (np.pi / exponents_atom[index_gaussian])**(3/2)
        assert integration_3_d_space == analytically_sol

        #scipy_solution = 0
        #for index_gaussian in range(0, len(coefficients_atom)):
        #    if coefficients_atom[index_gaussian] != 0.0 and exponents_atom[index_gaussian] != 0.0:
        #        scipy_solution += tplquad(lambda x, y, z : coefficients_atom[index_gaussian] * np.exp(-exponents_atom[index_gaussian] * np.sqrt(x**2 + y**2 + z**2)**2),
        #                                                                               -np.inf, np.inf, lambda x: -np.inf, lambda  x: np.inf,
        #                                                                               lambda x, y: -np.inf, lambda x , y:np.inf)[0]
        #assert np.abs(scipy_solution - integration_3_d_space) < 1e-4


    def test_new_thetas_two_basis_funcs(self):
        print(self.molecular_dens_2_coeffs.dimension)
        print(self.molecular_dens_2_coeffs.number_of_atoms)
        print(self.molecular_dens_2_coeffs.new_weights([0.1]))

if __name__ == "__main__":
    unittest.main()