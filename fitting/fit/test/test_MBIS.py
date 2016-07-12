import unittest
from fitting.fit.MBIS import update_coefficients
from fitting.density.atomic_slater_density import Atomic_Density
from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet
from scipy.integrate import quad, dblquad, tplquad, nquad, trapz
from scipy.misc import factorial
import sympy as sp
from sympy.abc import c, a
import numpy as np

#TODO LIST
#TODO: Error for test_one_coefficient_using_trapz has a really small error

class Test_MBIS_For_Coefficients(unittest.TestCase):
    def set_up_grid(self):
        ATOMIC_NUMBER = 4
        from fitting.density.radial_grid import Radial_Grid
        radial_grid = Radial_Grid(ATOMIC_NUMBER)
        NUMBER_OF_CORE_POINTS = 1000; NUMBER_OF_DIFFUSED_PTS = 1000
        row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
        column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))
        return column_grid_points

    def setUp(self):
        file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\be.slater"
        self.grid = self.set_up_grid()
        be = GaussianTotalBasisSet("be", self.grid, file_path)
        self.electron_density = be.electron_density

        be_atomic_dens = Atomic_Density(file_path, self.grid)
        self.slater_list_of_orbitals = be_atomic_dens.VALUES['orbitals']
        #print(be_atomic_dens.VALUES)
        self.slater_coeffs = be_atomic_dens.all_coeff_matrix('S')
        self.slater_basis_nums = be_atomic_dens.VALUES['basis_numbers']['S']
        self.slater_exps = be_atomic_dens.VALUES['orbitals_exp']['S']
        self.slater_orbitals_electron_array = be_atomic_dens.VALUES['orbitals_electron_array']
        #print(self.slater_coeffs,
        #      self.slater_basis_nums,
        #      self.slater_exps)

    def test_updating_one_coefficient_using_trapz(self):
        initial_coefficients = np.array([[5.0], [6.0], [10000.0], [200000.0]])
        constant_exponents = np.array([[20.] , [30.], [0.01], [1.5]])

        for x in range(0, initial_coefficients.shape[0]):
            new_coefficient = update_coefficients(np.array([initial_coefficients[x][0]]),
                                                  np.array([constant_exponents[x][0]]), self.electron_density, self.grid)
            expected_trapz_solution = (constant_exponents[x] / np.pi)**(3/2) *\
                                    np.trapz(y=np.ravel(self.electron_density), x=np.ravel(self.grid))
            assert np.abs(new_coefficient[0] - expected_trapz_solution) < 1e-1

    def test_zero_initial_coefficient(self):
        initial_coefficient = np.array([0.])
        try:
            new_coefficient = update_coefficients(initial_coefficient, np.array([0.]), self.electron_density, self.grid)
        except AssertionError:
            assert True

    def test_updating_two_coefficient_simple_initial_guess(self):
        initial_coefficients = np.array([1., 1.])
        constant_exponents = np.array([1., 2.])
        grid_squared = np.ravel(np.power(self.grid, 2.))
        new_coefficient_1, new_coefficient_2 = update_coefficients(initial_coefficients, constant_exponents, self.electron_density, self.grid)

        #################################
        ## Using Trapz and Masked Array
        #################################

        # First Coefficient
        integrand = np.ravel(self.electron_density) * np.exp(-1. * np.power(np.ravel(self.grid), 2.))
        masked_integrand = np.ma.array(integrand / (np.exp(-1 * grid_squared) + np.exp(-2. * grid_squared)))
        expected_trapz_solution_1 = (1. / np.pi)**(3/2) * np.trapz(y=masked_integrand, x=np.ravel(self.grid))
        assert np.abs(expected_trapz_solution_1 - new_coefficient_1) < 1e-5

        # Second Coefficient
        integrand_2 = np.ravel(self.electron_density) * np.exp(-2. * grid_squared)
        masked_integrand = np.ma.array(integrand / (np.exp(-1 * grid_squared) + np.exp(-2. * grid_squared)))
        expected_trapz_solution_2 = (2. / np.pi)**(3/2) * np.trapz(y=masked_integrand, x=np.ravel(self.grid))
        print(new_coefficient_2, expected_trapz_solution_2, np.abs(expected_trapz_solution_2 - new_coefficient_2) )
        assert np.abs(expected_trapz_solution_2 - new_coefficient_2) < 1.

        #################################
        ## Using Trapz and nan_to_num
        #################################

        # First Coefficient
        integrand = integrand / (np.exp(-1 * grid_squared) + np.exp(-2. * grid_squared))
        integrand_no_nan = np.nan_to_num(integrand)
        expected_trapz_solution_1 = (1. / np.pi)**(3/2) * np.trapz(y=integrand_no_nan, x=np.ravel(self.grid))
        assert np.abs(expected_trapz_solution_1 - new_coefficient_1) < 1e-5

        # Second Coefficient
        integrand_2 = integrand_2 / (np.exp(-1 * grid_squared) + np.exp(-2. * grid_squared))
        integrand_2_no_nan = np.nan_to_num(integrand_2)
        expected_trapz_solution_2 = (2. / np.pi)**(3/2) * np.trapz(y=masked_integrand, x=np.ravel(self.grid))
        assert np.abs(expected_trapz_solution_2 - new_coefficient_2) < 1.


    def test_updating_one_coefficient(self):
        initial_coefficient = np.array([5.0])
        constant_exponents = np.array([20.])
        new_coefficient = update_coefficients(initial_coefficient, constant_exponents, self.electron_density, self.grid)

        scipy_solution = 0
        def add_func(h, f, g):
            return lambda r: h(f(r), g(r))

        def scalar_mult_func(h, f, g):
            return lambda r : h(f(r), g)

        def norm_squared_func(f):
            return lambda r : (f(r))**2

        def add_stuff(f, exponent, coefficient):
            return lambda r : truediv(mul(f(r), np.exp(-exponent * r**2)), (coefficient * np.exp(-exponent * r**2)))
        from operator import add, mul, truediv

        from sympy.utilities import lambdify
        from sympy.abc import r

        phi_1s = None
        phi_2s = None
        for x in range(0, 2):
            for i in range(0, self.slater_coeffs.shape[0]):
                coefficient = self.slater_coeffs[i, x]
                exponent = np.ravel(self.slater_exps)[i]
                quantumNum = np.ravel(self.slater_basis_nums)[i]

                if i == 0:
                    if x == 0:
                        phi_1s = lambda r: coefficient * ((2 * exponent)**(quantumNum) * \
                                                     np.sqrt(2 * exponent / factorial(2 * quantumNum)) * r**(quantumNum - 1) * \
                                                     np.exp(-exponent * r))
                    else:
                        phi_2s = lambda r: coefficient * ((2 * exponent)**(quantumNum) * \
                                                     np.sqrt(2 * exponent / factorial(2 * quantumNum)) * r**(quantumNum - 1) * \
                                                     np.exp(-exponent * r))
                else:
                    if x == 0:
                        phi_1s = add_func(add, phi_1s,  lambda r: coefficient * (((2 * exponent)**(quantumNum) * \
                                                                        np.sqrt(2 * exponent / factorial(2 * quantumNum)) * r**(quantumNum - 1) * \
                                                                        np.exp(-exponent * r))))
                    else:
                        phi_2s = add_func(add, phi_2s,  lambda r: coefficient * (((2 * exponent)**(quantumNum) * \
                                                                        np.sqrt(2 * exponent / factorial(2 * quantumNum)) * r**(quantumNum - 1) * \
                                                                        np.exp(-exponent * r))))
        g = norm_squared_func(phi_2s)
        g = scalar_mult_func(mul, g, 2)
        f = norm_squared_func(phi_1s)
        f = scalar_mult_func(mul, f, 2)
        f = add_func(add, f, g)


        #f = add_stuff(f, constant_exponents[0], initial_coefficient[0])

        lambdify_f = lambdify(r, f(r))
        print(sp.integrate(lambdify_f, (r, 0, 100)))
        print("4 = ", quad(lambda r: f(r) * r**2 , 0, np.inf)[0])
        scipy_solution = quad(lambda r :(constant_exponents[0] / np.pi)**(3/2) * f(r), 0, np.inf)
        print(scipy_solution, new_coefficient)





if __name__ == "__main__":
    unittest.main()

