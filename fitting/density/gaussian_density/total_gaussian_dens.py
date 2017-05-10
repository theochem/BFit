from fitting.density.density_model import DensityModel
from fitting.density.slater_density.atomic_slater_density import Atomic_Density
import numpy as np


class GaussianTotalBasisSet(DensityModel):
    def __init__(self, element_name, grid, electron_density=None, file_path=None):
        if electron_density is None:
            assert file_path is not None, "File path is needed to compute default Electron Density by slater functions"
            electron_density = Atomic_Density(file_path, grid)
        DensityModel.__init__(self, element_name, grid, electron_density)

    def create_model(self, parameters, exponents=[], coeff=[], optimize_both=True,
                     optimize_coeff=False, optimize_exp=False):
        assert parameters.ndim == 1

        def check_dimension_and_shape():
            assert exponents.ndim == 1
            assert np.shape(exponential)[0] == np.shape(self.grid)[0]
            assert np.shape(exponential)[1] == np.shape(exponents)[0]
            assert exponential.ndim == 2
            assert coefficients.ndim == 1
            assert coefficients.shape[0] == exponential.shape[1]
            assert gaussian_density.ndim == 1
            assert np.shape(gaussian_density)[0] == np.shape(self.grid)[0]

        if optimize_both:
            coefficients = parameters[:len(parameters)//2]
            exponents = parameters[len(parameters)//2:]

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

        exponential = np.exp(-exponents * np.power(np.reshape(self.grid, (len(self.grid), 1)), 2.0))
        gaussian_density = np.dot(exponential, coefficients)
        check_dimension_and_shape()
        return gaussian_density

    def cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[], optimize_both=True,
                      optimize_coeff=False, optimize_exp=False):
        DensityModel.check_type(parameters, "numpy array")
        assert type(num_of_basis_funcs) is int

        if optimize_both:
            residual = super(GaussianTotalBasisSet, self).calculate_residual(parameters, num_of_basis_funcs,
                                                                             [], [], optimize_both, optimize_coeff,
                                                                             optimize_exp)

        elif optimize_coeff:
            coefficients = np.copy(parameters)
            exponents = exponents

            residual = super(GaussianTotalBasisSet, self).calculate_residual(coefficients, num_of_basis_funcs,
                                                                             exponents, [], optimize_both,
                                                                             optimize_coeff, optimize_exp)

        elif optimize_exp:
            exponents = np.copy(parameters)
            coefficients = coeff

            residual = super(GaussianTotalBasisSet, self).calculate_residual(exponents, num_of_basis_funcs, [],
                                                                             coefficients, optimize_both,
                                                                             optimize_coeff, optimize_exp)

        residual_squared = np.power(residual, 2.0)
        return np.sum(residual_squared)

    def derivative_of_cost_function(self, parameters, num_of_basis_funcs, exponents=[], coeff=[],
                                    optimize_both=True, optimize_coeff=False, optimize_exp=False):
        def derivative_wrt_coefficients():
            derivative_coeff = []
            for exp in exponents:
                g_function = -1.0 * np.exp(-1.0 * exp * self.grid**2)
                derivative = f_function * np.ravel(g_function)
                derivative_coeff.append(np.ravel(derivative))
            assert np.shape(derivative_coeff[0])[0] == np.shape(self.grid)[0]
            return derivative_coeff

        def derivative_wrt_exponents():
                derivative_exp = []
                for index, coeff in np.ndenumerate(np.ravel(coefficients)):
                    exponent = exponents[index]
                    g_function = -coeff * self.grid**2 * np.exp(-1.0 * exponent * self.grid**2)
                    derivative = -f_function * np.ravel(g_function)
                    derivative_exp.append(np.ravel(derivative))
                assert np.shape(derivative_exp[0])[0] == np.shape(self.grid)[0]
                return derivative_exp

        if optimize_both:
            coefficients = parameters[:num_of_basis_funcs]
            exponents = parameters[num_of_basis_funcs:]

            residual = super(GaussianTotalBasisSet, self).calculate_residual(parameters, num_of_basis_funcs, [],
                                                                             [], optimize_both, optimize_coeff,
                                                                             optimize_exp)

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
        return np.sum(derivative, axis=1)

    def create_cofactor_matrix(self, exponents):
        exponential = np.exp(-1.0 * exponents * np.power(self.grid, 2.0))
        assert(exponential.shape[1] == len(np.ravel(exponents)))
        assert(exponential.shape[0] == len(np.ravel(self.grid)))
        assert np.ndim(exponential) == 2
        return exponential
