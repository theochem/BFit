from __future__ import division
from mbis_abc import MBIS_ABC
import numpy as np

class AugmentedMBIS(MBIS_ABC):
    def __init__(self, element_name, atomic_number, grid_obj, electron_density, weights=None):
        super(AugmentedMBIS, self).__init__(element_name, atomic_number, grid_obj, electron_density, weights=weights)

    def get_normalized_coefficients(self, coeff_arr, exp_arr):
        normalized_constants = self.get_all_normalization_constants(exp_arr)
        assert len(normalized_constants) == len(coeff_arr)
        norm_coeff_arr = coeff_arr * normalized_constants
        assert norm_coeff_arr.ndim == 1
        assert len(norm_coeff_arr) == len(coeff_arr) == len(exp_arr)
        assert norm_coeff_arr[0] == coeff_arr[0] * normalized_constants[0], "Instead we get %r and %r * %r" %(norm_coeff_arr[0],
                                                                                                         coeff_arr[0],
                                                                                                         normalized_constants[0])
        return coeff_arr * normalized_constants

    def get_normalized_gaussian_density(self, coeff_arr, exp_arr):
        exponential = np.exp(-exp_arr * np.power(self.grid_points, 2.))
        assert exponential.shape == (len(self.grid_points), len(exp_arr))
        normalized_coeffs = self.get_normalized_coefficients(coeff_arr, exp_arr)
        assert normalized_coeffs.ndim == 1.
        normalized_gaussian_density = np.dot(exponential, normalized_coeffs)
        index_where_zero_occurs = np.where(normalized_gaussian_density == 0.)
        if len(index_where_zero_occurs[0]) != 0:
            normalized_gaussian_density[index_where_zero_occurs] = \
                    normalized_gaussian_density[index_where_zero_occurs[0][0] - 1]
        return normalized_gaussian_density

    def get_normalization_constant(self, exponent):
        return (exponent / np.pi)**(3. / 2.)

    def update_lambda(self, coeff, lamb, mu):
        return lamb - (coeff / mu)

    def update_coefficients(self, coeff_arr, exp_arr, mu, lamb_arr):
        assert len(lamb_arr) == len(coeff_arr)
        assert len(coeff_arr) == len(exp_arr)
        gaus_dens = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        ratio = self.masked_electron_density / gaus_dens

        new_coeff = coeff_arr.copy()
        for i, c in enumerate(coeff_arr):
            exponential = np.ma.asarray(np.exp(-exp_arr[i] * self.masked_grid_squared))

            if c - (mu * lamb_arr[i]) <= 0:
                ratio -= 1
                new_coeff[i] = self.grid_obj.integrate(ratio * self.get_normalization_constant(exp_arr[i]) * exponential)- \
                                lamb_arr[i]
                new_coeff[i] *= mu
            else:
                new_coeff[i] = c * (self.grid_obj.integrate(get_normalization_constant(exp_arr[i]) * exponential ) /
                            self.grid_obj.integrate(ratio * self.get_normalization_constant(exp_arr[i])  * exponential) )

            lamb_arr[i] = self.update_lambda(new_coeff[i], lamb_arr[i], mu)

        return new_coeff, lamb_arr

if __name__ == "__main__":
    ATOMIC_NUMBER = 9
    ELEMENT_NAME = "f"
    USE_HORTON = False
    USE_FILLED_VALUES_TO_ZERO = True
    THRESHOLD_COEFF = 1e-8
    THRESHOLD_EXPS = 40
    import os

    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data/examples//" + ELEMENT_NAME
    if USE_HORTON:
        import horton

        rtf = horton.ExpRTransform(1.0e-30, 25, 1000)
        radial_grid_2 = horton.RadialGrid(rtf)
        from fitting.density.radial_grid import Horton_Grid

        radial_grid = Horton_Grid(1e-80, 25, 1000, filled=USE_FILLED_VALUES_TO_ZERO)
    else:
        NUMB_OF_CORE_POINTS = 400;
        NUMB_OF_DIFFUSE_POINTS = 500
        from fitting.density.radial_grid import Radial_Grid
        from fitting.density.atomic_slater_density import Atomic_Density

        radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMB_OF_CORE_POINTS, NUMB_OF_DIFFUSE_POINTS, [50, 75, 100],
                                  filled=USE_FILLED_VALUES_TO_ZERO)

    from fitting.density import Atomic_Density

    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet

    from fitting.fit.model import Fitting

    atomic_gaussian = GaussianTotalBasisSet(ELEMENT_NAME, np.reshape(radial_grid.radii,
                                                                     (len(radial_grid.radii), 1)), file_path)
    weights = None  # (4. * np.pi * radial_grid.radii**1.)#1. / (1 + (4. * np.pi * radial_grid.radii ** 2.))#1. / (4. * np.pi * radial_grid.radii**0.5) #np.exp(-0.01 * radial_grid.radii**2.)

    fitting_obj = Fitting(atomic_gaussian)

    mbis = AugmentedMBIS(ELEMENT_NAME, ATOMIC_NUMBER, radial_grid, atomic_density.electron_density, weights=weights)

    exps = atomic_gaussian.UGBS_s_exponents[:-3]
    fitting_obj = Fitting(atomic_gaussian)
    coeffs = fitting_obj.optimize_using_nnls(atomic_gaussian.create_cofactor_matrix(exps))
    print(exps)
    coeffs[coeffs == 0.] = 1e-6

    mu_arr = np.array([1000., 100., 10., 1., 0.1, 0.01, 0.001, 0.0001])
    lam_arr = np.array([100. for x in exps])
    for mu in mu_arr:
        for x in range(0, 100):
            coeffs, lam_arr = mbis.update_coefficients(coeffs, exps, mu, lam_arr)
            print(coeffs[coeffs < 0.])
        model = mbis.get_normalized_gaussian_density(coeffs, exps)
        print(mu, mbis.get_descriptors_of_model(model))