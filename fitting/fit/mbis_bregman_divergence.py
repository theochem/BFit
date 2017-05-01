from __future__ import division

import numpy as np

from mbis_abc import MBIS_ABC


class BregmanMBIS(MBIS_ABC):
    def __init__(self, element_name, atomic_number, grid_obj, electron_density, weights=None):
        super(BregmanMBIS, self).__init__(element_name, atomic_number, grid_obj, electron_density, weights=weights)

    def get_normalization_constant(self, exponent):
        return (exponent / np.pi) ** (3./2.)

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

    def run(self):
        pass


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

    def get_objective_function(self, model):
        log_ratio_of_models = np.log(self.masked_electron_density / np.ma.asarray(model))
        KL_Divegence = self.grid_obj.integrate(self.masked_electron_density * self.weights * log_ratio_of_models)
        difference_in_models = self.grid_obj.integrate(self.weights * (self.masked_electron_density - model))
        return KL_Divegence - difference_in_models

    def get_lagrange_multiplier_bregman(self, model):
        true_model_integrated = self.grid_obj.integrate(self.weights * self.masked_electron_density)
        approx_model_integrated = self.grid_obj.integrate(self.weights * model)
        return (true_model_integrated - approx_model_integrated) / self.atomic_number


    def update_coefficients(self, coeff_arr, exp_arr):
        #assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
        masked_normed_gaussian = np.ma.asarray(model)
        assert masked_normed_gaussian.ndim == 1.
        ratio = self.masked_electron_density / masked_normed_gaussian

        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.grid_obj.integrate(self.weights * self.get_normalization_constant(exp_arr[i]) * ratio * np.exp(-exp_arr[i] * self.masked_grid_squared))
            #new_coeff[i] /= lagrange_mult
        model = self.get_normalized_gaussian_density(new_coeff, exp_arr)
        assert np.all(coeff_arr != np.inf)
        assert np.all(exp_arr != np.inf)
        return new_coeff

    def update_exponents(self, coeff_arr, exp_arr):
        #assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr)).copy()
        ratio = self.masked_electron_density / masked_normed_gaussian
        ratio -= 1
        #ratio[ratio <= 0.] = 1e-30
        new_exps = exp_arr.copy()

        for i in range(0, len(exp_arr)):
            exponential = np.ma.asarray(np.exp(-exp_arr[i] * self.masked_grid_squared))
            assert exp_arr[i] > 0.
            new_exps[i] = 3. * self.grid_obj.integrate(exponential * ratio)
            new_exps[i] /= (2. * self.grid_obj.integrate(exponential * self.masked_grid_squared * ratio))
            if new_exps[i] < 0.:
                import matplotlib.pyplot as plt
                plt.plot(self.grid_obj.radii,ratio * 4. * np.pi * self.grid_obj.radii ** 2.)
                plt.plot(self.grid_obj.radii,ratio * 4. * np.pi * self.grid_obj.radii ** 4.)
                b = np.array(ratio * 4. * np.pi * self.grid_obj.radii ** 2.)
                c = np.array(ratio * 4. * np.pi * self.grid_obj.radii ** 4.)
                print(b[b < 0.])
                print(c[c < 0.])
                plt.show()
            #assert np.all(self.weights >= 0.)
            #assert np.all(self.weights * ratio * np.exp(-exp_arr[i] * self.masked_grid_squared) >= 0.)
            if new_exps[i] < 0.:
                print(self.grid_obj.integrate(ratio * np.exp(-exp_arr[i] * self.masked_grid_squared)))
                print((2. * self.grid_obj.integrate(np.exp(-exp_arr[i] * self.masked_grid_squared)*\
                                                    self.masked_grid_squared * ratio)))
        print(new_exps)
        return new_exps

if __name__ == "__main__":
    #################
    ## SET UP#######
    ###########
    ATOMIC_NUMBER = 4
    ELEMENT_NAME = "be"
    USE_HORTON = True
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
        from fitting.radial_grid.radial_grid import Horton_Grid
        radial_grid = Horton_Grid(1e-80, 10, 1000, filled=USE_FILLED_VALUES_TO_ZERO)
    else:
        NUMB_OF_CORE_POINTS = 400; NUMB_OF_DIFFUSE_POINTS = 500
        from fitting.radial_grid.radial_grid import RadialGrid
        from fitting.density.atomic_density.atomic_slater_density import Atomic_Density
        radial_grid = RadialGrid(ATOMIC_NUMBER, NUMB_OF_CORE_POINTS, NUMB_OF_DIFFUSE_POINTS, [50, 75, 100], filled=USE_FILLED_VALUES_TO_ZERO)


    from fitting.density import Atomic_Density
    atomic_density = Atomic_Density(file_path, radial_grid.radii)
    from fitting.fit.GaussianBasisSet import GaussianTotalBasisSet

    from fitting.fit.model import Fitting
    atomic_gaussian = GaussianTotalBasisSet(ELEMENT_NAME, np.reshape(radial_grid.radii,
                                                                    (len(radial_grid.radii), 1)), file_path)
    weights = None

    mbis = BregmanMBIS(ELEMENT_NAME, ATOMIC_NUMBER, radial_grid, atomic_density.electron_density, weights=weights)
    exps = atomic_gaussian.UGBS_s_exponents[:-3]
    fitting_obj = Fitting(atomic_gaussian)
    coeffs = fitting_obj.optimize_using_nnls(atomic_gaussian.create_cofactor_matrix(exps))
    print(exps)
    coeffs[coeffs == 0.] = 1e-6

    for y in range(0, 5):
        for x in range(0, 10000):
            coeffs = mbis.update_coefficients(coeffs, exps)

            model = mbis.get_normalized_gaussian_density(coeffs, exps)
            print(x, np.sum(coeffs), mbis.get_descriptors_of_model(model), model[0], mbis.electron_density[0])
        for x in range(0, 100):
            exps = mbis.update_exponents(coeffs, exps)
            model = mbis.get_normalized_gaussian_density(coeffs, exps)
            print(1, x, mbis.get_descriptors_of_model(model), model[0], mbis.electron_density[0])