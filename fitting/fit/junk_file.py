import numpy as np
from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
from fitting.fit.mbis_abc import MBIS
def get_exponents(electron_density, grid, atomic_number, coeff):
    return -(np.trapz(y=np.ravel(electron_density) * np.log(np.ravel(electron_density) / coeff), x=np.ravel(grid))) / atomic_number
class MBIS():
    def __init__(self, model_object, weights, atomic_number, horton_grid=None):
        assert isinstance(model_object, DensityModel)
        assert type(atomic_number) is int, "Atomic Number should be of type int. Instead it is %r" % type(atomic_number)
        self.model_object = model_object
        self.weights = np.ma.asarray(weights)
        self.atomic_number = atomic_number #self.model_object.integrated_total_electron_density
        self.lamda_multplier = self.lagrange_multiplier()
        assert not np.isnan(self.lamda_multplier), "lagrange multiplier should not nan"
        assert self.lamda_multplier != 0., "lagrange multiplier cannot be zero"

    @staticmethod
    def get_normalization_constant(exponent):
        return (exponent / np.pi)**(3./2.)

    def get_all_normalization_constants(self, exp_arr):
        assert exp_arr.ndim == 1
        return np.array([MBIS.get_normalization_constant(x) for x in exp_arr])

    def get_normalized_coefficients(self, coeff_arr, exp_arr):
        normalized_constants = self.get_all_normalization_constants(exp_arr)
        assert len(normalized_constants) == len(coeff_arr)
        return coeff_arr * normalized_constants

    def get_normalized_gaussian_density(self, coeff_arr, exp_arr):
        exponential = np.exp(-exp_arr * np.power(self.model_object.grid, 2.))
        normalized_coeffs = self.get_normalized_coefficients(coeff_arr, exp_arr)
        assert normalized_coeffs.ndim == 1.
        return np.dot(exponential, normalized_coeffs)

    def lagrange_multiplier(self):
        grid_squared = np.ravel(np.power(self.model_object.grid, 2.))
        return 4 * np.pi * np.trapz(y=self.weights * np.ravel(self.model_object.electron_density) * grid_squared \
                                    , x=np.ravel(self.model_object.grid)) / self.atomic_number

    def get_integration_factor(self, exponent, masked_normed_gaussian):
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        ratio = masked_electron_density / masked_normed_gaussian
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))

        prefactor = 4 * np.pi * MBIS.get_normalization_constant(exponent)
        integrand = self.weights * ratio * np.exp(-exponent * masked_grid_squared)
        return prefactor * np.trapz(y=integrand * masked_grid_squared, x=np.ravel(self.model_object.grid))

    def update_coefficients(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))

        new_coeff = coeff_arr.copy()
        for i in range(0, len(coeff_arr)):
            new_coeff[i] *= self.get_integration_factor(exp_arr[i], masked_normed_gaussian)
            new_coeff[i] /= self.lamda_multplier
        return new_coeff

    def update_exponents(self, coeff_arr, exp_arr):
        assert np.all(coeff_arr > 0), "Coefficients should be positive. Instead we got %r" % coeff_arr
        assert np.all(exp_arr > 0), "Exponents should be positive. Instead we got %r" % exp_arr
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        masked_grid_quadrupled = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 4.)))
        ratio = masked_electron_density / masked_normed_gaussian

        new_exps = exp_arr.copy()
        for i in range(0, len(exp_arr)):
            prefactor = 4 * np.pi * MBIS.get_normalization_constant(exp_arr[i])
            integrand = self.weights * ratio * np.exp(-exp_arr[i] * masked_grid_squared)
            new_exps[i] = 3 * self.lamda_multplier
            integration = np.trapz(y=integrand * masked_grid_quadrupled, x=np.ravel(self.model_object.grid))
            assert prefactor != 0, "prefactor should not be zero but the exponent for normalization constant is %r" % str(exp_arr[i])
            assert integration != 0, "Integration of the integrand is zero. The Integrand Times r^4 is %r" % str(integrand * masked_grid_quadrupled)
            assert not np.isnan(integration), "Integration should not be nan"
            new_exps[i] /= ( 2 * prefactor * integration)
        return new_exps

    def run(self, threshold_coeff, threshold_exp, coeff_arr, exp_arr, iprint=False, iplot=False):
        old_coeffs = coeff_arr.copy() + threshold_coeff * 2.
        new_coeffs = coeff_arr.copy()
        old_exps = exp_arr.copy() + threshold_exp * 2.
        new_exps = exp_arr.copy()
        storage_of_errors = [["""Integration Using Trapz"""],
                             ["""Sum of Coefficients"""],
                             [""" goodness of fit"""],
                             [""" goof of fit with r^2"""],
                             [""" KL Divergence Formula"""]]

        counter = 0
        while np.any(np.abs(new_exps - old_exps) > threshold_exp ):
            temp_storage_coeffs = new_coeffs.copy()
            new_coeffs = self.update_coefficients(new_coeffs, new_exps)
            old_coeffs = temp_storage_coeffs.copy()

            while np.any(np.abs(old_coeffs - new_coeffs) > threshold_coeff):
                temp_storage_coeffs = new_coeffs.copy()
                new_coeffs = self.update_coefficients(new_coeffs, new_exps)
                # print(new_coeffs)
                old_coeffs = temp_storage_coeffs.copy()
                model = self.get_normalized_gaussian_density(new_coeffs , new_exps)

                integration_model_four_pi = self.integrate_model_with_four_pi(model)
                sum_of_coeffs = np.sum(new_coeffs)
                goodness_of_fit = self.goodness_of_fit(model)
                goodness_of_fit_r_squared = self.goodness_of_fit_grid_squared(model)
                objective_function = self.get_objective_function(new_coeffs, new_exps)
                if iprint:
                    print(counter, integration_model_four_pi, sum_of_coeffs, \
                          goodness_of_fit, goodness_of_fit_r_squared, \
                          objective_function,  True, np.max(np.abs(old_coeffs - new_coeffs)))
                if iplot:
                    storage_of_errors[0].append(integration_model_four_pi)
                    storage_of_errors[1].append(sum_of_coeffs)
                    storage_of_errors[2].append(goodness_of_fit)
                    storage_of_errors[3].append(goodness_of_fit_r_squared)
                    storage_of_errors[4].append(objective_function)
                counter += 1

            temp_storage_exps = new_exps.copy()
            new_exps = self.update_exponents(new_coeffs, new_exps)
            #print(new_exps)
            old_exps = temp_storage_exps.copy()
            model = self.get_normalized_gaussian_density(new_coeffs, new_exps)

            integration_model_four_pi = self.integrate_model_with_four_pi(model)
            sum_of_coeffs = np.sum(new_coeffs)
            goodness_of_fit = self.goodness_of_fit(model)
            goodness_of_fit_r_squared = self.goodness_of_fit_grid_squared(model)
            objective_function = self.get_objective_function(new_coeffs, new_exps)
            if iprint:
                if counter % 100 == 0.:
                    for x in range(0, len(new_coeffs)):
                        print(new_coeffs[x], new_exps[x])
                print(counter, integration_model_four_pi, sum_of_coeffs, \
                      goodness_of_fit, goodness_of_fit_r_squared, \
                      objective_function, False, np.max(np.abs(new_exps - old_exps)))
            if iplot:
                storage_of_errors[0].append(integration_model_four_pi)
                storage_of_errors[1].append(sum_of_coeffs)
                storage_of_errors[2].append(goodness_of_fit)
                storage_of_errors[3].append(goodness_of_fit_r_squared)
                storage_of_errors[4].append(objective_function)
            counter += 1

        #########
        # PLotting
        #####
        if iplot:
            storage_of_errors = np.array(storage_of_errors)
            plt.plot(storage_of_errors[0])
            plt.title(self.model_object.element + " - Integration of Model Using Trapz")
            plt.xlabel("Num of Iterations")
            plt.ylabel("Integration of Model Using Trapz")
            plt.savefig(self.model_object.element + "_Integration_Trapz.png")
            plt.close()

            plt.plot(storage_of_errors[1])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " - Sum of Coefficients")
            plt.ylabel("Sum of Coefficients")
            plt.savefig(self.model_object.element + "_Sum_of_coefficients.png")
            plt.close()

            plt.semilogy(storage_of_errors[2])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " - Goodness of Fit")
            plt.ylabel("Int |Model - True| dr")
            plt.savefig(self.model_object.element + "_good_of_fit.png")
            plt.close()

            plt.semilogy(storage_of_errors[3])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " - Goodness of Fit with r^2")
            plt.ylabel("Int |Model - True| r^2 dr")
            plt.savefig(self.model_object.element + "_goodness_of_fit_r_squared.png")
            plt.close()

            plt.semilogy(storage_of_errors[4])
            plt.xlabel("Num of Iterations")
            plt.title(self.model_object.element + " -  Objective Function")
            plt.ylabel("KL Divergence Formula")
            plt.savefig(self.model_object.element + "_objective_function.png")
            plt.close()
        return new_coeffs, new_exps

    def run_greedy(self, threshold, iprint=False):
        def next_list_of_exponents(exponents_array, coefficient_array, factor, coeff_guess):
            assert exponents_array.ndim == 1
            size = exponents_array.shape[0]
            all_choices_of_exponents = []
            all_choices_of_coeffs = []

            for index, exp in np.ndenumerate(exponents_array):
                if index[0] == 0:
                    exponent_array = np.insert(exponents_array, index, exp / factor )
                    coeff_array = np.insert(coefficient_array, index, coeff_guess)
                elif index[0] <= size:
                    exponent_array = np.insert(exponents_array, index, (exponents_array[index[0] - 1] + exponents_array[index[0]])/2)
                    coeff_array = np.insert(coeff_arr, index, coeff_guess)

                all_choices_of_exponents.append(exponent_array)
                all_choices_of_coeffs.append(coeff_array)

                if index[0] == size - 1:
                    exponent_array = np.append(exponents_array, np.array([ exp * factor] ))
                    coeff_array = np.append(coeff_arr, np.array([coeff_guess]))
                    all_choices_of_exponents.append(exponent_array)
                    all_choices_of_coeffs.append(coeff_array)
            return(all_choices_of_exponents, all_choices_of_coeffs)

        coeff_arr = np.array([4.])
        exp_arr = np.array([333.])
        coeff_arr, exp_arr = self.run(1e-1, 20, coeff_arr, exp_arr)
        model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)

        for x in range(0, 50):
            next_list_of_exps, next_list_of_coeffs = next_list_of_exponents(exp_arr, coeff_arr, 100., coeff_guess=1.)

            best_next_exps = None
            best_next_coeffs = None
            best_objective_func = 1e10
            for i in range(0, len(next_list_of_exps)):
                objective_func = self.get_objective_function(next_list_of_coeffs[i], next_list_of_exps[i])

                #objective_func, coeffs, exps = self.optimize_using_slsqp(np.append(next_list_of_coeffs[i], next_list_of_exps[i]))
                if objective_func < best_objective_func:
                    best_next_exps = next_list_of_exps[i]
                    best_next_coeffs = next_list_of_coeffs[i]

            best_next_coeffs[best_next_coeffs == 0.] = 1e-12
            coeff_arr, exp_arr = self.run(1e-1, 1e-1, best_next_coeffs, best_next_exps,iprint=False)
            model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
            if iprint:
                print("\n", x, self.integrate_model(model), self.integrate_model_with_four_pi(model), np.sum(coeff_arr),\
                      self.goodness_of_fit(model), self.goodness_of_fit_grid_squared(model), \
                      self.get_objective_function(coeff_arr, exp_arr))


        return coeff_arr, exp_arr


    def fixed_iterations(self, coeff_arr, exp_arr):
        for y in range(0, 5):
            for x in range(0, 1000):
                coeff_arr = self.update_coefficients(coeff_arr, exp_arr)
                #print(coeff_arr)
                model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
                print(x, self.integrate_model(model), self.integrate_model_with_four_pi(model), np.sum(coeff_arr),\
                      self.goodness_of_fit(model), self.goodness_of_fit_grid_squared(model), \
                      self.get_objective_function(coeff_arr, exp_arr))
            for x in range(0, 1000):
                exp_arr = self.update_exponents(coeff_arr, exp_arr)
                print(exp_arr)
                model = self.get_normalized_gaussian_density(coeff_arr, exp_arr)
                print(x, self.integrate_model(model), self.integrate_model_with_four_pi(model), np.sum(coeff_arr),\
                      self.goodness_of_fit(model), self.goodness_of_fit_grid_squared(model), \
                      self.get_objective_function(coeff_arr, exp_arr))

    def get_objective_function(self, coeff_arr, exp_arr):
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_electron_density * self.weights * 4 * np.pi * masked_grid_squared * np.log(masked_electron_density / masked_normed_gaussian),\
                        x=np.ravel(self.model_object.grid))

    def integrate_model(self, model):
        return np.trapz(y=model * np.ravel(np.power(self.model_object.grid, 2.)), x=np.ravel(self.model_object.grid))

    def integrate_model_with_four_pi(self, model):
        return np.trapz(y= 4 * np.pi * model * np.ravel(np.power(self.model_object.grid, 2.)), x=np.ravel(self.model_object.grid))

    def goodness_of_fit_grid_squared(self, model):
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_grid_squared *np.abs(model - np.ravel(self.model_object.electron_density)), x=np.ravel(self.model_object.grid))

    def goodness_of_fit(self, model):
        return np.trapz(y=np.abs(model - np.ravel(self.model_object.electron_density)), x=np.ravel(self.model_object.grid))

    def maximum_difference(self):
        return np.max()

    def get_KL_divergence(self, parameters):
        coeff_arr = parameters[:len(parameters)//2]
        exp_arr = parameters[len(parameters)//2:]
        masked_normed_gaussian = np.ma.asarray(self.get_normalized_gaussian_density(coeff_arr, exp_arr))
        masked_electron_density = np.ma.asarray(np.ravel(self.model_object.electron_density))
        masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.model_object.grid, 2.)))
        return np.trapz(y=masked_electron_density * self.weights * 4 * np.pi * masked_grid_squared * np.log(masked_electron_density / masked_normed_gaussian),\
                        x=np.ravel(self.model_object.grid))

    def optimize_using_slsqp(self, parameters):
        def constraint(x):
            num_of_funcs = len(x)//2
            return self.atomic_number - np.sum(x[:num_of_funcs])
        def constraint2(x):
            num_of_funcs = len(x)//2
            model = self.get_normalized_gaussian_density(x[:num_of_funcs], x[num_of_funcs:])
            return np.sum(np.abs(np.ravel(self.model_object.electron_density) - model))

        #cons=({'type':'eq', 'fun':constraint},
        #      {'type':'eq', 'fun':constraint2})
        cons = ({'type':'eq', 'fun':constraint})
        bounds = np.array([(0.0, np.inf) for x in range(0, len(parameters))], dtype=np.float64)
        f_min_slsqp = scipy.optimize.minimize(self.get_KL_divergence, x0=parameters, method="SLSQP",
                                              bounds=bounds, constraints=cons, jac=False)
        parameters = f_min_slsqp['x']
        coeffs = parameters[:len(parameters)//2]
        exps = parameters[len(parameters)//2:]
        objective_func = f_min_slsqp['fun']
        return objective_func, coeffs, exps

if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_NAME + ".slater"
    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    NUMBER_OF_CORE_POINTS = 300; NUMBER_OF_DIFFUSED_PTS = 400
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])[1:]
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    fitting_obj = Fitting(be)
    coeffs = np.array([  7.67163334e-15,   3.81490190e-03,   4.58541643e-18,   2.36675454e-01,
                       1.26174567e-01 ,  1.20608719e-01,   1.88961187e-09 ,  6.62059483e-02,
                       6.34315834e-01 ,  9.08376837e-01,   7.14355583e-01 ,  5.41385482e-01,
                       3.16551329e-01 ,  1.56727505e-01,   8.47536805e-02 ,  4.57210564e-02,
                       2.26040275e-02 ,  1.04208945e-02,   5.66069946e-03 ,  2.84130951e-03,
                       1.43757892e-03 ,  6.64960536e-04,   3.53818357e-04 ,  1.72414813e-04,
                       9.15607124e-05 ,  4.18651473e-05,   2.22983242e-05 ,  1.05309750e-05,
                       5.80365083e-06 ,  2.60922355e-06,   1.29509724e-06 ,  7.79661885e-07,
                       3.43721155e-07 ,  8.39357998e-08,   1.82224296e-07 ,  1.42750438e-17,
                       3.85412742e-18 ,  4.34530002e-08] )
    exps = np.array([  7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03 ,7.82852278e+03,
                      7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 , 7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.86683468e+02,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03,   7.82852278e+03  , 7.82852278e+03,
                       7.82852278e+03 ,  7.82852278e+03])


    import sympy as sp
    from sympy.abc import x, y
    print(sp.integrate())