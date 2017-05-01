import abc
import numpy as np
from fitting.gbasis.gbasis import UGBSBasis


class DensityModel(object):
    """
    This is an abstract density model used for fitting.
    You have to implement the your model with defined parameters.
    Everything that needs to be implemented is for the
    least squares fitting.

    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, element, grid, electron_density=None):
        self.element = element.lower()
        self.grid = np.copy(grid)
        self.electron_density = electron_density

        gbasis = UGBSBasis(element)
        self.UGBS_s_exponents = 2.0 * gbasis.exponents('s')
        self.UGBS_p_exponents = 2.0 * gbasis.exponents('p')
        if self.UGBS_p_exponents.size == 0.0:
            self.UGBS_p_exponents = np.copy(self.UGBS_s_exponents)

    @abc.abstractmethod
    def create_model(self):
        """
        TODO
        Insert Documentation about the model
        """
        raise NotImplementedError("Please Implement Your Density Model")

    @abc.abstractmethod
    def cost_function(self):
        """
        TODO
        """
        raise NotImplementedError("Please Implement Your Cost Function")

    @abc.abstractmethod
    def derivative_of_cost_function(self):
        """
        TODO
        """
        raise NotImplementedError("Please Implement Your Derivative of Cost Function")

    @abc.abstractmethod
    def create_cofactor_matrix(self):
        pass

    def calculate_residual(self, *args):
        residual = np.ravel(self.electron_density) - self.density_model.create_model(*args)
        return residual

    """
    def calculate_residual_based_on_core(self, *args):
        residual = np.ravel(self.electron_density_core) - self.density_model.create_model(*args)
        return residual

    def calculate_residual_based_on_valence(self, *args):
        residual = np.ravel(self.electron_density_valence) - self.density_model.create_model(*args)
        return residual
    """

    def integrate_model_using_trapz(self, approximate_model):
        integrate = np.trapz(y=np.ravel(self.grid**2.) * np.ravel(approximate_model), x=np.ravel(self.grid))
        return integrate

    def measure_error_by_integration_of_difference(self, true_model, approximate_model):
        error = np.trapz(y=np.ravel(self.grid**2) * (np.absolute(np.ravel(true_model) - np.ravel(approximate_model))),
                         x=np.ravel(self.grid))
        return error

    def measure_error_by_difference_of_integration(self, true_model, approximate_model):
        integration_of_true_model = self.integrate_model_using_trapz(true_model)
        integration_of_approx_model = self.integrate_model_using_trapz(approximate_model)

        difference_of_models = integration_of_true_model - integration_of_approx_model

        return np.absolute(difference_of_models)

    def generation_of_UGBS_exponents(self, p, UGBS_exponents):
        max_number_of_UGBS = np.amax(UGBS_exponents)
        min_number_of_UGBS = np.amin(UGBS_exponents)
        print("max_number_of_UGBS", max_number_of_UGBS)
        print("min_number_of_UGBS", min_number_of_UGBS)

        def calculate_number_of_gaussian_functions(p, max, min):
            num_of_basis_functions = np.log(2 * max / min) / np.log(p)
            return num_of_basis_functions

        num_of_basis_functions = calculate_number_of_gaussian_functions(p, max_number_of_UGBS, min_number_of_UGBS)
        num_of_basis_functions = num_of_basis_functions.astype(int)

        new_gaussian_exponents = np.array([min_number_of_UGBS])
        for n in range(1, num_of_basis_functions + 1):
            next_exponent = min_number_of_UGBS * np.power(p, n)
            new_gaussian_exponents = np.append(new_gaussian_exponents, next_exponent)

        return new_gaussian_exponents

    @staticmethod
    def plot_atomic_density(radial_grid, density_list, title, figure_name):
        #Density List should be in the form
        # [(electron density, legend reference),(model1, legend reference), ..]
        import matplotlib.pyplot as plt
        colors = ["#FF00FF", "#FF0000", "#FFAA00", "#00AA00", "#00AAFF", "#0000FF", "#777777", "#00AA00", "#00AAFF"]
        ls_list = ['-', ':', ':', '-.', '-.', '--', '--', ':', ':']
        assert isinstance(density_list, list)
        # convert a.u. to angstrom
        radial_grid *= 0.5291772082999999
        for i, item in enumerate(density_list):
            dens, label = item
            # plot with log scaling on the y axis
            plt.semilogy(radial_grid, dens, lw=3, label=label, color=colors[i], ls=ls_list[i])

        #plt.xlim(0, 25.0*0.5291772082999999)
        plt.xlim(0, 9)
        plt.ylim(ymin=1e-9)
        plt.xlabel('Distance from the nucleus [A]')
        plt.ylabel('Log(density [Bohr**-3])')
        plt.title(title)
        plt.legend()
        plt.savefig(figure_name)
        plt.close()