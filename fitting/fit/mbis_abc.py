from __future__ import division

from fitting.fit.model import *
from fitting.fit.GaussianBasisSet import *
import  matplotlib.pyplot as plt
from scipy.integrate import simps, trapz
import numpy as np

from abc import ABCMeta, abstractmethod

class MBIS_ABC():
    __metaclass__ = ABCMeta

    def __init__(self, element_name, atomic_number, grid_obj, electron_density, weights=None):
        assert isinstance(atomic_number, int), "atomic_number should be of type Integer instead it is %r" % type(atomic_number)
        assert np.abs(grid_obj.integrate(electron_density) - atomic_number) < 1e-1, \
                    "Atomic number doesn't match the integration value. Instead it is %r" % grid_obj.integrate(electron_density)
        self.grid_obj = grid_obj
        self.grid_points = np.ma.asarray(np.reshape(grid_obj.radii, (len(grid_obj.radii), 1)))

        if weights is None:
            self.weights = np.ma.asarray(np.ones(len(self.grid_obj.radii)))
        else:
            self.weights = np.ma.asarray(weights) #Weights are masked due the fact that they tend to be small
        self.atomic_number = atomic_number
        self.element_name = element_name
        if element_name == "h" and atomic_number == 1:
            self.electron_density = self.get_hydrogen_electron_density()
        else:
            self.electron_density = electron_density
        # Various methods relay on masked values due to division of small numbers.
        self.masked_electron_density = np.ma.asarray(np.ravel(self.electron_density))
        self.masked_grid_squared = np.ma.asarray(np.ravel(np.power(self.grid_points, 2.)))
        self.lagrange_multiplier = self.get_lagrange_multiplier()
        assert not np.isnan(self.lagrange_multiplier), "Lagrange multiplier is not a number, NAN."
        assert self.lagrange_multiplier != 0., "Lagange multiplier cannot be zero"


    @abstractmethod
    def get_normalized_gaussian_density(self):
        raise NotImplementedError()

    @abstractmethod
    def update_coefficients(self):
        raise NotImplementedError()

    def update_valence_coefficients(self):
        pass

    @abstractmethod
    def update_exponents(self):
        raise NotImplementedError()

    def update_valence_exponents(self):
        pass

    @abstractmethod
    def get_normalization_constant(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()

    def run_for_valence_density(self):
        pass

    def run_greedy(self):
        pass

    def get_lagrange_multiplier(self):
        return self.grid_obj.integrate(self.weights * self.masked_electron_density)\
                / (self.atomic_number)

    def get_all_normalization_constants(self, exp_arr):
        assert exp_arr.ndim == 1
        return np.array([self.get_normalization_constant(x) for x in exp_arr])

    def get_objective_function(self, model):
        log_ratio_of_models = np.log(self.masked_electron_density / np.ma.asarray(model))
        return self.grid_obj.integrate(self.masked_electron_density * self.weights * log_ratio_of_models)

    def integrate_model_with_four_pi(self, model):
        return self.grid_obj.integrate(model)

    def goodness_of_fit_grid_squared(self, model):
        return self.grid_obj.integrate(np.abs(model - np.ravel(self.electron_density))) / (4 * np.pi)

    def goodness_of_fit(self, model):
        return self.grid_obj.integrate(np.abs(model - np.ravel(self.electron_density)) / self.masked_grid_squared) / (4 * np.pi)

    def get_descriptors_of_model(self, model):
        return(self.integrate_model_with_four_pi(model),
               self.goodness_of_fit(model),
               self.goodness_of_fit_grid_squared(model),
               self.get_objective_function(model))


    def create_plots(self, integration_trapz, goodness_of_fit, goodness_of_fit_with_r_sq, objective_function):
        plt.plot(integration_trapz)
        plt.title(self.element_name + " - Integration of Model Using Trapz")
        plt.xlabel("Num of Iterations")
        plt.ylabel("Integration of Model Using Trapz")
        plt.savefig(self.element_name + "_Integration_Trapz.png")
        plt.close()

        plt.semilogy(goodness_of_fit)
        plt.xlabel("Num of Iterations")
        plt.title(self.element_name + " - Goodness of Fit")
        plt.ylabel("Int |Model - True| dr")
        plt.savefig(self.element_name + "_good_of_fit.png")
        plt.close()

        plt.semilogy(goodness_of_fit_with_r_sq)
        plt.xlabel("Num of Iterations")
        plt.title(self.element_name + " - Goodness of Fit with r^2")
        plt.ylabel("Int |Model - True| r^2 dr")
        plt.savefig(self.element_name + "_goodness_of_fit_r_squared.png")
        plt.close()

        plt.semilogy(objective_function)
        plt.xlabel("Num of Iterations")
        plt.title(self.element_name + " -  Objective Function")
        plt.ylabel("KL Divergence Formula")
        plt.savefig(self.element_name + "_objective_function.png")
        plt.close()






if __name__ == "__main__":
    ELEMENT_NAME = "be"
    ATOMIC_NUMBER = 4

    import os
    current_directory = os.path.dirname(os.path.abspath(__file__))[:-3]
    file_path = current_directory + "data\examples\\" + ELEMENT_NAME #+ ".slater"

    from fitting.density.radial_grid import *

    NUMBER_OF_CORE_POINTS = 400; NUMBER_OF_DIFFUSED_PTS = 500
    radial_grid = Radial_Grid(ATOMIC_NUMBER, NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    radial_grid.radii = radial_grid.radii
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    #import horton
    #rtf = horton.ExpRTransform(1.0e-4, 25, 800)
    #radial_grid = horton.RadialGrid(rtf)
    #row_grid_points = radial_grid.radii
    #column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    be =  GaussianTotalBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    #be_val = GaussianValenceBasisSet(ELEMENT_NAME, column_grid_points, file_path)
    #fitting_obj = Fitting(be)
    #fitting_obj_val = Fitting(be_val)
    be.electron_density /= (4 * np.pi)
    #be_val.electron_density_valence /= (4 * np.pi)

    #exps = be.UGBS_s_exponents
    #coeffs = fitting_obj.optimize_using_nnls(be.create_cofactor_matrix(exps))
    #coeffs[coeffs == 0.] = 1e-12

    #exps_valence = be_val.UGBS_p_exponents
    #coeffs_valence = fitting_obj_val.optimize_using_nnls(be.create_cofactor_matrix(exps_valence))
    #coeffs_valence[coeffs_valence == 0.] = 1e-12

    def f():
        pass

    weights = np.ones(len(row_grid_points)) #  1 / (4 * np.pi * np.power(row_grid_points, .1))
    mbis_obj = TotalMBIS(be.electron_density, radial_grid, weights=weights, atomic_number=ATOMIC_NUMBER, element_name=ELEMENT_NAME)

    def generation_of_UGBS_exponents(p, UGBS_exponents):
        max_number_of_UGBS = np.amax(UGBS_exponents)
        min_number_of_UGBS = np.amin(UGBS_exponents)

        def calculate_number_of_Gaussian_functions(p, max, min):
            num_of_basis_functions = np.log(2 * max / min) / np.log(p)
            return num_of_basis_functions

        num_of_basis_functions = calculate_number_of_Gaussian_functions(p, max_number_of_UGBS, min_number_of_UGBS)
        num_of_basis_functions = num_of_basis_functions.astype(int)

        new_Gaussian_exponents = np.array([min_number_of_UGBS])
        for n in range(1, num_of_basis_functions + 1):
            next_exponent = min_number_of_UGBS * np.power(p, n)
            new_Gaussian_exponents = np.append(new_Gaussian_exponents, next_exponent)

        return new_Gaussian_exponents
    #exps = generation_of_UGBS_exponents(4., np.array([1000000, 0.00001]))
    #coeffs =np.array([10.*np.random.random() for x in exps])
    #print(len(coeffs))

    coeffs = np.array([float(ATOMIC_NUMBER)])
    exps = np.array([100.])
    coeffs, exps = mbis_obj.run(1e-4, 1e-2, coeffs, exps, iprint=True)
    #coeffs, exps = mbis_obj.run_greedy(1e-3, 1e-3, 1., iprint=True)
    #coeffs, coeffs2, exps, exps2 = mbis_obj_val.run_valence(1e-2, 1e-2, coeffs, coeffs_valence, exps, exps_valence, iprint=True)

    print("final", coeffs, exps)
    #model = mbis_obj.get_normalized_gaussian_density(coeffs, exps)

    def plot_atomic_density(radial_grid, model, electron_density, title, figure_name):
        #Density List should be in the form
        # [(electron density, legend reference),(model1, legend reference), ..]
        import matplotlib.pyplot as plt
        colors = ["#FF00FF", "#FF0000", "#FFAA00", "#00AA00", "#00AAFF", "#0000FF", "#777777", "#00AA00", "#00AAFF"]
        ls_list = ['-', ':', ':', '-.', '-.', '--', '--', ':', ':']

        radial_grid *= 0.5291772082999999   #convert a.u. to angstrom
        plt.semilogy(radial_grid, model, lw=3, label="approx_model", color=colors[0], ls=ls_list[0])
        plt.semilogy(radial_grid, electron_density, lw=3, label="True Model", color=colors[2], ls=ls_list[3])
        #plt.xlim(0, 25.0*0.5291772082999999)
        plt.xlim(0, 9)
        plt.ylim(ymin=1e-9)
        plt.xlabel('Distance from the nucleus [A]')
        plt.ylabel('Log(density [Bohr**-3])')
        plt.title(title)
        plt.legend()
        plt.savefig(figure_name)
        plt.show()
        plt.close()
    def plot_density_sections(dens, prodens, points, title='None'):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from matplotlib import rcParams
        # choose fonts
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']

        # plotting intervals
        sector = 1.e-5
        conditions = [points <= sector,
                      np.logical_and(points > sector, points <= 100 * sector),
                      np.logical_and(points > 100 * sector, points <= 1000 * sector),
                      np.logical_and(points > 1000 * sector, points <= 1.e5 * sector),
                      np.logical_and(points > 1.e5 * sector, points <= 2.e5 * sector),
                      np.logical_and(points > 2.e5 * sector, points <= 5.0),
                      np.logical_and(points > 5.0, points <= 10.0)]

        # plot within each interval
        for cond in conditions:
            # setup figure
            fig, axes = plt.subplots(2, 1)
            fig.suptitle(title, fontsize=12, fontweight='bold')

            # plot true & model density
            ax1 = axes[0]
            ax1.plot(points[cond], dens[cond], 'ro', linestyle='-', label='True')
            ax1.plot(points[cond], prodens[cond], 'bo', linestyle='--', label='Approx')
            ax1.legend(loc=0, frameon=False)
            xmin, xmax = np.min(points[cond]), np.max(points[cond])
            ax1.set_xticks(ticks=np.linspace(xmin, xmax, 5))
            ymin, ymax = np.min(dens[cond]), np.max(dens[cond])
            ax1.set_yticks(ticks=np.linspace(ymin, ymax, 5))
            ax1.set_ylabel('Density')
            if np.any(points[cond] < 1.0):
                ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            # Hide the right and top spines
            ax1.spines['right'].set_visible(False)
            ax1.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax1.yaxis.set_ticks_position('left')
            ax1.xaxis.set_ticks_position('bottom')
            ax1.grid(True, zorder=0, color='0.60')

            # plot difference of true & model density
            ax2 = axes[1]
            ax2.plot(points[cond], dens[cond] - prodens[cond], 'ko', linestyle='-')
            ax2.set_xticks(ticks=np.linspace(xmin, xmax, 5))
            ax2.set_ylabel('True - Approx')
            ax2.set_xlabel('Distance from the nueleus')
            ymin, ymax = np.min(dens[cond] - prodens[cond]), np.max(dens[cond] - prodens[cond])
            ax2.set_yticks(ticks=np.linspace(ymin, ymax, 5))
            if np.any(points[cond] < 1.0):
                ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            # Hide the right and top spines
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            # Only show ticks on the left and bottom spines
            ax2.yaxis.set_ticks_position('left')
            ax2.xaxis.set_ticks_position('bottom')
            ax2.grid(True, zorder=0, color='0.60')

            plt.tight_layout()
            plt.show()
            plt.close()

    #plot_atomic_density(row_grid_points, model, np.ravel(be.electron_density), "Hey I Just Met You", "This is Crazy")
    #plot_density_sections(np.ravel(be.electron_density), model, row_grid_points, title="hey i just met you")



    ##############################
    ##### ONE GAUSSIAN FUNCTION###
    ##############################
    def try_one_gaussian_func(coeff):
        return ((np.trapz(y=mbis_obj.masked_electron_density * 3 * mbis_obj.masked_grid_squared, x=
        np.ravel(mbis_obj.grid_points)))
                       / (np.trapz(mbis_obj.masked_electron_density * 2. * np.power(mbis_obj.masked_grid_squared, 2.),
                                      x=np.ravel(mbis_obj.grid_points))))
    def try_cusp_right(exp):
        factor = ((3 * ATOMIC_NUMBER) / (2 * exp))
        factor -= np.trapz(y=mbis_obj.masked_electron_density * np.power(mbis_obj.masked_grid_squared, 2.) * 4 * np.pi,
                           x=np.ravel(mbis_obj.grid_points))
        return factor * ((2 * np.pi**(3/2) * mbis_obj.electron_density[0]) / (3 * exp**(1./2.) * ATOMIC_NUMBER))
    #exp = try_one_gaussian_func(float(ATOMIC_NUMBER))
    exps =0.345
    coeff = try_cusp_right(exps)
    print(3 * ATOMIC_NUMBER / (2 * np.trapz(y=mbis_obj.masked_electron_density * np.power(mbis_obj.masked_grid_squared, 2.) * 4 * np.pi,
                           x=np.ravel(mbis_obj.grid_points))))
    print("coeff", coeff)
    model = mbis_obj.get_normalized_gaussian_density(np.array(coeff), np.array([exps]))

    print(model[0], mbis_obj.electron_density[0])
    print(mbis_obj.get_descriptors_of_model(model))
