import numpy as np
from scipy.special import erf
from fitting.fit.GaussianBasisSet import *


############## TODO LIST #######################
#TODO: Ask them if they want |R - R_{alpha}| or |R|
#TODO: Add Tests for N-Dimensions
#TODO: Why did I add moleuclar density to the class instants

class Molecular_Density_Transformation():
    def __init__(self, molecular_density, coefficients_per_atom, exponents_per_atom, num_of_coordinates, displacement_vector):
        assert isinstance(coefficients_per_atom, np.ndarray), "coefficients_per_atom should be a numpy array: %r" % type(coefficients_per_atom)
        assert np.all(coefficients_per_atom, coefficients_per_atom > 0.), "coefficients should all be non-zero positive numbers"
        assert np.all(exponents_per_atom, exponents_per_atom > 0.), "exponents should all be non-zero positive numbers"
        assert isinstance(exponents_per_atom, np.ndarray), "exponents_per_atom should be a numpy array: %r" % type(exponents_per_atom)
        assert type(num_of_coordinates) is int, "The number of coordinates should be of type integer: %r" % type(num_of_coordinates)
        assert isinstance(displacement_vector, np.ndarray), "The displacement vector should be a numpy array: %r" % type(displacement_vector)
        assert coefficients_per_atom.ndim == 2, "Coefficients per atoms should be in a MxN matrix where M = corresponding atom and N = atom's exponents"
        assert exponents_per_atom.ndim == 2, "Exponents per atoms should be in a MxN matrix where M = corresponding atom and N = atom's exponents"
        assert num_of_coordinates in [1, 2, 3], "Number of coordinates should be 1 or 2 or 3: %r" % num_of_coordinates
        assert coefficients_per_atom.shape[0] > 0. and coefficients_per_atom.shape[1] > 0., "Coefficients_per_atom cannot be empty"
        assert exponents_per_atom.shape[0] > 0. and exponents_per_atom.shape[1] > 0., "exponents_per_atom cannot be empty"
        assert coefficients_per_atom.shape[0] == exponents_per_atom.shape[0], \
            "Length of coefficients_per_atom should equal exponents_per_atom" % (coefficients_per_atom.shape[0], exponents_per_atom.shape[0])

        self.number_of_atoms = coefficients_per_atom.shape[0]
        self.dimension = num_of_coordinates * self.number_of_atoms
        self.molecular_density = molecular_density
        self.coefficients_per_atom = coefficients_per_atom
        self.exponents_per_atom = exponents_per_atom
        self.displacement_vector = displacement_vector


    def integrate_a_gaussian_over_the_reals(self, gaussian_coefficient, gaussian_exponent):
        assert type(gaussian_coefficient) is float, "The coefficient for the gaussian should be of type float: %r" % type(gaussian_coefficient)
        assert type(gaussian_exponent) is float, "The exponent for the gaussian should be of type float: %r" % type(gaussian_exponent)
        assert gaussian_coefficient > 0, "The coefficient should be positive non zero number: %r" % gaussian_coefficient
        assert gaussian_exponent > 0, "The exponent should be a positive non zero number: %r" % gaussian_exponent

        return gaussian_coefficient * (np.pi / gaussian_exponent)**(self.dimension/2.0)


    def integrate_promolecular_density(self):
        integration_of_molecular_density = 0.
        for index_atom in range(0, self.number_of_atoms):
            coefficient_of_atom = self.coefficients_per_atom[index_atom]
            exponent_of_atom = self.exponents_per_atom[index_atom]

            for index_gaussian in range(0, len(coefficient_of_atom)):
                integration_of_molecular_density += self.integrate_a_gaussian_over_the_reals(coefficient_of_atom[index_gaussian],
                                                                                              exponent_of_atom[index_gaussian])
        assert integration_of_molecular_density > 0, "The integration of the molecular density should be positive: %r" % integration_of_molecular_density
        return integration_of_molecular_density


    def transform_coordinates_to_hyper_cube(self, initial_coordinates):
        assert len(initial_coordinates) == self.dimension
        new_coordinates = np.empty(self.dimension)

        for i in range(0, self.dimension):
            theta_num = 0
            theta_den = 0

            prefactor_exponent = (self.dimension - i) / 2.0

            for index_atom in range(0, self.number_of_atoms):
                coefficient_of_atom, exponent_of_atom = self.coefficients_per_atom[index_atom], self.exponents_per_atom[index_atom]

                for index_gaussian in range(0, len(coefficient_of_atom)):
                    prefactor = coefficient_of_atom[index_gaussian] * 0.5 * (np.pi / exponent_of_atom[index_gaussian])**prefactor_exponent

                    norm_of_radius = self.get_norm(index_atom, new_coordinates, i)

                    theta_num += self.get_numerator_for_conditional_distribution(prefactor,
                                                                                 exponent_of_atom[index_gaussian],
                                                                                 norm_of_radius, initial_coordinates[i], i)

                    theta_den += self.get_denominator_for_condition_distribution(prefactor, exponent_of_atom[index_gaussian], norm_of_radius)

            # The denominator of the first transformation of coordinates is the integration of the promolecular density over the entire space
            if i == 0:
                theta_den = self.integrate_promolecular_density

            new_coordinates[i] = theta_num / theta_den
            assert 0. < new_coordinates[i] < 1., "the transformed coordinate is not between 0 and 1: %r" % new_coordinates[i]
        return new_coordinates

    def get_norm(self, index_atom, new_coordinates, index):
        norm_of_radius = 0
        for i in range(0, index_atom):
            norm_of_radius += (new_coordinates[index] - self.displacement_vector[index_atom])**2
        return(norm_of_radius)

    def get_numerator_for_conditional_distribution(self, prefactor, exponent, norm_of_radius, coordinate, index):
        if index == 0:
            return prefactor * (erf(np.sqrt(exponent) * (coordinate - self.displacement_vector[index])) + 1)
        return (-1) * prefactor * np.exp(-exponent * norm_of_radius) * (erf(np.sqrt(exponent * (self.displacement_vector[index] - coordinate))) - 1)

    def get_denominator_for_condition_distribution(self, prefactor, exponent, norm_of_radius):
        return prefactor * 2.0 * np.exp(-exponent * norm_of_radius)

    def transform_entire_grid_to_hyper_cube(self, grid):
        assert isinstance(grid, np.ndarray), "Grid should be a Numpy array: %r" % type(grid)
        assert grid.ndim == 2, "Grid should be of dimension two: %r" % grid.ndim
        assert grid.shape[1] is self.dimension, "Grid points should be of dimension %r instead of %r"% (self.dimension, grid.shape[1])

        number_of_points = grid.shape[0]
        new_grid = np.empty(shape=(number_of_points,), dtype=np.object)
        for i in range(0, number_of_points):
            new_grid[i] = self.transform_coordinates_to_hyper_cube(grid[i])
            assert 0. < new_grid[i] < 1., "Transformed grid point should be between 0 and 1: %r" % new_grid[i]
        return new_grid

if __name__ == "__main__":
    ELEMENT = "B"
    ATOMIC_NUMBER = 5
    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT

    ELEMENT_2 = "F"
    ATOMIC_NUMBER_2 = 9
    file_path_2 = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT_2

    #Create Grid for the modeling
    from fitting.density.radial_grid import *
    radial_grid = Radial_Grid(ATOMIC_NUMBER)
    radial_grid_2 = Radial_Grid(ATOMIC_NUMBER_2)
    NUMBER_OF_CORE_POINTS = 200; NUMBER_OF_DIFFUSED_PTS = 300
    row_grid_points = radial_grid.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])
    column_grid_points = np.reshape(row_grid_points, (len(row_grid_points), 1))

    row_grid_points_2 = radial_grid_2.grid_points(NUMBER_OF_CORE_POINTS, NUMBER_OF_DIFFUSED_PTS, [50, 75, 100])

    # Create a total gaussian basis set
    be = GaussianTotalBasisSet(ELEMENT, column_grid_points, file_path)
    fitting_object = Fitting(be)

    flu = GaussianTotalBasisSet(ELEMENT_2, column_grid_points, file_path_2)
    fitting_object_flu = Fitting(flu)



