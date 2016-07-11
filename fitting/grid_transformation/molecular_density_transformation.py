import numpy as np
from scipy.special import erf
from fitting.fit.GaussianBasisSet import *


############## TODO LIST #######################
#TODO: Ask them if they want |R - R_{alpha}| or |R|
#TODO: Add Tests for N-Dimensions
#TODO: Why did I add moleuclar density to the class instants
#TODO: Line 17-18, 39-40 should all coefficients and exponents be > or >= 0.
#TODO: What to do with the transformed grid point is either 0. or 1.?? For now it is assumed to can 0. or 1. Line 86
#TODO: get num and get den, I put the exponential to the smallest positive number if it is zero

class Molecular_Density_Transformation():
    """
    Given a specified pro-molecular density
    composed of gaussian-type function. Transform
    the integration of the density over the reals
    to integration over the unit hyper-cube.

    Attributes
    ----------
    coefficients_per_atom : array_like
                            MxN array where M equals number of
                            atoms and N equals the number of
                            coefficients for that atom.

    exponents_per_atom : array_like
                            MxN array where M equals number of
                            atoms and N equals the number of
                            exponents for that atom.

    num_of_coordinates : int
                        Indicates what dimension is the
                        pro-molecular density w.r.t to the
                        radius. Can only be 1, 2 or 3 dimensions.

    displacement_vector : array_like
                          MxN array where M equals number of
                          atoms i.e. each row corresponds
                          to the atom's nuclei position.
                          The dimension of nuclei position is
                          specified by the num_of_coordinates.

    Methods
    -------
    transform_coordinates_to_hyper_cube : float
                                          Converts a real vector to its
                                          corresponding value inside a hypercube

    transform_entire_grid_to_hyper_cube : array_like
                                          Converts all vectors inside the array
                                          to it's corresponding value inside
                                          a hypercube

    Notes
    -----
    ..  [1] Juan I Rodriguez, David C. Thompson, James S. M. Anderson,
        Jordan W Thomson and Paul W Ayers, "A physically motivated
        sparse cubature scheme with applications to molecular
        density-functional theory"
    """
    def __init__(self, coefficients_per_atom, exponents_per_atom, dimension_of_coordinates, displacement_vector):
        assert isinstance(coefficients_per_atom, np.ndarray), "coefficients_per_atom should be a numpy array: %r" % type(coefficients_per_atom)
        assert np.all(coefficients_per_atom >= 0.), "coefficients should all be non-zero positive numbers"
        assert np.all(exponents_per_atom >= 0.), "exponents should all be non-zero positive numbers"
        assert isinstance(exponents_per_atom, np.ndarray), "exponents_per_atom should be a numpy array: %r" % type(exponents_per_atom)
        assert type(dimension_of_coordinates) is int, "The dimension of the coordinates should be of type integer: %r" % type(dimension_of_coordinates)
        assert isinstance(displacement_vector, np.ndarray), "The displacement vector should be a numpy array: %r" % type(displacement_vector)
        assert coefficients_per_atom.ndim == 2, \
            "Coefficients per atoms should be in a MxN matrix where M = corresponding atom and N = # of atom's exponents"
        assert exponents_per_atom.ndim == 2, \
            "Exponents per atoms should be in a MxN matrix where M = corresponding atom and N = # of atom's exponents"
        assert dimension_of_coordinates in [1, 2, 3], "Number of coordinates should be 1 or 2 or 3: %r" % dimension_of_coordinates
        assert coefficients_per_atom.shape[0] > 0. and coefficients_per_atom.shape[1] > 0., "Coefficients_per_atom cannot be empty"
        assert exponents_per_atom.shape[0] > 0. and exponents_per_atom.shape[1] > 0., "Exponents_per_atom cannot be empty"
        assert coefficients_per_atom.shape[0] == exponents_per_atom.shape[0], \
            "Length of coefficients_per_atom should equal exponents_per_atom" % (coefficients_per_atom.shape[0], exponents_per_atom.shape[0])

        self.number_of_atoms = coefficients_per_atom.shape[0]
        #self.dimension = dimension_of_coordinates * self.number_of_atoms
        self.dimension = dimension_of_coordinates
        self.coefficients_per_atom = coefficients_per_atom
        self.exponents_per_atom = exponents_per_atom
        self.displacement_vector = displacement_vector


    def integrate_a_gaussian_over_the_reals(self, gaussian_coefficient, gaussian_exponent):
        """ Integrate a gaussian-type function from negative infinity to infinity.

        Parameters
        ----------
        gaussian_coefficient : float
                               coefficient for the gaussian-type function

        gaussian_exponent : float
                            exponent for the gaussian-type function

        Returns
        -------
        float
            analytical solution for the integration of gaussian-type function
        """
        assert type(gaussian_coefficient) in (float, np.float64) , "The coefficient for the gaussian should be of type float: %r" % type(gaussian_coefficient)
        assert type(gaussian_exponent) in (float, np.float64), "The exponent for the gaussian should be of type float: %r" % type(gaussian_exponent)
        assert gaussian_coefficient >= 0, "The coefficient should be positive non zero number: %r" % gaussian_coefficient
        assert gaussian_exponent >= 0, "The exponent should be a positive non zero number: %r" % gaussian_exponent

        return gaussian_coefficient * (np.pi / gaussian_exponent)**(self.dimension/2.0)


    def integrate_promolecular_density(self):
        """Integration of promolecular density function over the reals
        (i.e. sum of electronic density functions for each atom that is
        composed of a sum of gaussian-type functions).


        Returns
        -------
        float
            analytical solution using the linearity of the integration operator

        See Also
        --------
        integrate_a_gaussian_over_the_reals : Single integration of a
                                                gaussian-type function
        """
        integration_of_promolecular_density = 0.
        for index_atom in range(0, self.number_of_atoms):
            coefficient_of_atom = self.coefficients_per_atom[index_atom]
            exponent_of_atom = self.exponents_per_atom[index_atom]

            for index_gaussian in range(0, len(coefficient_of_atom)):
                integration_of_promolecular_density += self.integrate_a_gaussian_over_the_reals(coefficient_of_atom[index_gaussian],
                                                                                              exponent_of_atom[index_gaussian])
        assert integration_of_promolecular_density > 0, "The integration of the molecular density should be positive: %r" % integration_of_promolecular_density
        return integration_of_promolecular_density


    def transform_coordinates_to_hyper_cube(self, initial_coordinates):
        """ A single vector from the D-dimensional real space is
            converted to a unit hyper-cube by the one-to-one
            conditional distribution function.

        Parameters
        ---------
        initial_coordinates : float array_like
                              D-dimensional vector

        Returns
        -------
        new_coordinates : array_like
                          The transformed vector where each
                          element should be between 0 and 1

        See Also
        --------
        get_norm : float
                   Computes the norm squared between the point
                   and the nuclei position

        get_numerator_for_conditional_distribution : float
                                                    Computes the numerator for
                                                    the conditional distribution method

        get_denominator_for_condition_distribution : float
                                                     Computes the denominator for
                                                     the conditional distribution method

        integrate_promolecular_density : float
                                         Integrates over the reals iteratively for
                                         the pro-molecular density function.

        Notes
        ---------
        ..  [1] Juan I Rodriguez, David C. Thompson, James S. M. Anderson,
            Jordan W Thomson and Paul W Ayers, "A physically motivated
            sparse cubature scheme with applications to molecular
            density-functional theory"

        """
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

                    norm_of_radius = self.get_norm(new_coordinates, i)

                    theta_num += self.get_numerator_for_conditional_distribution(prefactor, exponent_of_atom[index_gaussian],
                                                                                 norm_of_radius, initial_coordinates[i], i)

                    theta_den += self.get_denominator_for_condition_distribution(prefactor, exponent_of_atom[index_gaussian], norm_of_radius)

            # The denominator of the first transformation of coordinates is the integration of the promolecular density over the entire space
            if i == 0:
                theta_den = self.integrate_promolecular_density()

            new_coordinates[i] = theta_num / theta_den

            #assert 0. <= new_coordinates[i] <= 1., "the transformed coordinate is not between 0 and 1: %r" % new_coordinates[i]
        return new_coordinates

    def get_norm(self, new_coordinates, index):
        norm_of_radius = 0
        for i in range(0, index):
            norm_of_radius += (new_coordinates[i] - self.displacement_vector[i])**2
        return(norm_of_radius)

    def get_numerator_for_conditional_distribution(self, prefactor, exponent, norm_of_radius, coordinate, index):
        """A helper function for integrating the numerator
            of the conditional distribution method.

        Parameters
        ----------
        prefactor : float


        exponent : float


        norm_of_radius : float


        coordinate : array_like


        index : int


        Returns
        -------


        See Also
        --------
        transform_coordinates_to_hyper_cube : used in this function to convert coordinate to
                                                be in a hyper-cube

        Notes
        -----

        """
        if index == 0:
            return prefactor * (erf(np.sqrt(exponent) * (coordinate - self.displacement_vector[index])) + 1)
        exponential = np.exp(-exponent * norm_of_radius)
        #if exponential == 0.0:
        #    exponential = np.finfo(np.double).tiny #Smallest Positive number
        return (-1) * prefactor * exponential * (erf(np.sqrt(exponent) * (self.displacement_vector[index] - coordinate)) - 1)

    def get_denominator_for_condition_distribution(self, prefactor, exponent, norm_of_radius):
        """A helper function for integrating the denominator
            of the conditional distribution method.

        Parameters
        ----------
        prefactor : float
                    Combination of constants that don't
                    depend on the integration

        exponent : float


        norm_of_radius : float
                        norm between the coordinate and
                        distance of the nuclei from the
                        centre

        Returns
        ------


        See Also
        --------
        transform_coordinates_to_hyper_cube : used in this function to convert coordinate to
                                                be in a hyper-cube

        Examples
        --------

        """
        exponential = np.exp(-exponent * norm_of_radius)
        #if exponential == 0.0:
        #    exponential = np.finfo(np.double).tiny #Smallest Positive number
        return prefactor * 2.0 * exponential

    def transform_entire_grid_to_hyper_cube(self, grid):
        """Given a set of points on a D-dimensional real space.
            Each point is transformed to a D-dimensional hypercube.


        Parameters
        ----------
        grid : array_like
               MxD array where M equals the number of points
                and D equals the dimension of each vector.

        Returns
        -------
        new_grid : array_like
                   MxD array where each point is on the
                   D-dimensional hyper-cube.

        See Also
        --------
        transform_coordinates_to_hyper_cube :

        """
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



