import numpy as np
from scipy.special import erf
from fitting.fit.GaussianBasisSet import *


############## TODO LIST #######################
#TODO: Ask them if they want |R - R_{alpha}| or |R|
#TODO: Add Tests for N-Dimensions

class Molecular_Density_Transformation():
    def __init__(self, molecular_density, coefficients_per_atom, exponents_per_atom, num_of_coordinates, displacement_vector):
        assert coefficients_per_atom.ndim == 2
        assert  exponents_per_atom.ndim == 2
        assert (coefficients_per_atom).shape[0] == exponents_per_atom.shape[0]
        assert  num_of_coordinates == 1 or num_of_coordinates == 2 or num_of_coordinates == 3

        self.number_of_atoms = (coefficients_per_atom).shape[0]
        self.dimension = num_of_coordinates * self.number_of_atoms
        self.molecular_density = molecular_density
        self.coefficients_per_atom = coefficients_per_atom
        self.exponents_per_atom = exponents_per_atom
        self.displacement_vector = displacement_vector


    def integrate_of_one_gaussian_over_reals(self, gaussian_coefficient, gaussian_exponent):
        return gaussian_coefficient * (np.pi / gaussian_exponent)**(self.dimension/2.0)


    def integration_of_molecular_density_over_space(self):
        integration_of_molecular_density = 0
        for index_atom in range(0, self.number_of_atoms):
            coefficient_of_atom = self.coefficients_per_atom[index_atom]
            exponent_of_atom = self.exponents_per_atom[index_atom]

            for index_gaussian in range(0, len(coefficient_of_atom)):
                integration_of_molecular_density += self.integrate_of_one_gaussian_over_reals(coefficient_of_atom[index_gaussian],
                                                                                              exponent_of_atom[index_gaussian])
        return integration_of_molecular_density


    def new_weights(self, points_on_real_lines):
        assert len(points_on_real_lines) == self.dimension

        theta_list = np.empty(self.dimension)

        for index_theta in range(0, self.dimension):
            theta_num = 0
            theta_den = 0

            prefactor_exponent = (self.dimension - index_theta) / 2.0

            for index_atom in range(0, self.number_of_atoms):
                coefficient_of_atom = self.coefficients_per_atom[index_atom]
                exponent_of_atom = self.exponents_per_atom[index_atom]


                for index_gaussian in range(0, len(coefficient_of_atom)):
                    prefactor = coefficient_of_atom[index_gaussian] * 0.5 * (np.pi / exponent_of_atom[index_gaussian])**prefactor_exponent

                    if index_theta == 0:
                        theta_num += prefactor * (erf(np.sqrt(exponent_of_atom[index_gaussian]) * (points_on_real_lines[index_theta] - self.displacement_vector[index_theta])) + 1)
                    else:
                        norm_of_radius = 0
                        for index_prev_calc_theta in range(0, index_atom):
                            norm_of_radius += (theta_list[index_theta] - self.displacement_vector[index_theta])**2

                        theta_num -= prefactor * np.exp(-exponent_of_atom[index_gaussian] * norm_of_radius) * \
                                 (erf(np.sqrt(exponent_of_atom[index_gaussian]) * (self.displacement_vector[index_theta] - points_on_real_lines[index_theta]))  - 1)

                        theta_den += (prefactor) * 2.0 * np.exp(-exponent_of_atom[index_gaussian] * norm_of_radius)

            if index_theta == 0:
                theta_den = self.integration_of_molecular_density_over_space()
            #print("num", theta_num, "den", theta_den)
            theta_list[index_theta] = theta_num / theta_den

        return theta_list


if __name__ == "__main__":
    ELEMENT = "BE"
    ATOMIC_NUMBER = 4
    file_path = r"C:\Users\Alireza\PycharmProjects\fitting\fitting\data\examples\\" + ELEMENT + ".slater"

    ELEMENT_2 = "cl"
    ATOMIC_NUMBER_2 = 17
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
    cofactor_matrix = be.create_cofactor_matrix(be.UGBS_s_exponents)
    coeff = fitting_object.optimize_using_nnls(cofactor_matrix)
    coeff_exp = fitting_object.optimize_using_l_bfgs(np.concatenate((coeff, be.UGBS_s_exponents)), len(coeff))

    elec_dens = be.create_model(coeff_exp, len(coeff))
    print(be.integrated_total_electron_density)
    print(be.integrate_model_using_trapz(elec_dens))
    print(coeff_exp[:len(coeff)], "Coefficients")
    print(coeff_exp[len(coeff):], "Exp")
    plt.plot(be.electron_density)
    plt.plot(elec_dens)
    plt.show()


    molecular_dens = Molecular_Density_Transformation(elec_dens, np.reshape(coeff_exp[:len(coeff)], (-1, len(coeff))), np.reshape(coeff_exp[len(coeff):], (-1, len(coeff))), 2, np.array([0 for x in range(0, len(coeff))]))
    print(molecular_dens.integration_of_molecular_density_over_space())
    print(molecular_dens.new_weights(row_grid_points))





