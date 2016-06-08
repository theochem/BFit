import numpy as np
from scipy.special import erf

class Molecular_Density_Transformation():
    #TODO: Ask them if they want |R - R_{alpha}| or |R|
    def __init__(self, molecular_density, coefficients_per_atom, exponents_per_atom, num_of_coordinates, displacement_vector):
        assert coefficients_per_atom.ndim == 2
        assert  exponents_per_atom.ndim == 2
        assert (coefficients_per_atom).shape[0] == exponents_per_atom.shape[0]
        assert  num_of_coordinates == 0 or num_of_coordinates == 1 or num_of_coordinates == 3

        self.number_of_atoms = (coefficients_per_atom).shape[0]
        self.dimension = num_of_coordinates * self.number_of_atoms
        self.molecular_density = molecular_density
        self.coefficients_per_atom = coefficients_per_atom
        self.exponents_per_atom = exponents_per_atom
        self.positioned_exponents = exponents_per_atom - displacement_vector
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
        theta_list = np.empty(self.dimension)

        for index_theta in range(0, self.dimension):
            theta_num = 0
            theta_den = 0

            prefactor_exponent = (self.dimension - index_theta)/2.0
            for index_atom in range(0, self.number_of_atoms):
                coefficient_of_atom = self.coefficients_per_atom[index_atom]
                exponent_of_atom = self.exponents_per_atom[index_atom]


                for index_gaussian in range(0, len(coefficient_of_atom)):
                    prefactor = coefficient_of_atom[index_gaussian] * 0.5 * (np.pi / exponent_of_atom[index_gaussian])**prefactor_exponent

                    if index_atom != 0:
                        theta_num += prefactor * (erf(np.sqrt(exponent_of_atom[index_gaussian]) * (points_on_real_lines[theta_list] - self.displacement_vector[index_theta])) - 1)
                    else:
                        norm_of_radius = 0
                        for index_prev_calc_theta in range(0, index_atom):
                            norm_of_radius += (theta_list[index_theta] - self.displacement_vector[index_theta])**2

                        theta_num -= prefactor * np.exp(-exponent_of_atom[index_gaussian] * norm_of_radius) * \
                                 (erf(np.sqrt(exponent_of_atom[index_gaussian]) * (self.displacement_vector[index_theta] - points_on_real_lines[index_theta]))  - 1)

                        theta_den += (prefactor / 2.0) * np.exp(-exponent_of_atom[index_gaussian] * norm_of_radius)


            if index_theta == 0:
                theta_den = self.integration_of_molecular_density_over_space()
            theta_list[index_theta] = theta_num / theta_den

        return theta_list


if __name__ == "__main__":

    pass