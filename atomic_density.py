import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\fitting\io')
import slater_basic2 as sb
import numpy as np
import scipy.misc
import scipy
import scipy.integrate
import matplotlib.pyplot as plt


elementFile = "/Users/Alireza/Desktop/neutral/be"


class Electron_Structure():
    def __init__(self, file, grid):
        self.values = sb.load_slater_basis(file)
        self.grid = grid
        self.all_slator_orbitals = self.slator_dict()

    def slator_type_orbital(self, exponent, quantumNum, r):
        """
        Computes the Slator Type Orbital
        :param exponent: alpha
        :param quantumNum: principal quantum number
        :param r: distance form the nuclei
        :return: an number or array depending on input
        """
        return ((2 * exponent)**quantumNum)   *    np.sqrt(((2 * exponent) / scipy.misc.factorial(2 * quantumNum)))    *      (r ** (quantumNum - 1)) * (np.exp(-exponent * r))

    def slator_dict(self):
        """
        Groups Each Slater Equations Based On The SubShell inside an dictionary.
        This is then used to multiply by coefficient array to obtain all phi equations
        for that subshell. Hence each subshell will have their own slator matrix
        and their own coefficient matrix, dot product between them will obtain
        the phi equations(MO) for that subshell.
        :return: row = number of points, column = number of slater equations
        """
        dict = {x[1]:0 for x in self.values['orbitals'] }
        for subshell in dict:
            dict[subshell] = np.transpose(self.slator_type_orbital(self.values['orbitals_exp'][subshell], self.values['basis_numbers'][subshell], self.grid ))

        return dict

    def all_coeff_matrix(self, subshell):
        """
        This Groups all of the coefficients based on the subshell.
        This is used to multiply by the specific slator array from the
        slator_dict function. In order to obtain a phi array
        :param subshell: this is either S or P or D Or F
        :return: an array wherre rows = number of slater/basis and columns = number of orbitals or phi
        """
        subshell_coeffs = {key:values for key, values in self.values['orbitals_coeff'].items() if subshell == key[1]}
        a = 0;
        array_coeffs = 0
        for key in [x for x in self.values['orbitals'] if x[1] == subshell]:
            if a== 0:
                array_coeffs = self.values['orbitals_coeff'][key]
                a += 1;
            else:
                array_coeffs = np.concatenate((array_coeffs, self.values['orbitals_coeff'][key]), axis = 1)

        return array_coeffs

    def phi_LCAO(self, subshell):
        """
        Calculates phi/linear combination of atomic orbitals
        by the dot product of slator array (from slator_dict)
        and coeff array (from all_coeff_matrix(subshell)) for
        a specific subshell. Hence, to obtain all of the
        phi equations for the specific element it must be
        repeated for each subshell.
        :param slatorFunctio n:
        :param coeffMatrix:
        :return: array where row = number of points and column = number of phi/orbitals.
        For example, beryllium will have row = # of points and column = 2 (1S and 2S)
        """
        return np.dot(self.all_slator_orbitals[subshell] , self.all_coeff_matrix(subshell))

    def phi_matrix(self): #connect all phis together
        """
        Connects phi equations into an array, horizontally
        For Example, for beryllium [phi(1S), phi(2S)] is the array
        E.G. Carbon [phi(1S), phi(2S), phi(2P)]
        :return: array where all of the phi equations
        for each orbital is connected together, horizontally.
        row = number of points and col = each phi equation for each orbital
        """
        list_orbitals = ['S', 'P', 'D', 'F']

        a = 0
        array = 0
        for orbital in list_orbitals:
            if orbital in self.values['orbitals_exp']:
                if a == 0:
                    array = self.phi_LCAO(orbital)
                    a += 1
                else:
                    array = np.concatenate((array, self.phi_LCAO(orbital)), axis = 1)
        print(array)
        return array

    def atomic_density(self):
        """
        By Taking the occupation numbers and multiplying it
        to the corresponding phi to obtain rho
        :param dict:
        :param LCAO:
        :return: the electron density where row = number of points
                and column = 1
        """
        listofAllOrbitals = [str(x) + "S" for x in range(1,6)] + [str(x) + "P" for x in range(2,6)] + [str(x) + "D" for x in range(3,6)] + [str(x) + "F" for x in range(4,6)]
        return np.dot(np.absolute(self.phi_matrix()**1), self.values['orbitals_electron_array'] )






p, w = np.polynomial.laguerre.laggauss(100)
#print(p)
be = Electron_Structure(elementFile, p)

electron_density = be.atomic_density()
print('electron_density', electron_density)
#assert np.shape(electron_density) == (100, 1)

plt.plot(electron_density)
plt.show()

p = np.asarray(p).reshape((100, 1))
ED_times_4pir_squared = np.absolute(electron_density ** 2) * 4 * np.pi * p * p
plt.plot(ED_times_4pir_squared)
plt.show()

ED_divide_by_enegativer = ED_times_4pir_squared / np.exp(-(p))
plt.plot(ED_divide_by_enegativer)
plt.show()

w = np.asarray(w).reshape((100, 1))
ED_times_weight = ED_divide_by_enegativer * w
plt.plot(ED_times_weight)
plt.show()

print("The Sum Is: ", np.sum(ED_times_weight))
print("The Sum Is: " , np.sum(ED_times_weight)/ 430)






