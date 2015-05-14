import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
import slater_basic as sb
import grid
import numpy as np
import scipy.misc
import scipy

elementFile = "/Users/Alireza/Desktop/neutral/ne"


class Electron_Structure():
    def __init__(self, file, grid):
        self.element = sb.Element_Data(file)
        self.values = self.element.slater_basis
        self.grid = grid
        self.all_slator_orbitals = self.slator_dict()

    def slator_type_orbital(self, exponent, quantumNum, r : 'distance'):
        return ((2 * exponent)**quantumNum)   *    np.sqrt(((2 * exponent) / scipy.misc.factorial(2 * quantumNum)))    *      (r ** (quantumNum - 1)) * (np.exp(-exponent * r))

    def slator_dict(self):
        dict = {x[1]:0 for x in self.values['orbitals'] }
        for subshell in dict:
            dict = np.transpose(self.slator_type_orbital(self.values['orbitals_exp'][subshell], self.values['quantum_numbers'][subshell], self.grid ))
        return dict

    def all_coeff_matrix(self, subshell):
        pass

    def phi_LCAO(self):
        """
        Calculates phi/linear combination of atomic orbitals
        by the dot product of slator and coeffmatrix
        :param slatorFunctio n:
        :param coeffMatrix:
        :return: a new matrix
        """
        slator_matrix = slator_type_orbital(values['orbital_exp'])
        return np.dot(self.all_slator_orbitals , coeffMatrix)

    def atomic_density(dict :"Occupation Numbers", LCAO : "Matrix rows = points, column =  phi"):
        """
        By Taking the occupation numbers and multiplying it
        to the corresponding phi to obtain rho
        :param dict:
        :param LCAO:
        :return:
        """
        listofAllOrbitals = [str(x) + "S" for x in range(1,6)] + [str(x) + "P" for x in range(2,6)] + [str(x) + "D" for x in range(3,6)] + [str(x) + "F" for x in range(4,6)]

        row, col = np.shape(LCAO)


        column = 0
        for orbital in listofAllOrbitals:
            if dict[orbital] != 0 and column < col:
                LCAO[:,column] = np.absolute(LCAO[:,column] * LCAO[:,column]) * dict[orbital]
                column += 1
        return LCAO

p, w = np.polynomial.laguerre.laggauss(100)
be = Electron_Structure(elementFile, p)

print(be.all_coeff_matrix('S'))

