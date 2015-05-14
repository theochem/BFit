import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
import slater_basic as sb
import numpy as np
import scipy.misc
import scipy

elementFile = "/Users/Alireza/Desktop/neutral/be"

a = sb.load_slater_basis(elementFile)


def slator_type_orbital(exponent, quantumNum, r : 'distance'):
    return ((2 * exponent)**quantumNum)   *    np.sqrt(((2 * exponent) / scipy.misc.factorial(2 * quantumNum)))    *      (r ** (quantumNum - 1)) * (np.exp(-exponent * r))

def phi_LCAO(slatorFunction, coeffMatrix):
    """
    Calculates phi/linear combination of atomic orbitals
    by the dot product of slator and coeffmatrix
    :param slatorFunction:
    :param coeffMatrix:
    :return: a new matrix
    """
    return np.dot(np.transpose(slatorFunction) , coeffMatrix)

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
