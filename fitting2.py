import numpy as np;
import math
import scipy.misc
import scipy.integrate
import matplotlib.pyplot as mp
import os
import re


path = '/Users/Alireza/Desktop/neutral'
elementFile = "/Users/Alireza/Desktop/neutral/be"
phiFile = "/Users/Alireza/PycharmProjects/IDKWHATTHISIS/phi.txt"

"""
ALGORITHM
1) Obtain Array Of Exponents = E
2) Obtain Array Of Coefficients = C
3) Obtain Array Of QuantumNumber = QN
4) Obtain Array Of R Values(grid) = R
5) Create Slator Function with 3 parameters, alpha = exponent , quantumNumber equal to first column, r which is obtained from grid S(E,QN,R)
6) Add Three Arrays To Slator Eqn
7) Multiply Coefficients Array to Slator Eqn To Obtain a LCAO.
8) Multiply Each LCAO by corresponding Electron Configuration to get rho (electron density)
9) Multiply rho by weight of points to obtain an approximation for integration
10) Add the sum of Points Together to obtain the integration of rho
"""

def getExponents(elementFile, subshell):
    """
    Obtains all exponents of the subshells from
    the file in the form of an array
    :param elementFile: the path to element
    :param subshells:
    :return:An array of the exponents of size (n, 1) where n is an integer
    """
    assert subshell == "S" or subshell == "P" or subshell == "D" or subshell == "F"
    file = open(elementFile, 'r')
    input = file.read()
    #print(input)

    exponentArray = np.zeros(shape = (30, 1)) #create function to check shape

    i = 0
    for line in input.split("\n"):
        if re.match(r'^\d' + subshell[0], line.lstrip()):
            rowList = line.split()
            exponentArray[i] = float(rowList[1])
            i += 1

    return np.trim_zeros(exponentArray)

def getColumn(Torbital):
    """
    The Columns get screwed over sine s orbitals start with one while p orbitals start at energy 2
    Therefore this corrects the error in order to retrieve correct column
    :param Torbital: orbital i.e. "1S" or "2P" or "3D"
    :return:
    """
    if Torbital[1] == "S":
        return int(Torbital[0]) + 1
    elif Torbital[1] == "P":
        return int(Torbital[0])
    elif Torbital[1] == "D":
        return int(Torbital[0]) - 1

def getCoefficients(elementFile, subshell):
    """
    Obtains the coefficients of a specific subshell
    from the file
    :param elementFile:
    :param subshell:
    :return: array of coefficients
    """
    assert subshell == "S" or subshell == "P" or subshell == "D" or subshell == "F"
    file = open(elementFile, 'r')
    input = file.read()

    coeffArray = np.zeros(shape = (20, 10))
    row = 0
    col = 0
    for line in input.split("\n"):
        if re.match(r'^\d' + subshell, line.lstrip()):
            #print(line.split())
            for coeff in line.split()[2:]:
                coeffArray[row, col] = coeff
                col += 1
            col = 0
            row += 1

    a = coeffArray[~np.all(coeffArray == 0, axis=1)]
    condition = np.mod(a, 3)!=0

    return np.delete(a, np.nonzero((a==0).sum(axis=0) > 0), axis=1) #a[:, np.apply_along_axis(np.count_nonzero, 0, a) >= 0.0001]

def getQuantumNumber(elementFile, subshell):
    """
    get the quantum numbers for a specific subshell

    :param elementFile:
    :param subshell:
    :return:
    """
    assert subshell == "S" or subshell == "P" or subshell == "D" or subshell == "F"
    file = open(elementFile, 'r')
    input = file.read()

    quantNumArray = np.zeros(shape = (20, 1))
    i = 0
    for line in input.split("\n"):
        if re.match(r'^\d' + subshell, line.lstrip()):
            quantNumArray[i] = int(line.split()[0][0])
            i += 1

    return np.trim_zeros(quantNumArray)

def getSlatorTypeOrbital(exponent, quantumNum, r : 'distance'):
    return ((2 * exponent)**quantumNum)   *    np.sqrt(((2 * exponent) / scipy.misc.factorial(2 * quantumNum)))    *      (r ** (quantumNum - 1)) * (np.exp(-exponent * r))

p, w = np.polynomial.laguerre.laggauss(100)

def getLCAO(slatorFunction, coeffMatrix):
    """
    Calculates phi/linear combination of atomic orbitals
    by the dot product of slator and coeffmatrix
    :param slatorFunction:
    :param coeffMatrix:
    :return: a new matrix
    """
    return np.dot(np.transpose(slatorFunction) , coeffMatrix)
LCAO = getLCAO(getSlatorTypeOrbital(getExponents(elementFile, 'S'), getQuantumNumber(elementFile, 'S'), p ), getCoefficients(elementFile, "S"))
#print(LCAO)

def getOccupationNumber(elementFile):
    """
    Gets the Occupation Number for all orbitals
    of an element
    :param elementFile:
    :return: a dict containing the number and orbital
    """
    file = open(elementFile, 'r')
    electronConfigList = file.readline().split()[1]

    shells = ["K", "L", "M", "N"]

    myDic = {}
    listOrbitals = [str(x) + "S" for x in range(1,8)] + [str(x) + "P" for x in range(2,8)] + [str(x) + "D" for x in range(3,8)] + [str(x) + "F" for x in range(4,8)]
    for list in listOrbitals:
        myDic[list] = 0 #initilize all atomic orbitals to zero electrons

    for x in shells:
        if x in electronConfigList :
            if x == "K":
                myDic["1S"] = 2
            elif x == "L":
                myDic["2S"] = 2; myDic["2P"] = 6
            elif x == "M":
                myDic["3S"] = 2; myDic["3P"] = 6; myDic["3D"] = 10
            elif x== "N":
                myDic["4S"] = 2; myDic["4P"] = 6; myDic["4D"] = 10; myDic["4F"] = 14

    for x in listOrbitals:
        if x in electronConfigList :
            index = electronConfigList .index(x)
            orbital = (electronConfigList [index: index + 2])

            if orbital[1] == "D" or orbital[1] == "F":
                numElectrons = (re.sub('[(){}<>,]', "", electronConfigList.split(orbital)[1]))
                myDic[orbital] = int(numElectrons)
            else:
                myDic[ orbital] = int(  electronConfigList [index + 3: index + 4]  )

    return myDic

def rho(dict :"Occupation Numbers", LCAO : "Matrix rows = points, column =  phi"):
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
rho1 = (rho(getOccupationNumber(elementFile), LCAO)); print(rho)

def divideByr(rho, r, w):
    row, col = np.shape(rho)

    for x in range(col):
        rho[:,x] = (rho[:,x] * w * 4 * np.pi * r**2)/ np.exp(-r)
    return rho

print(np.sum(divideByr(rho(getOccupationNumber(elementFile), LCAO), 1 , w)) )



