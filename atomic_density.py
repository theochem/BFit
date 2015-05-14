import sys
sys.path.append(r'C:\Users\Alireza\PycharmProjects\fitting\io')
import slater_basic as sb
import numpy as np
import scipy.misc
import scipy

elementFile = "/Users/Alireza/Desktop/neutral/be"

a = sb.load_slater_basis(elementFile)

def getSlatorTypeOrbital(exponent, quantumNum, r : 'distance'):
    return ((2 * exponent)**quantumNum)   *    np.sqrt(((2 * exponent) / scipy.misc.factorial(2 * quantumNum)))    *      (r ** (quantumNum - 1)) * (np.exp(-exponent * r))

print(getSlatorTypeOrbital(a['orbitals_exp']['S'], sb.getQuantumNumber(elementFile, 'S'), [2,2]))

def getLCAO(slatorFunction, coeffMatrix):
    """
    Calculates phi/linear combination of atomic orbitals
    by the dot product of slator and coeffmatrix
    :param slatorFunction:
    :param coeffMatrix:
    :return: a new matrix
    """
    return np.dot(np.transpose(slatorFunction) , coeffMatrix)

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
