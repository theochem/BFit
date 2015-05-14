import numpy as np
import re
elementFile = "/Users/Alireza/Desktop/neutral/ne"

def getExponents(input, subshell):
    """
    Obtains all exponents of the subshells from
    the file in the form of an array
    :param elementFile: the path to element
    :param subshells:
    :return:An array of the exponents of size (n, 1) where n is an integer
    """
    assert subshell == "S" or subshell == "P" or subshell == "D" or subshell == "F"

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

def getCoefficients(input, orbital):
    """
    Obtains the coefficients of a specific subshell
    from the file
    :param elementFile:
    :param subshell:
    :return: array of coefficients
    """

    coeffArray = np.zeros(shape = (20, 1))
    row = 0
    col = 0
    for line in input.split("\n"):
        if re.match(r'^\d' + orbital[1], line.lstrip()):
            coeffArray[row, 0] = line.split()[getColumn(orbital)]
            row += 1

    return np.trim_zeros(coeffArray) #a[:, np.apply_along_axis(np.count_nonzero, 0, a) >= 0.0001]

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

def getEnergy(input): # Need To Fix This
    input = input.split("\n")

    energy = []
    firstLine = input[1]
    secondLine = input[2]

    energy = (re.findall("[= -]\d+.\d+", firstLine + secondLine))
    return [float(x) for x in energy[:-1]]

def getOrbitals(input):
    """
    Finds the number of atomic orbitals
    :param path: path to the file containing element's data
    :return: number of atomic orbitals and list of all atomic orbitals of an element
    """
    counter = 0;
    listOfOrbitals = []


    typeOfOrbitals = ["  S  ", "  P  ", "  D  ", "  F  "]
    for line in input.split("\n"):   #This splits input into seperate lines
        for x in typeOfOrbitals:
            if x in line:
                listOfOrbitals += line.split()[1:]
                counter += len(line.split()) - 1
    return counter, listOfOrbitals

def getCusp(input):
    cusp = []
    a = input.split("\n")
    dict = {'S' : 0 , 'P' : 0, 'D': 0, 'F':0}
    for line in a:
        if len(re.findall('CUSP', line)) != 0:
            if dict['S'] == 0:
                dict['S'] = [float(x) for x in line.split()[1:]]
            elif dict['P'] == 0:
                dict['P'] = [float(x) for x in line.split()[1:]]
            elif dict['D'] == 0:
                dict['D'] = [float(x) for x in line.split()[1:]]
            elif dict['F'] == 0:
                dict['F'] = [float(x) for x in line.split()[1:]]

    return {key:value for key,value in dict.items() if value != 0}

def getOrbitalEnergy(input):
    dict = {'S' : 0 , 'P' : 0, 'D': 0, 'F':0}
    for line in input.split("\n"):
        if (re.match('BASIS/ORB.ENERGY', line.lstrip())):
            if dict['S'] == 0:
                dict['S'] = [float(x) for x in line.split()[1:]]
            elif dict['P'] == 0:
                dict['P'] = [float(x) for x in line.split()[1:]]
            elif dict['D'] == 0:
                dict['D'] = [float(x) for x in line.split()[1:]]
            elif dict['F'] == 0:
                dict['F'] = [float(x) for x in line.split()[1:]]

    return {key:value for key,value in dict.items() if value != 0}

def getOrbitalBasis(input):
    dict = {'S': [], 'P': [], 'D': [], 'F': []}

    for line in input.split("\n"):
        for subshell in ["S", "P", "D", "F"]:
            if re.match(r'^\d' + subshell, line.lstrip()):
                #print(line)
                dict[subshell].append(line.split()[0])
    return {key:value for key,value in dict.items() if len(value) != 0}

def getOrbitalExponents(input):
    dict = {'S':0, 'P':0, 'D':0, 'F':0}
    for key in dict:
        dict[key] = getExponents(input, key)
    return {key:value for key,value in dict.items() if len(value) != 0}

def getOrbitalCoefficient(input):
    dict = {}
    for orbital in getOrbitals(input)[1]:
        #print(orbital)
        dict[orbital] = getCoefficients(input, orbital)
    return dict

def load_slater_basis(file):
    file = open(file, 'r')
    input = file.read()

    return {'configuration': input.split("\n")[0].split()[1].replace(",", "") ,
            'energy':getEnergy(input) ,
            'orbitals':getOrbitals(input)[1] ,
            'orbitals_energy': getOrbitalEnergy(input) ,
            'orbitals_cusp': getCusp(input),
            'orbitals_basis': getOrbitalBasis(input),
            'orbitals_exp': getOrbitalExponents(input),
            'orbitals_coeff': getOrbitalCoefficient(input)}
#print(load_slater_basis(elementFile))