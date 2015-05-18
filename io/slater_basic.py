import numpy as np
import re
elementFile = "/Users/Alireza/Desktop/neutral/be"

class Element_Data():

    def __init__(self, file):
        f = open(file, 'r')
        self.input = f.read()
        self.slater_basis = self.load_slater_basis()
        f.close()

    def getExponents(self, subshell):
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
        for line in self.input.split("\n"):
            if re.match(r'^\d' + subshell[0], line.lstrip()):
                rowList = line.split()
                exponentArray[i] = float(rowList[1])
                i += 1

        return np.trim_zeros(exponentArray)

    def getColumn(self, Torbital):
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

    def getCoefficients(self, orbital):
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
        for line in self.input.split("\n"):
            if re.match(r'^\d' + orbital[1], line.lstrip()):
                coeffArray[row, 0] = line.split()[self.getColumn(orbital)]
                row += 1

        return np.trim_zeros(coeffArray) #a[:, np.apply_along_axis(np.count_nonzero, 0, a) >= 0.0001]

    def getQuantumNumber(self):
        """
        get the quantum numbers for a specific subshell

        :param elementFile:
        :param subshell:
        :return:
        """
        dict = {'S' : 0 , 'P' : 0, 'D': 0, 'F':0}

        quantNumArray = np.zeros(shape = (20, 1))
        i = 0
        for line in self.input.split("\n"):
            if re.match(r'^\d', line.lstrip()):
                quantNumArray[i] = int(line.split()[0][0])
                i += 1

        return np.trim_zeros(quantNumArray)

    def getEnergy(self): # Need To Fix This
        input = self.input.split("\n")

        energy = []
        firstLine = input[1]
        secondLine = input[2]

        energy = (re.findall("[= -]\d+.\d+", firstLine + secondLine))
        return [float(x) for x in energy[:-1]]

    def getOrbitals(self):
        """
        Finds the number of atomic orbitals
        :param path: path to the file containing element's data
        :return: number of atomic orbitals and list of all atomic orbitals of an element
        """
        counter = 0;
        listOfOrbitals = []


        typeOfOrbitals = ["  S  ", "  P  ", "  D  ", "  F  "]
        for line in self.input.split("\n"):   #This splits input into seperate lines
            for x in typeOfOrbitals:
                if x in line:
                    listOfOrbitals += line.split()[1:]
                    counter += len(line.split()) - 1
        return counter, listOfOrbitals

    def getCusp(self):
        cusp = []
        a = self.input.split("\n")
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

    def getOrbitalEnergy(self):
        dict = {'S' : 0 , 'P' : 0, 'D': 0, 'F':0}
        for line in self.input.split("\n"):
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

    def getOrbitalBasis(self):
        dict = {'S': [], 'P': [], 'D': [], 'F': []}

        for line in self.input.split("\n"):
            for subshell in ["S", "P", "D", "F"]:
                if re.match(r'^\d' + subshell, line.lstrip()):
                    #print(line)
                    dict[subshell].append(line.split()[0])

        return {key:value for key,value in dict.items() if len(value) != 0}

    def getOrbitalExponents(self):
        dict = {'S':0, 'P':0, 'D':0, 'F':0}
        for key in dict:
            dict[key] = self.getExponents(key)
        return {key:value for key,value in dict.items() if len(value) != 0}

    def getOrbitalCoefficient(self):
        dict = {}
        for orbital in self.getOrbitals()[1]:
            #print(orbital)
            dict[orbital] = self.getCoefficients(orbital)
        return dict

    def getNumberOfElectronsPerOrbital(self):
        """
        Gets the Occupation Number for all orbitals
        of an element
        :param elementFile:
        :return: a dict containing the number and orbital
        """
        electronConfigList = self.input.split("\n")[0].split()[1]

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

        return {key:value for key,value in myDic.items() if value != 0}

    def getQuantumNumbers(self):
        dict = {'S': [], 'P': [], 'D': [], 'F': []}

        for line in self.input.split("\n"):
            for subshell in ["S", "P", "D", "F"]:
                if re.match(r'^\d' + subshell, line.lstrip()):
                    #print(line)
                    dict[subshell].append(line.split()[0][0])

        return {key:np.asarray([[int(x)] for x in value]) for key,value in dict.items() if len(value) != 0}

    def load_slater_basis(self):

        return {'configuration': self.input.split("\n")[0].split()[1].replace(",", "") ,
                'energy':self.getEnergy() ,
                'orbitals':self.getOrbitals()[1] ,
                'orbitals_energy': self.getOrbitalEnergy() ,
                'orbitals_cusp': self.getCusp(),
                'orbitals_basis': self.getOrbitalBasis(),
                'orbitals_exp': self.getOrbitalExponents(),
                'orbitals_coeff': self.getOrbitalCoefficient(),
                'orbitals_electron_number' : self.getNumberOfElectronsPerOrbital(),
                'quantum_numbers' : self.getQuantumNumbers()}

be = Element_Data(elementFile)
print(be.slater_basis)