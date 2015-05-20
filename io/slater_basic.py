import re
import numpy as np

def load_slater_basis(file_name):
    '''Return the data recorded in the Slater atomic density file as a dictionary.

       ** Arguments **

       file_name
           The path to the Slater atomic density file.
    '''
    def getNumberOfElectronsPerOrbital(string_configuration):
        """
        Gets the Occupation Number for all orbitals
        of an element returing an dictionary
        :param elementFile:
        :return: a dict containing the number and orbital
        """
        electronConfigList = string_configuration

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

    def getArrayOfElectrons(dict):
        """
        Computes The Number Of Electrons in Each Orbital As An Array
        i.e. be = [[2], [2]] 2 electrons in 1S and 2S
        :param dict:
        :return: column vector of number of electrons in each orbital
        """
        array = np.empty((len(dict.keys()), 1))
        row = 0
        for orbital in orbitals:
            array[row, 0] = dict[orbital]
            row += 1
        return array

    with open(file_name) as f:
        configuration = f.readline().split()[1].replace(",", "")
        energy = [float(f.readline().split()[2])] + [float(x) for x in (re.findall("[= -]\d+.\d+", f.readline()))[:-1]]
        assert re.search(r'ORBITAL ENERGIES AND EXPANSION COEFFICIENTS', f.readline())

        orbitals = []
        orbitals_basis = {'S':[], 'P':[], 'D':[], "F":[]}
        orbitals_cusp = {'S':0, 'P':0, 'D':0, "F":0}
        orbitals_energy = {'S':[], 'P':[], 'D':[], "F":[]}
        orbitals_exp = {'S':[], 'P':[], 'D':[], "F":[]}
        orbitals_coeff = {}

        line = f.readline()
        while line.strip() != "":

            if re.search(r'  [S|P|D|F]  ', line): #if line has ___S___ or P or D where _ = " ".
                #Get All The Orbitals
                subshell = line.split()[0]
                list_of_orbitals = line.split()[1:]
                orbitals += list_of_orbitals
                for x in list_of_orbitals:
                    orbitals_coeff[x] = []   #initilize orbitals inside coefficient dictionary

                #Get Energy, Cusp Levels
                line = f.readline()
                orbitals_energy[subshell] = [float(x) for x in line.split()[1:]]
                line = f.readline()
                orbitals_cusp[subshell] = [float(x) for x in line.split()[1:]]
                line = f.readline()


                #Get Exponents, Coefficients, Orbital Basis
                while re.match(r'\A^\d' + subshell, line.lstrip()):

                    list_words = line.split()
                    orbitals_exp[subshell] += [float(list_words[1])]
                    orbitals_basis[subshell] += [list_words[0]]

                    for x in list_of_orbitals:
                        orbitals_coeff[x] += [float(list_words[getColumn(x)])]
                    line = f.readline()


    data = {'configuration': configuration ,
            'energy': energy,
            'orbitals': orbitals ,
            'orbitals_energy': orbitals_energy ,
            'orbitals_cusp': orbitals_cusp,
            'orbitals_basis': orbitals_basis,
            'orbitals_exp':
            {key:np.asarray(value).reshape(len(value), 1) for key,value in orbitals_exp.items() if value != []},
            'orbitals_coeff':
            {key:np.asarray(value).reshape(len(value), 1) for key,value in orbitals_coeff.items() if value != []},
            'orbitals_occupation': getNumberOfElectronsPerOrbital(configuration),
            'orbitals_electron_array': getArrayOfElectrons(getNumberOfElectronsPerOrbital(configuration)),
            'basis_numbers' :
            {key:np.asarray([[int(x[0])] for x in value]) for key,value in orbitals_basis.items() if len(value) != 0}
            }

    return data
