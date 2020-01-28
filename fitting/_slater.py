# -*- coding: utf-8 -*-
# FittingBasisSets is a basis-set curve-fitting optimization package.
#
# Copyright (C) 2018 The FittingBasisSets Development Team.
#
# This file is part of FittingBasisSets.
#
# FittingBasisSets is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# FittingBasisSets is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""Module responsible for reading and storing information from '.slater' files.

Functions
---------
load_slater_wfn : Function for reading and returning information from '.slater' files.

"""

import os
import re
import numpy as np

__all__ = ["load_slater_wfn"]


def load_slater_wfn(element):
    """
    Return the data recorded in the atomic Slater wave-function file as a dictionary.

    Parameters
    ----------
    file_name : str
        The path to the Slater atomic file.

    """
    file_name = os.path.join(os.path.dirname(__file__) + "/data/examples/%s.slater" % element.lower())

    def get_number_of_electrons_per_orbital(configuration):
        """
        Get the Occupation Number for all orbitals of an _element returing an dictionary.

        Parameters
        ----------
        configuration : str
            The electron configuration.

        Returns
        --------
        dict
            a dict containing the number and orbital.

        """
        electron_config_list = configuration

        shells = ["K", "L", "M", "N"]

        out = {}
        orbitals = [str(x) + "S" for x in range(1, 8)] + [str(x) + "P" for x in range(2, 8)] + \
                   [str(x) + "D" for x in range(3, 8)] + [str(x) + "F" for x in range(4, 8)]
        for orb in orbitals:
            # Initialize all atomic orbitals to zero electrons
            out[orb] = 0

        for x in shells:
            if x in electron_config_list:
                if x == "K":
                    out["1S"] = 2
                elif x == "L":
                    out["2S"] = 2
                    out["2P"] = 6
                elif x == "M":
                    out["3S"] = 2
                    out["3P"] = 6
                    out["3D"] = 10
                elif x == "N":
                    out["4S"] = 2
                    out["4P"] = 6
                    out["4D"] = 10
                    out["4F"] = 14

        for x in orbitals:
            if x in electron_config_list:
                index = electron_config_list.index(x)
                orbital = (electron_config_list[index: index + 2])

                if orbital[1] == "D" or orbital[1] == "F":
                    num_electrons = re.sub('[(){}<>,]', "", electron_config_list.split(orbital)[1])
                    out[orbital] = int(num_electrons)
                else:
                    out[orbital] = int(electron_config_list[index + 3: index + 4])

        return {key: value for key, value in out.items() if value != 0}

    def get_column(t_orbital):
        """
        Correct the error in order to retrieve the correct column.

        The Columns are harder to parse since the orbitals start with one while p orbitals start at energy.

        Parameters
        ----------
        t_orbital : str
            orbital i.e. "1S" or "2P" or "3D"

        Returns
        -------
        int :
            Retrieve teh right column index depending on whether it is "S", "P" or "D" orbital.
        
        """
        if t_orbital[1] == "S":
            return int(t_orbital[0]) + 1
        elif t_orbital[1] == "P":
            return int(t_orbital[0])
        elif t_orbital[1] == "D":
            return int(t_orbital[0]) - 1

    with open(file_name, "r") as f:
        line = f.readline()
        configuration = line.split()[1].replace(",", "")

        next_line = f.readline()
        while len(next_line.strip()) == 0:
            next_line = f.readline()
        energy = [float(f.readline().split()[2])] + \
                 [float(x) for x in (re.findall(r"[= -]\d+.\d+", f.readline()))[:-1]]

        orbitals = []
        orbitals_basis = {'S': [], 'P': [], 'D': [], "F": []}
        orbitals_cusp = []
        orbitals_energy = []
        orbitals_exp = {'S': [], 'P': [], 'D': [], "F": []}
        orbitals_coeff = {}

        line = f.readline()
        while line.strip() != "":
            # If line has ___S___ or P or D where _ = " ".
            if re.search(r'  [S|P|D|F]  ', line):
                # Get All The Orbitals
                subshell = line.split()[0]
                list_of_orbitals = line.split()[1:]
                orbitals += list_of_orbitals
                for x in list_of_orbitals:
                    orbitals_coeff[x] = []   # Initilize orbitals inside coefficient dictionary

                # Get Energy, Cusp Levels
                line = f.readline()
                orbitals_energy.extend([float(x) for x in line.split()[1:]])
                line = f.readline()
                orbitals_cusp.extend([float(x) for x in line.split()[1:]])
                line = f.readline()

                # Get Exponents, Coefficients, Orbital Basis
                while re.match(r'\A^\d' + subshell, line.lstrip()):

                    list_words = line.split()
                    orbitals_exp[subshell] += [float(list_words[1])]
                    orbitals_basis[subshell] += [list_words[0]]

                    for x in list_of_orbitals:
                        orbitals_coeff[x] += [float(list_words[get_column(x)])]
                    line = f.readline()
            else:
                line = f.readline()

    data = {'configuration': configuration,
            'energy': energy,
            'orbitals': orbitals,
            'orbitals_energy': np.array(orbitals_energy)[:, None],
            'orbitals_cusp': np.array(orbitals_cusp)[:, None],
            'orbitals_basis': orbitals_basis,
            'orbitals_exp':
            {key: np.asarray(value).reshape(len(value), 1) for key, value in orbitals_exp.items()
             if value != []},
            'orbitals_coeff':
            {key: np.asarray(value).reshape(len(value), 1)
             for key, value in orbitals_coeff.items() if value != []},
            'orbitals_occupation': np.array([get_number_of_electrons_per_orbital(configuration)[k] for k in orbitals])[:, None],
            'basis_numbers':
            {key: np.asarray([[int(x[0])] for x in value])
             for key, value in orbitals_basis.items() if len(value) != 0}
            }

    return data
