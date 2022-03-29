# -*- coding: utf-8 -*-
# BFit is a Python library for fitting a convex sum of Gaussian
# functions to any probability distribution
#
# Copyright (C) 2020- The QC-Devs Community
#
# This file is part of BFit.
#
# BFit is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# BFit is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# ---
r"""Parsing Slater files."""
import os
import re

import numpy as np


__all__ = ["load_slater_wfn"]


def load_slater_wfn(element, anion=False, cation=False):
    """
    Return the data inside the atomic Slater '.slater' files wave-function file as a dictionary.

    Parameters
    ----------
    element : str
        The atom/element.
    anion : bool
        If true, then the anion of element is used.
    cation : bool
        If true, then the cation of element is used.
    """
    # Heavy atoms from atom cs to lr.
    heavy_atoms = ["cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy", "ho",
                   "er", "tm", "yb", "lu", "hf", "ta", "w", "re", "os", "ir", "pt", "au", "hg",
                   "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th", "pa", "u", "np",
                   "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no", "lr"]

    anion_atoms = ["ag", "al", "as", "b", "br", "c", "cl", "co", "cr", "cu", "f", "fe", "ga",
                   "ge", "h", "i", "in", "k", "li", "mn", "mo", "n", "na", "nb", "ni", "o",
                   "p", "pd", "rb", "rh", "ru", "s", "sb", "sc", "se", "si", "sn", "tc", "te",
                   "ti", "v", "y", "zr"]

    cation_atoms = ["ag", "al", "ar", "as", "b", "be", "br", "c", "ca", "cd", "cl", "co",
                    "cr", "cs", "cu", "f", "fe", "ga", "ge", "i", "in", "k", "kr", "li",
                    "mg", "mn", "mo", "n", "na", "nb", "ne", "ni", "o", "p", "pd", "rb",
                    "rh", "ru", "s", "sb", "sc", "se", "si", "sn", "sr", "tc", "te", "ti",
                    "v", "xe", "y", "zn", "zr"]

    is_heavy_element = element.lower() in heavy_atoms
    if anion:
        if element.lower() in anion_atoms:
            file_path = f"/data/anion/{element.lower()}.an"
        else:
            raise ValueError(
                f"Anion Slater File for element {element} does not exist."
            )
    elif cation:
        if element.lower() in cation_atoms:
            file_path = f"/data/cation/{element.lower()}.cat"
        else:
            raise ValueError(
                f"Cation Slater File for element {element} does not exist."
            )
    else:
        file_path = f"/data/neutral/{element.lower()}.slater"

    file_name = os.path.join(os.path.dirname(__file__) + file_path)

    def _get_number_of_electrons_per_orbital(configuration):
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

        # Sometimes electron configuration includes K L M N for simplicity
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
                    num_electrons = re.search(orbital + r"\((.*?)\)", electron_config_list).group(1)
                    out[orbital] = int(num_electrons)
                else:
                    out[orbital] = int(electron_config_list[index + 3: index + 4])

        return {key: value for key, value in out.items() if value != 0}

    def _get_column(t_orbital):
        """
        Correct the error in order to retrieve the correct column.

        The Columns are harder to parse as the orbitals start with one while p orbitals start at
            energy.

        Parameters
        ----------
        t_orbital : str
            orbital i.e. "1S" or "2P" or "3D"

        Returns
        -------
        int :
            Retrieve the right column index depending on whether it is "S", "P" or "D" orbital.

        """
        if t_orbital[1] == "S":
            return int(t_orbital[0]) + 1
        elif t_orbital[1] == "P":
            return int(t_orbital[0])
        elif t_orbital[1] == "D":
            return int(t_orbital[0]) - 1
        elif t_orbital[1] == "F":
            return int(t_orbital[0]) - 2
        else:
            raise ValueError(f"Did not recognize orbital {t_orbital}.")

    def _configuration_exact_for_heavy_elements(configuration):
        r"""Later file for heavy elements does not contain the configuration in right format."""
        true_configuration = ""
        if "[XE]" in configuration:
            true_configuration += "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)"
            true_configuration += configuration.split("[XE]")[1]
        elif "[RN]" in configuration:
            # Add Xenon
            true_configuration += "K(2)L(8)M(18)4S(2)4P(6)5S(2)4D(10)5P(6)"
            # Add Rn
            true_configuration += "4F(14)6S(2)5D(10)6P(6)"
            # Add rest
            true_configuration += configuration.split("[RN]")[1]
        else:
            raise ValueError("Heavy element is not the right format for parsing.")
        return true_configuration

    with open(file_name, "r", encoding="utf8") as f:
        line = f.readline()
        configuration = line.split()[1].replace(",", "")
        if is_heavy_element:
            configuration = _configuration_exact_for_heavy_elements(configuration)

        next_line = f.readline()
        # Sometimes there are blank lines.
        while len(next_line.strip()) == 0:
            next_line = f.readline()
        if is_heavy_element:
            # Heavy element slater files has extra redundant information of 5 lines.
            for _ in range(0, 6):
                next_line = f.readline()

        # Get energy from "E=..." line
        split_energy_line = next_line.split("=")
        if not split_energy_line[0].strip() == "E":
            raise RuntimeError("Parsing error of energy term 'E='.")
        energy = float(split_energy_line[1])

        # Split the kinetic, potential energy.
        split_energy_line = re.findall(r"[= -]\d+.\d+", f.readline())
        assert len(split_energy_line) == 3
        kinetic_energy = float(split_energy_line[0])
        potential_energy = float(split_energy_line[1])

        orbitals = []
        orbitals_basis = {'S': [], 'P': [], 'D': [], "F": []}
        orbitals_cusp = []
        orbitals_energy = []
        orbitals_exp = {'S': [], 'P': [], 'D': [], "F": []}
        orbitals_coeff = {}

        line = f.readline()
        while line.strip() == "":
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
                if not is_heavy_element:
                    # Heavy atoms slater files, doesn't have cusp values,.
                    line = f.readline()
                    orbitals_cusp.extend([float(x) for x in line.split()[1:]])
                line = f.readline()

                # Get Exponents, Coefficients, Orbital Basis
                while re.match(r'\A^\d' + subshell, line.lstrip()):

                    list_words = line.split()
                    orbitals_exp[subshell] += [float(list_words[1])]
                    orbitals_basis[subshell] += [list_words[0]]

                    for x in list_of_orbitals:
                        orbitals_coeff[x] += [float(list_words[_get_column(x)])]
                    line = f.readline()
            else:
                line = f.readline()

    data = {'configuration': configuration,
            "energy": energy,
            'kinetic_energy': kinetic_energy,
            'potential_energy': potential_energy,
            'orbitals': orbitals,
            'orbitals_energy': np.array(orbitals_energy)[:, None],
            'orbitals_cusp': np.array(orbitals_cusp)[:, None],
            'orbitals_basis': orbitals_basis,
            'orbitals_exp':
                {key: np.asarray(value).reshape(len(value), 1)
                 for key, value in orbitals_exp.items()
                 if value},
            'orbitals_coeff':
                {key: np.asarray(value).reshape(len(value), 1)
                 for key, value in orbitals_coeff.items() if value != []},
            'orbitals_occupation': np.array([_get_number_of_electrons_per_orbital(configuration)[k]
                                             for k in orbitals])[:, None],
            'basis_numbers':
                {key: np.asarray([[int(x[0])] for x in value])
                 for key, value in orbitals_basis.items() if len(value) != 0}
            }

    return data
