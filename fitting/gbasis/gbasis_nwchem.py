
import numpy as np

__all__ = ["fortran_float", "load_gbasis_nwchem_format"]


def fortran_float(s):
    '''Convert a string to a float. Works also with D before the mantissa'''
    return float(s.replace('D', 'E').replace('d', 'e'))


def load_gbasis_nwchem_format(filename):
    '''Load the atomic basis set(s) from a NWChem file containing the basis set for
       one or more atoms.

        **Arguments:**

        filename
            The path to the file containing the atomic basis set(s) in NWChem format.
            This file contains the basis set for one or more atoms.
            This file is usually obtained from https://bse.pnl.gov/bse/portal

        **Returns:**
            A dictionary

    '''
    from fitting.gbasis.gbasis import ContractedGaussianBasis

    f = open(filename)
    basis_atom_map = {}
    for line in f:
        # strip of comments and white space
        line = line[:line.find('#')].strip()
        if len(line) == 0:
            continue
        if line == 'END':
            break
        if line.startswith('BASIS'):
            continue
        words = line.split()
        if words[0].isalpha():
            assert len(words) == 2
            element = words[0].lower()
            shell_types = words[1].lower()
            contracted_basis = [ContractedGaussianBasis(shell_type, [], []) for shell_type in shell_types]
            basis = basis_atom_map.get(element)
            if basis is None:
                # Add the new atom to the dictionary
                basis_atom_map[element] = []
            basis_atom_map[element].extend(contracted_basis)
        else:
            # An extra primitive for the current contraction(s).
            primitive_exponent = fortran_float(words[0])
            primitive_coefficients = [fortran_float(w) for w in words[1:]]
            assert len(primitive_coefficients) == len(contracted_basis)
            for i, cb in enumerate(contracted_basis):
                cb.exponents.append(primitive_exponent)
                cb.coefficients.append(primitive_coefficients[i::len(contracted_basis)])
    f.close()
    # Turn the contracted exponents and coefficients lists into arrays
    for element in basis_atom_map.keys():
        for contracted_basis in basis_atom_map[element]:
            contracted_basis.to_arrays()
    return basis_atom_map
